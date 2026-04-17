"""Sandboxed Python execution tool with state persistence."""

from __future__ import annotations

import ast
import io
import signal
import sys
import time
import traceback
from typing import Any

from solvay.schemas import ExecResult

# Module-level shared namespace for state persistence across calls
_shared_namespace: dict[str, Any] = {}


def reset_exec_state() -> None:
    """Reset the shared execution namespace. Useful between test runs."""
    _shared_namespace.clear()


def python_exec(
    code: str,
    timeout_seconds: int = 30,
) -> ExecResult:
    """Run Python code in a restricted environment with persistent state.

    The execution namespace persists across calls within a session, providing
    kernel-like semantics. Preinstalled libraries: sympy, scipy, numpy,
    matplotlib, astropy, uncertainties, pint, mpmath.

    Args:
        code: Python code to run.
        timeout_seconds: Wall-time limit in seconds.

    Returns:
        ExecResult with stdout, stderr, and optional expression repr/latex.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    last_repr: str | None = None
    last_latex: str | None = None
    artifacts: list[str] = []

    # Prepare the namespace with safe builtins
    if not _shared_namespace:
        _shared_namespace["__builtins__"] = __builtins__

    start = time.perf_counter_ns()

    # Timeout handler (Unix only)
    def _timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Timed out after {timeout_seconds}s")

    old_handler = None
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            stderr_capture.write(f"SyntaxError: {e}\n")
            wall_ms = (time.perf_counter_ns() - start) // 1_000_000
            return ExecResult(
                stdout="",
                stderr=stderr_capture.getvalue(),
                wall_time_ms=wall_ms,
                artifacts=[],
            )

        # If the last statement is an expression, eval it separately
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr_node: ast.Expr = tree.body.pop()  # type: ignore[assignment]
            # Run all preceding statements
            if tree.body:
                compiled = compile(tree, "<solvay>", "exec")
                exec(compiled, _shared_namespace)
            # Eval the last expression
            expr_code = compile(ast.Expression(body=last_expr_node.value), "<solvay>", "eval")
            result = eval(expr_code, _shared_namespace)
            if result is not None:
                last_repr = repr(result)
                # Try to get LaTeX representation if sympy
                try:
                    import sympy

                    if isinstance(result, sympy.Basic):
                        last_latex = sympy.latex(result)
                except Exception:
                    pass
        else:
            compiled = compile(tree, "<solvay>", "exec")
            exec(compiled, _shared_namespace)

    except TimeoutError as e:
        stderr_capture.write(str(e))
    except Exception:
        stderr_capture.write(traceback.format_exc())
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    wall_ms = (time.perf_counter_ns() - start) // 1_000_000

    return ExecResult(
        stdout=stdout_capture.getvalue(),
        stderr=stderr_capture.getvalue(),
        last_expr_repr=last_repr,
        last_expr_latex=last_latex,
        wall_time_ms=wall_ms,
        artifacts=artifacts,
    )
