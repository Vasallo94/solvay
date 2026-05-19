"""Deterministic verification of generator output.

This verifier does NOT re-solve the problem from the statement (that would
require NLP-level understanding). Instead it verifies the *coherence* of the
generator's own output: that the answer is parseable in the declared
mode, that the declared units exist, and that numeric answers are
within tolerance of themselves when re-evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.physics.units import convert_to  # noqa: F401  (kept for future method)
from sympy.physics.units.systems import SI

from solvay.benchmark.schema import Problem


@dataclass(frozen=True)
class VerifyOutcome:
    ok: bool
    method: str
    reason: str | None = None


def _sympify_with_given(expr: str, given: dict[str, str]) -> sympy.Expr:
    namespace: dict[str, object] = {k: sympy.Symbol(k) for k in given}
    namespace["sqrt"] = sympy.sqrt
    result = sympy.sympify(expr, locals=namespace)
    if not isinstance(result, sympy.Expr):
        raise ValueError(f"not a sympy expression: {result!r}")
    allowed = set(given.keys())
    for symbol in result.free_symbols:
        if str(symbol) not in allowed:
            raise ValueError(f"unknown symbol: {symbol}")
    return result


def _unit_is_known(unit_str: str) -> bool:
    known_names: set[str] = set()
    ns: dict[str, object] = {}
    for u in SI.get_units_non_prefixed():
        name = str(u)
        ns[name] = u
        known_names.add(name)
        try:
            abbrev = str(u.abbrev)
            ns.setdefault(abbrev, u)
            known_names.add(abbrev)
        except Exception:
            pass
    try:
        parsed = sympy.sympify(unit_str, locals=ns, evaluate=False)
    except Exception:
        return False
    if not hasattr(parsed, "atoms"):
        return False
    return all(str(symbol) in known_names for symbol in parsed.atoms(sympy.Symbol))


def verify_problem(problem: Problem) -> VerifyOutcome:
    method = problem.expected.verification.method
    try:
        if method == "sympy_equivalence":
            _sympify_with_given(problem.expected.value, problem.given)
            return VerifyOutcome(ok=True, method=method)
        if method == "numeric_eval":
            float(sympy.sympify(problem.expected.value).evalf())
            return VerifyOutcome(ok=True, method=method)
        if method == "dimensional_only":
            if problem.expected.unit is None:
                return VerifyOutcome(ok=False, method=method, reason="unit missing")
            if not _unit_is_known(problem.expected.unit):
                return VerifyOutcome(
                    ok=False, method=method, reason=f"unknown unit: {problem.expected.unit}"
                )
            return VerifyOutcome(ok=True, method=method)
    except Exception as exc:
        return VerifyOutcome(ok=False, method=method, reason=str(exc)[:200])
    return VerifyOutcome(ok=False, method=method, reason=f"unknown method: {method}")
