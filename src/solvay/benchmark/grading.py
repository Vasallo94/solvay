"""Two-stage physics answer grader: SymPy equivalence check with LLM-judge fallback."""

from __future__ import annotations

import json
import re

import sympy
from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Expected, Problem

# Regex patterns used for extraction
_NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_SYMBOLIC_PATTERN = re.compile(
    r"[a-zA-Z_]\w*(?:\s*[\*\+\-/\^]\s*[a-zA-Z_\d\.\(\)]*)+"
)
_FRAC_PATTERN = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_]\w*")


def _match_braces(text: str, start: int) -> str:
    """Extract content between matched braces starting at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return text[start + 1 :]


def _latex_to_sympy_str(latex: str) -> str:
    """Best-effort conversion from LaTeX math to a SymPy-parseable string."""
    s = latex.strip()
    parts = re.split(r"\\(?:sim|approx|propto|simeq)\s*", s, maxsplit=1)
    if len(parts) == 2:
        s = parts[1]
    for cmd in ["partial", "nabla", "infty", "alpha", "beta", "theta", "phi",
                "psi", "omega", "Omega", "pi", "epsilon", "delta", "Delta",
                "sigma", "Sigma", "lambda", "Lambda", "mu", "nu", "rho", "tau",
                "kappa", "chi", "eta", "xi", "zeta"]:
        s = s.replace(f"\\{cmd}", cmd)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    s = re.sub(r"\\(?:text|mathrm|mathbf|boldsymbol|hat|vec)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac")
    while _FRAC_PATTERN.search(s):
        s = _FRAC_PATTERN.sub(r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\^{([^}]+)}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    s = re.sub(r"_{([^}]+)}", r"_\1", s)
    s = s.replace("\\", "")
    lhs_split = re.split(r"\s*=\s*", s, maxsplit=1)
    if len(lhs_split) == 2:
        s = lhs_split[1]
    s = re.sub(r"(?<=\w)\s+(?=\w)", "*", s)
    s = re.sub(r"(?<=\))(?=[a-zA-Z(])", "*", s)
    s = re.sub(r"(?<=\d)(?=[a-zA-Z])", "*", s)
    _KNOWN_FUNCS = {"sqrt", "sin", "cos", "tan", "exp", "log", "ln", "abs"}
    for fn in _KNOWN_FUNCS:
        s = s.replace(f"{fn}*", f"{fn}")
    return s


def _extract_latex_expressions(text: str) -> list[str]:
    """Extract LaTeX math expressions from model output."""
    candidates: list[str] = []
    for m in re.finditer(r"\\boxed\{", text):
        inner = _match_braces(text, m.end() - 1)
        candidates.append(_latex_to_sympy_str(inner))
    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        candidates.append(_latex_to_sympy_str(m.group(1)))
    for m in re.finditer(r"\$\$([^$]+)\$\$", text):
        candidates.append(_latex_to_sympy_str(m.group(1)))
    for m in re.finditer(r"(?<!\$)\$(?!\$)([^$]+)\$(?!\$)", text):
        inner = m.group(1).strip()
        if any(c in inner for c in ["\\frac", "^", "_", "\\cdot"]):
            candidates.append(_latex_to_sympy_str(inner))
    return candidates


def sympy_grade(answer_raw: str, expected: Expected) -> bool | None:
    """Attempt to grade answer_raw against expected using SymPy.

    Returns:
        True  – answer is correct.
        False – answer is definitively wrong.
        None  – could not determine (no parseable expression found, or
                expected itself could not be parsed).
    """
    if expected.kind == "numeric":
        return _grade_numeric(answer_raw, expected)
    elif expected.kind == "symbolic":
        return _grade_symbolic(answer_raw, expected)
    return None


def _extract_boxed_numbers(text: str) -> list[str]:
    """Extract numeric strings from \\boxed{...} expressions."""
    results: list[str] = []
    for m in re.finditer(r"\\boxed\{", text):
        inner = _match_braces(text, m.end() - 1)
        results.extend(_NUMERIC_PATTERN.findall(inner))
    return results


def _extract_display_math_numbers(text: str) -> list[str]:
    """Extract numeric strings from display math (\\[...\\] and $$...$$)."""
    results: list[str] = []
    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        results.extend(_NUMERIC_PATTERN.findall(m.group(1)))
    for m in re.finditer(r"\$\$([^$]+)\$\$", text):
        results.extend(_NUMERIC_PATTERN.findall(m.group(1)))
    return results


def _check_numeric_match(candidates: list[str], target: float, tol: float) -> bool:
    """Check whether any candidate number matches target within tolerance."""
    for num_str in candidates:
        try:
            candidate = float(num_str)
        except ValueError:
            continue
        if target == 0.0:
            if abs(candidate) <= tol:
                return True
        else:
            if abs(candidate - target) / abs(target) <= tol:
                return True
    return False


def _grade_numeric(answer_raw: str, expected: Expected) -> bool | None:
    """Grade a numeric answer using prioritised extraction.

    Priority: boxed > display math > all numbers in text.
    Once candidates are found at a given tier, only those are checked.
    """
    try:
        target = float(sympy.sympify(expected.value).evalf())
    except Exception:
        return None

    tol = expected.tolerance_rel

    boxed = _extract_boxed_numbers(answer_raw)
    if boxed:
        return _check_numeric_match(boxed, target, tol)

    display = _extract_display_math_numbers(answer_raw)
    if display:
        return _check_numeric_match(display, target, tol)

    all_numbers = _NUMERIC_PATTERN.findall(answer_raw)
    if not all_numbers:
        return None

    return _check_numeric_match(all_numbers, target, tol)


def _safe_sympify(expr_str: str) -> sympy.Expr | None:
    """Parse expr_str into a SymPy expression, shielding variable names
    that collide with SymPy built-ins (Q, S, N, I, E, O, ...).
    """
    _SYMPY_FUNCS = {"sqrt", "sin", "cos", "tan", "exp", "log", "ln", "abs",
                    "pi", "oo", "zoo", "nan", "true", "false"}
    identifiers = set(_IDENTIFIER_PATTERN.findall(expr_str))
    local_dict = {
        name: sympy.Symbol(name)
        for name in identifiers
        if name not in _SYMPY_FUNCS
    }
    try:
        return sympy.sympify(expr_str, locals=local_dict)
    except Exception:
        return None


def _grade_symbolic(answer_raw: str, expected: Expected) -> bool | None:
    """Grade a symbolic answer."""
    expected_expr = _safe_sympify(expected.value)
    if expected_expr is None:
        return None

    candidates: list[str] = []

    for expr_str in _extract_latex_expressions(answer_raw):
        candidates.append(expr_str)

    for match in _SYMBOLIC_PATTERN.finditer(answer_raw):
        candidates.append(match.group())

    for match in _NUMERIC_PATTERN.finditer(answer_raw):
        candidates.append(match.group())

    if not candidates:
        return None

    any_parsed = False
    for raw_candidate in candidates:
        raw_candidate = raw_candidate.rstrip(".,;:!? ")
        normalised = raw_candidate.replace("^", "**")
        for expr_str in [normalised, _strip_subscript_zeros(normalised)]:
            candidate_expr = _safe_sympify(expr_str)
            if candidate_expr is None:
                continue
            any_parsed = True

            try:
                diff = sympy.simplify(candidate_expr - expected_expr)
                if diff == 0:
                    return True
                ratio = sympy.simplify(candidate_expr / expected_expr)
                if ratio.is_number and abs(complex(ratio)) == 1:
                    return True
            except Exception:
                continue

    if not any_parsed:
        return None

    return False


def _strip_subscript_zeros(expr_str: str) -> str:
    """Normalize variable names: B_0 -> B0, omega_0 -> omega0, etc."""
    return re.sub(r"(\w)_(\d+)", r"\1\2", expr_str)


def llm_grade(
    answer_raw: str,
    expected: Expected,
    problem: Problem,
    config: BenchConfig,
) -> bool:
    """Grade answer_raw using an LLM as judge.

    Returns True if the LLM decides the answer is correct, False otherwise.
    Unparseable LLM responses are treated as incorrect (False).
    """
    grader_model = getattr(config, "grader_model", "fake-model")
    llm = init_chat_model(grader_model)

    unit_str = expected.unit or ""
    prompt = (
        "Compare this physics answer to the expected answer.\n\n"
        f"Problem: {problem.statement}\n"
        f"Student answer: {answer_raw}\n"
        f"Expected: {expected.value} {unit_str}\n\n"
        'Are they mathematically equivalent? Return JSON: {"correct": true/false, "reason": "..."}'
    )

    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        return bool(data["correct"])
    except Exception:
        return False


def evaluate_correct(
    answer_raw: str,
    expected: Expected,
    problem: Problem,
    config: BenchConfig,
) -> tuple[bool, str]:
    """Grade an answer using SymPy first, falling back to LLM-judge.

    Returns:
        (correct, grading_method) where grading_method is "sympy" or "llm-judge".
    """
    sympy_result = sympy_grade(answer_raw, expected)
    if sympy_result is not None:
        return (sympy_result, "sympy")

    llm_result = llm_grade(answer_raw, expected, problem, config)
    return (llm_result, "llm-judge")
