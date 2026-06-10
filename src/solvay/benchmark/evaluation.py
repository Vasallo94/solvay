"""Answer evaluation: numeric with unit normalization, and symbolic equivalence.

The evaluator prefers the answer the model explicitly declared (the
``FINAL ANSWER:`` line extracted by the profile). Only when no declared
answer exists does it fall back to the LAST number in the raw text — never
"any number anywhere", which produced false positives from statement data
and intermediate steps.
"""

from __future__ import annotations

import re
from functools import cache

import sympy

from solvay.benchmark.schema import Expected

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@cache
def _units_namespace() -> dict[str, object]:
    import sympy.physics.units as units

    namespace: dict[str, object] = {}
    namespace.update(vars(sympy))
    namespace.update(vars(units))
    return namespace


def _parse_unit(unit_str: str) -> sympy.Expr | None:
    try:
        parsed = sympy.sympify(unit_str, locals=dict(_units_namespace()))
    except Exception:
        return None
    if not isinstance(parsed, sympy.Expr) or parsed.free_symbols:
        return None
    return parsed


def _unit_conversion_factor(candidate_unit: str, expected_unit: str) -> float | None:
    """Return the factor that converts candidate_unit into expected_unit."""
    from sympy.physics.units import convert_to

    cand = _parse_unit(candidate_unit)
    exp = _parse_unit(expected_unit)
    if cand is None or exp is None:
        return None
    try:
        ratio = sympy.simplify(convert_to(cand, exp) / exp)
        if not ratio.is_number:
            return None
        return float(ratio)
    except Exception:
        return None


def _numeric_target(expected: Expected) -> float | None:
    try:
        return float(sympy.sympify(expected.value).evalf())
    except (ValueError, TypeError, sympy.SympifyError):
        return None


def _within_tolerance(candidate: float, target: float, tolerance_rel: float) -> bool:
    if target == 0:
        return abs(candidate) <= tolerance_rel
    return abs(candidate - target) / abs(target) <= tolerance_rel


def _evaluate_numeric(
    answer_extracted: str | None,
    answer_raw: str,
    expected: Expected,
) -> bool:
    target = _numeric_target(expected)
    if target is None:
        return False

    if answer_extracted is not None:
        match = _NUMBER_RE.search(answer_extracted)
        if match is None:
            return False
        candidate = float(match.group())
        unit_text = answer_extracted[match.end() :].strip()
        if unit_text and expected.unit:
            factor = _unit_conversion_factor(unit_text, expected.unit)
            if factor is not None:
                candidate *= factor
        return _within_tolerance(candidate, target, expected.tolerance_rel)

    # Fallback: the LAST number in the raw text, on the assumption that
    # answers conclude with the result. No unit handling in this mode.
    numbers = _NUMBER_RE.findall(answer_raw)
    if not numbers:
        return False
    try:
        candidate = float(numbers[-1])
    except ValueError:
        return False
    return _within_tolerance(candidate, target, expected.tolerance_rel)


def _evaluate_symbolic(
    answer_extracted: str | None,
    answer_raw: str,
    expected: Expected,
) -> bool:
    if answer_extracted is not None:
        candidate_text = answer_extracted
    else:
        lines = [line.strip() for line in answer_raw.splitlines() if line.strip()]
        if not lines:
            return False
        candidate_text = lines[-1]
    candidate_text = candidate_text.strip("$`").replace("\\", "")

    try:
        expected_expr = sympy.sympify(expected.value)
        candidate_expr = sympy.sympify(candidate_text)
        return bool(sympy.simplify(expected_expr - candidate_expr) == 0)
    except Exception:
        return False


def evaluate_correct(
    answer_extracted: str | None,
    answer_raw: str,
    expected: Expected,
) -> bool:
    """Judge a benchmark answer against the expected value.

    Args:
        answer_extracted: The declared FINAL ANSWER line, if the profile
            extracted one. Takes precedence over the raw text.
        answer_raw: The full model response (fallback evaluation source).
        expected: The problem's expected answer specification.

    Returns:
        True when the answer matches within tolerance (numeric, with unit
        normalization when both units parse) or is symbolically equivalent.
    """
    if expected.kind == "symbolic":
        return _evaluate_symbolic(answer_extracted, answer_raw, expected)
    return _evaluate_numeric(answer_extracted, answer_raw, expected)
