"""Tests for benchmark answer evaluation (numeric, units, symbolic)."""

from __future__ import annotations

from solvay.benchmark.evaluation import evaluate_correct
from solvay.benchmark.schema import Expected, Verification


def _numeric(value: str, unit: str | None = None, tol: float = 0.01) -> Expected:
    return Expected(
        kind="numeric",
        value=value,
        unit=unit,
        tolerance_rel=tol,
        verification=Verification(method="numeric_eval", script=None),
    )


def _symbolic(value: str) -> Expected:
    return Expected(
        kind="symbolic",
        value=value,
        unit=None,
        tolerance_rel=0.01,
        verification=Verification(method="sympy_equivalence", script=None),
    )


class TestNumericEvaluation:
    def test_extracted_answer_within_tolerance(self) -> None:
        assert evaluate_correct("4.9 m/s^2", "ignored", _numeric("4.905", "m/s^2"))

    def test_extracted_answer_outside_tolerance(self) -> None:
        assert not evaluate_correct("5.5 m/s^2", "ignored", _numeric("4.905", "m/s^2"))

    def test_extracted_takes_precedence_over_raw(self) -> None:
        # Raw contains the right value but the declared final answer is wrong.
        raw = "Intermediate: 4.905, but I conclude differently."
        assert not evaluate_correct("7.2 m/s^2", raw, _numeric("4.905", "m/s^2"))

    def test_fallback_uses_last_number_not_any(self) -> None:
        # Statement data (30, 9.81) and intermediates must not match; only the
        # final number in the text counts when there is no extracted answer.
        raw = "With theta=30 deg and g=9.81, the acceleration is 7.2 m/s^2"
        assert not evaluate_correct(None, raw, _numeric("9.81"))
        raw_correct = "With theta=30 deg, a = g*sin(30) = 4.905"
        assert evaluate_correct(None, raw_correct, _numeric("4.905"))

    def test_unit_conversion_cm_to_m(self) -> None:
        assert evaluate_correct("490.5 cm/s^2", "ignored", _numeric("4.905", "m/s^2"))

    def test_wrong_magnitude_in_other_unit_fails(self) -> None:
        assert not evaluate_correct("4.905 cm/s^2", "ignored", _numeric("4.905", "m/s^2"))

    def test_unparseable_unit_falls_back_to_bare_number(self) -> None:
        assert evaluate_correct("4.905 furlongs??", "ignored", _numeric("4.905", "m/s^2"))

    def test_expected_zero_uses_absolute_tolerance(self) -> None:
        assert evaluate_correct("0.0001", "ignored", _numeric("0", None, tol=0.01))
        assert not evaluate_correct("0.5", "ignored", _numeric("0", None, tol=0.01))

    def test_no_numbers_anywhere_is_incorrect(self) -> None:
        assert not evaluate_correct(None, "no numerals here", _numeric("4.905"))


class TestSymbolicEvaluation:
    def test_equivalent_expressions_match(self) -> None:
        assert evaluate_correct("sin(theta)*g", "ignored", _symbolic("g*sin(theta)"))

    def test_non_equivalent_expressions_fail(self) -> None:
        assert not evaluate_correct("g*cos(theta)", "ignored", _symbolic("g*sin(theta)"))

    def test_fallback_uses_last_nonempty_line_of_raw(self) -> None:
        raw = "We derive the result.\n\ng*sin(theta)\n"
        assert evaluate_correct(None, raw, _symbolic("g*sin(theta)"))

    def test_unparseable_candidate_is_incorrect(self) -> None:
        assert not evaluate_correct("the answer is gravity-ish", "x", _symbolic("g*sin(theta)"))
