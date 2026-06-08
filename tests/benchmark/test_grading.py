"""Tests for the two-stage grading module (SymPy + LLM-judge)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.grading import (
    _check_numeric_match,
    _extract_boxed_numbers,
    _extract_display_math_numbers,
    _extract_latex_expressions,
    _latex_to_sympy_str,
    _safe_sympify,
    evaluate_correct,
    llm_grade,
    sympy_grade,
)
from solvay.benchmark.schema import Expected, Problem, Verification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_expected(value: str, tolerance_rel: float = 0.01) -> Expected:
    return Expected(
        kind="numeric",
        value=value,
        unit="m/s^2",
        tolerance_rel=tolerance_rel,
        verification=Verification(method="numeric_eval", script=None),
    )


def _symbolic_expected(value: str) -> Expected:
    return Expected(
        kind="symbolic",
        value=value,
        unit=None,
        tolerance_rel=0.01,
        verification=Verification(method="sympy_equivalence", script=None),
    )


def _fake_problem() -> Problem:
    return Problem.model_validate({
        "id": "test-001",
        "version": 1,
        "source": {
            "kind": "synthetic",
            "origin": "test",
            "generated_at": "2026-01-01T00:00:00Z",
            "seed": 0,
        },
        "domain": "mechanics",
        "statement": "Find the acceleration.",
        "given": {"m": "2 kg"},
        "find": "acceleration",
        "expected": {
            "kind": "numeric",
            "value": "2",
            "unit": "m/s^2",
            "tolerance_rel": 0.01,
            "verification": {"method": "numeric_eval", "script": None},
        },
    })


def _default_config() -> BenchConfig:
    return BenchConfig()


# ---------------------------------------------------------------------------
# TestSympyGrade
# ---------------------------------------------------------------------------

class TestSympyGrade:
    def test_numeric_exact_match(self):
        expected = _numeric_expected("4.905")
        result = sympy_grade("The answer is 4.905 m/s^2.", expected)
        assert result is True

    def test_numeric_within_tolerance(self):
        # 4.905 * 1.005 is within 1 % of 4.905
        close_value = 4.905 * 1.005
        expected = _numeric_expected("4.905", tolerance_rel=0.01)
        result = sympy_grade(f"Approximately {close_value:.4f} m/s^2.", expected)
        assert result is True

    def test_numeric_outside_tolerance(self):
        # 5.5 is > 1 % away from 4.905
        expected = _numeric_expected("4.905", tolerance_rel=0.01)
        result = sympy_grade("The answer is 5.5 m/s^2.", expected)
        assert result is False

    def test_symbolic_equivalent(self):
        # x*y == y*x (commutativity)
        expected = _symbolic_expected("x*y")
        result = sympy_grade("The result is y*x.", expected)
        assert result is True

    def test_symbolic_different_form(self):
        # x + y != x - y
        expected = _symbolic_expected("x + y")
        result = sympy_grade("The result is x - y.", expected)
        assert result is False

    def test_unparseable_expected_returns_none(self):
        # expected.value is gibberish that sympy cannot parse
        expected = Expected(
            kind="numeric",
            value="%%%NOT_A_NUMBER%%%",
            unit="m",
            tolerance_rel=0.01,
            verification=Verification(method="numeric_eval", script=None),
        )
        result = sympy_grade("The answer is 42.", expected)
        assert result is None

    def test_no_numbers_returns_none(self):
        expected = _numeric_expected("4.905")
        result = sympy_grade("No numerical value is stated here.", expected)
        assert result is None

    def test_symbolic_latex_frac(self):
        expected = _symbolic_expected("k_B*T/d**3")
        answer = r"""\[P \sim \frac{k_B T}{d^3}\]"""
        assert sympy_grade(answer, expected) is True

    def test_symbolic_latex_negative_sign_scaling(self):
        expected = _symbolic_expected("k_B*T/d**3")
        answer = r"""\[P \sim -\frac{k_B T}{d^3}.\]"""
        assert sympy_grade(answer, expected) is True

    def test_symbolic_boxed_with_frac(self):
        expected = _symbolic_expected("k_B*T/d**3")
        answer = r"""\boxed{\frac{k_B T}{d^3}}"""
        assert sympy_grade(answer, expected) is True

    def test_symbolic_latex_gamma_power(self):
        expected = _symbolic_expected("gamma**2")
        answer = r"""\[\omega_c \sim \gamma^2 \omega_B\]"""
        result = sympy_grade(answer, expected)
        assert result is True or result is None


class TestLatexConversion:
    def test_frac(self):
        result = _latex_to_sympy_str(r"\frac{a}{b}")
        assert "a" in result and "b" in result and "/" in result

    def test_sim_strips_lhs(self):
        result = _latex_to_sympy_str(r"P \sim \frac{k_B T}{d^3}")
        assert "P" not in result.split("*")

    def test_greek_letters(self):
        result = _latex_to_sympy_str(r"\gamma^2 \omega_B")
        assert "gamma" in result and "omega" in result

    def test_extract_boxed_nested_braces(self):
        exprs = _extract_latex_expressions(r"\boxed{\frac{a}{b}}")
        assert len(exprs) >= 1
        assert "a" in exprs[0] and "b" in exprs[0]

    def test_extract_display_math(self):
        text = r"before \[x + y\] after"
        exprs = _extract_latex_expressions(text)
        assert len(exprs) >= 1


# ---------------------------------------------------------------------------
# TestPrioritizedNumericGrading
# ---------------------------------------------------------------------------

class TestPrioritizedNumericGrading:
    """Tests for prioritised number extraction (boxed > display math > text)."""

    def test_boxed_answer_wins_over_incidental_match(self):
        """The exact bug: derivation mentions '3' but boxed answer is '2'."""
        answer = (
            r"The allowed wavevectors are k_m, m=1,2,3,\dots "
            r"The force scales as 1/d^2. \[\boxed{n = 2}.\]"
        )
        expected = _numeric_expected("3")
        assert sympy_grade(answer, expected) is False

    def test_boxed_correct_answer(self):
        answer = r"After calculation, \boxed{n = 3}."
        expected = _numeric_expected("3")
        assert sympy_grade(answer, expected) is True

    def test_display_math_wins_over_text_numbers(self):
        answer = r"We try 1, 2, 3 approaches. The final result is \[n = 5.\]"
        expected = _numeric_expected("3")
        assert sympy_grade(answer, expected) is False

    def test_display_math_correct(self):
        answer = r"The exponent is \[n = 4.\]"
        expected = _numeric_expected("4")
        assert sympy_grade(answer, expected) is True

    def test_fallback_to_all_numbers_when_no_math(self):
        answer = "The answer is 7."
        expected = _numeric_expected("7")
        assert sympy_grade(answer, expected) is True

    def test_extract_boxed_numbers(self):
        text = r"Some text \boxed{n = 42} more text"
        assert "42" in _extract_boxed_numbers(text)

    def test_extract_display_math_numbers(self):
        text = r"Before \[x = 99\] after"
        assert "99" in _extract_display_math_numbers(text)

    def test_extract_display_math_dollar(self):
        text = "Before $$y = 77$$ after"
        assert "77" in _extract_display_math_numbers(text)

    def test_check_numeric_match_relative(self):
        assert _check_numeric_match(["9.81"], 9.81, 0.01) is True
        assert _check_numeric_match(["10.0"], 9.81, 0.01) is False

    def test_check_numeric_match_zero_target(self):
        assert _check_numeric_match(["0.005"], 0.0, 0.01) is True
        assert _check_numeric_match(["0.5"], 0.0, 0.01) is False


# ---------------------------------------------------------------------------
# TestSafeSympify
# ---------------------------------------------------------------------------

class TestSafeSympify:
    """Tests for _safe_sympify handling SymPy namespace collisions."""

    def test_Q_as_symbol(self):
        expr = _safe_sympify("hbar*c_s**2/(k_B*Q)")
        assert expr is not None
        assert str(expr) == "c_s**2*hbar/(Q*k_B)"

    def test_reserved_single_letter_symbols(self):
        for name in ["Q", "S", "N", "E", "I", "O"]:
            expr = _safe_sympify(f"x*{name}")
            assert expr is not None, f"Failed to parse 'x*{name}'"

    def test_symbolic_grade_with_Q_variable(self):
        expected = _symbolic_expected("hbar*c_s**2/(k_B*Q)")
        answer = r"\[\frac{\hbar c_s^2}{k_B Q}\]"
        assert sympy_grade(answer, expected) is True

    def test_symbolic_grade_with_Q_wrong_answer(self):
        expected = _symbolic_expected("hbar*c_s**2/(k_B*Q)")
        answer = r"\[\frac{\hbar c_s}{k_B Q}\]"
        assert sympy_grade(answer, expected) is False


# ---------------------------------------------------------------------------
# TestLlmGrade
# ---------------------------------------------------------------------------

class TestLlmGrade:
    def _mock_llm(self, content: str):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=content)
        return mock_model

    def test_correct_verdict(self):
        problem = _fake_problem()
        config = _default_config()
        expected = _numeric_expected("9.81")
        llm_response = '{"correct": true, "reason": "match"}'

        with patch(
            "solvay.benchmark.grading.init_chat_model",
            return_value=self._mock_llm(llm_response),
        ):
            result = llm_grade("9.81 m/s^2", expected, problem, config)

        assert result is True

    def test_incorrect_verdict(self):
        problem = _fake_problem()
        config = _default_config()
        expected = _numeric_expected("9.81")
        llm_response = '{"correct": false, "reason": "wrong value"}'

        with patch(
            "solvay.benchmark.grading.init_chat_model",
            return_value=self._mock_llm(llm_response),
        ):
            result = llm_grade("3.14 m/s^2", expected, problem, config)

        assert result is False

    def test_unparseable_response_returns_false(self):
        problem = _fake_problem()
        config = _default_config()
        expected = _numeric_expected("9.81")
        # LLM returns something that is not valid JSON
        llm_response = "I am not sure."

        with patch(
            "solvay.benchmark.grading.init_chat_model",
            return_value=self._mock_llm(llm_response),
        ):
            result = llm_grade("9.81 m/s^2", expected, problem, config)

        assert result is False


# ---------------------------------------------------------------------------
# TestEvaluateCorrect
# ---------------------------------------------------------------------------

class TestEvaluateCorrect:
    def test_uses_sympy_when_parseable(self):
        problem = _fake_problem()
        config = _default_config()
        expected = _numeric_expected("9.81")

        correct, method = evaluate_correct("9.81 m/s^2", expected, problem, config)

        assert correct is True
        assert method == "sympy"

    def test_falls_back_to_llm_when_sympy_returns_none(self):
        problem = _fake_problem()
        config = _default_config()
        # No number in answer_raw -> sympy_grade returns None
        expected = _numeric_expected("9.81")
        llm_response = '{"correct": true, "reason": "semantically correct"}'

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=llm_response)

        with patch(
            "solvay.benchmark.grading.init_chat_model",
            return_value=mock_model,
        ):
            correct, method = evaluate_correct(
                "approximately ten metres per second squared",
                expected,
                problem,
                config,
            )

        assert correct is True
        assert method == "llm-judge"
