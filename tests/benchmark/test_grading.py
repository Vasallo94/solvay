"""Tests for the two-stage grading module (SymPy + LLM-judge)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.grading import evaluate_correct, llm_grade, sympy_grade
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
