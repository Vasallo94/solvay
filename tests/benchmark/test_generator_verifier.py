"""Tests for the symbolic/numeric verifier."""

from __future__ import annotations

from solvay.benchmark.generator.verifier import VerifyOutcome, verify_problem
from solvay.benchmark.schema import Problem


def _make(value: str, method: str, kind: str = "symbolic", unit: str | None = None) -> Problem:
    return Problem.model_validate(
        {
            "id": "t",
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "t",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 0,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "a",
            "given": {"g": "9.81", "L": "1"},
            "find": "omega",
            "expected": {
                "kind": kind,
                "value": value,
                "unit": unit,
                "tolerance_rel": 0.01,
                "verification": {"method": method, "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_sympy_equivalence_accepts_parseable_answer() -> None:
    p = _make("sqrt(g/L)", "sympy_equivalence", kind="symbolic")
    outcome = verify_problem(p)
    assert outcome.ok
    assert outcome.method == "sympy_equivalence"


def test_sympy_equivalence_rejects_unparseable_answer() -> None:
    p = _make("this is not sympy", "sympy_equivalence", kind="symbolic")
    outcome = verify_problem(p)
    assert not outcome.ok


def test_numeric_eval_passes_when_value_is_number() -> None:
    p = _make("3.13", "numeric_eval", kind="numeric")
    outcome = verify_problem(p)
    assert outcome.ok


def test_dimensional_only_passes_when_unit_is_known() -> None:
    p = _make("g / L", "dimensional_only", kind="symbolic", unit="1/second**2")
    outcome = verify_problem(p)
    assert outcome.ok


def test_dimensional_only_rejects_unknown_unit() -> None:
    p = _make("g / L", "dimensional_only", kind="symbolic", unit="made_up_unit")
    outcome = verify_problem(p)
    assert not outcome.ok
    assert isinstance(outcome, VerifyOutcome)
