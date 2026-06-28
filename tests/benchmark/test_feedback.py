"""Tests for AFP friction detectors in the benchmark pipeline."""

from __future__ import annotations

import pytest

# The AFP feedback pipeline depends on the optional `afp` package
# (Agent Feedback Protocol). cli.py already imports it lazily under a
# try/except; skip this whole module when afp is not installed so the
# suite still collects.
pytest.importorskip("afp")

from solvay.benchmark.feedback import (  # noqa: E402
    analyze_run,
    detect_grading_fallback,
    detect_inconsistent_grading,
    detect_unanimous_disagreement,
)
from solvay.benchmark.jsonl import RunRecord
from solvay.benchmark.schema import Problem


def _problem(pid: str = "test-001", expected_value: str = "3", kind: str = "numeric") -> Problem:
    return Problem.model_validate(
        {
            "id": pid,
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "test",
                "generated_at": "2026-01-01T00:00:00Z",
                "seed": 0,
            },
            "domain": "mechanics",
            "statement": "Find the answer.",
            "given": {},
            "find": "answer",
            "expected": {
                "kind": kind,
                "value": expected_value,
                "unit": None,
                "tolerance_rel": 0.01,
                "verification": {
                    "method": "numeric_eval" if kind == "numeric" else "sympy_equivalence",
                    "script": None,
                },
            },
        }
    )


def _record(
    problem_id: str = "test-001",
    profile: str = "bare",
    correct: bool = True,
    grading_method: str = "sympy",
    answer_raw: str = "The answer is 3.",
    error: str | None = None,
) -> RunRecord:
    return RunRecord(
        problem_id=problem_id,
        profile=profile,
        model="test-model",
        repeat_idx=0,
        answer_raw=answer_raw,
        answer_extracted=None,
        correct=correct,
        elapsed_seconds=1.0,
        grading_method=grading_method,
        error=error,
    )


# --- Detector A: grading fallback ---


class TestDetectGradingFallback:
    """Mirrors Bug 1: SymPy can't parse expected value (e.g. Q namespace collision)."""

    def test_flags_llm_judge_fallback(self):
        records = [_record(grading_method="llm-judge", correct=False)]
        problems = {"test-001": _problem(expected_value="hbar*c_s/(k_B*Q)")}
        reports = detect_grading_fallback(records, problems)
        assert len(reports) == 1
        assert reports[0].friction_type.value == "bug"
        assert "hbar*c_s/(k_B*Q)" in reports[0].observed

    def test_ignores_sympy_grading(self):
        records = [_record(grading_method="sympy")]
        reports = detect_grading_fallback(records, {"test-001": _problem()})
        assert len(reports) == 0

    def test_ignores_error_records(self):
        records = [_record(grading_method="llm-judge", error="timeout")]
        reports = detect_grading_fallback(records, {"test-001": _problem()})
        assert len(reports) == 0

    def test_deduplicates_same_problem(self):
        records = [
            _record(profile="bare", grading_method="llm-judge"),
            _record(profile="prompted", grading_method="llm-judge"),
        ]
        reports = detect_grading_fallback(records, {"test-001": _problem()})
        assert len(reports) == 1


# --- Detector B: inconsistent grading ---


class TestDetectInconsistentGrading:
    """Mirrors Bug 2: same n=2 answer graded True for prompted, False for bare."""

    def test_flags_same_answer_different_grades(self):
        records = [
            _record(profile="bare", correct=False, answer_raw="n = 2"),
            _record(profile="prompted", correct=True, answer_raw="n = 2"),
        ]
        reports = detect_inconsistent_grading(records)
        assert len(reports) == 1
        assert reports[0].friction_type.value == "wrong_output"
        assert reports[0].severity.value == "blocked"

    def test_ignores_consistent_grades(self):
        records = [
            _record(profile="bare", correct=True, answer_raw="42"),
            _record(profile="prompted", correct=True, answer_raw="42"),
        ]
        reports = detect_inconsistent_grading(records)
        assert len(reports) == 0

    def test_ignores_different_answers(self):
        records = [
            _record(profile="bare", correct=False, answer_raw="answer A"),
            _record(profile="prompted", correct=True, answer_raw="answer B"),
        ]
        reports = detect_inconsistent_grading(records)
        assert len(reports) == 0


# --- Detector C: unanimous disagreement ---


class TestDetectUnanimousDisagreement:
    """Mirrors Bug 3: all profiles give correct answer but expected is wrong."""

    def test_flags_all_wrong(self):
        records = [
            _record(profile="bare", correct=False),
            _record(profile="prompted", correct=False),
            _record(profile="solvay-full", correct=False),
        ]
        problems = {"test-001": _problem(expected_value="wrong_value")}
        reports = detect_unanimous_disagreement(records, problems)
        assert len(reports) == 1
        assert reports[0].fault_domain.value == "ambiguous_contract"
        assert "wrong_value" in reports[0].observed

    def test_ignores_when_any_correct(self):
        records = [
            _record(profile="bare", correct=False),
            _record(profile="prompted", correct=True),
        ]
        reports = detect_unanimous_disagreement(records, {"test-001": _problem()})
        assert len(reports) == 0

    def test_requires_multiple_profiles(self):
        records = [_record(profile="bare", correct=False)]
        reports = detect_unanimous_disagreement(records, {"test-001": _problem()})
        assert len(reports) == 0


# --- Integration: analyze_run ---


class TestAnalyzeRun:
    def test_combines_all_detectors(self):
        records = [
            _record(problem_id="p1", profile="bare", correct=False, grading_method="llm-judge"),
            _record(
                problem_id="p1", profile="prompted", correct=False, grading_method="llm-judge"
            ),
            _record(problem_id="p2", profile="bare", correct=False, answer_raw="x=5"),
            _record(problem_id="p2", profile="prompted", correct=True, answer_raw="x=5"),
        ]
        problems = {
            "p1": _problem("p1", expected_value="hbar*Q"),
            "p2": _problem("p2"),
        }
        reports = analyze_run(records, problems)
        types = {r.dedupe_key for r in reports}
        assert "grading-fallback:p1" in types
        assert "unanimous-disagreement:p1" in types
        assert "inconsistent-grading:p2" in types

    def test_deduplicates_by_key(self):
        records = [
            _record(profile="bare", correct=False, grading_method="llm-judge"),
            _record(profile="prompted", correct=False, grading_method="llm-judge"),
        ]
        problems = {"test-001": _problem()}
        reports = analyze_run(records, problems)
        keys = [r.dedupe_key for r in reports]
        assert len(keys) == len(set(keys))
