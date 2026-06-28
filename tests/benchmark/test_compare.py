"""Tests for solvay.benchmark.compare."""

from __future__ import annotations

import pytest

from solvay.benchmark.compare import build_comparison, format_markdown
from solvay.benchmark.jsonl import RunHeader, RunRecord

# ── Helpers ──────────────────────────────────────────────────────────────────


def _header() -> RunHeader:
    return RunHeader(
        run_id="test",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:test",
        started_at="2026-06-05T00:00:00Z",
    )


def _record(
    problem_id: str,
    profile: str,
    correct: bool,
    elapsed: float = 1.0,
    model: str = "test",
) -> RunRecord:
    return RunRecord(
        problem_id=problem_id,
        profile=profile,
        model=model,
        repeat_idx=0,
        answer_raw="test",
        answer_extracted=None,
        correct=correct,
        elapsed_seconds=elapsed,
        tokens={"input": 100, "output": 200},
        grading_method="sympy",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_builds_from_records() -> None:
    """build_comparison discovers all unique profiles and problems."""
    records = [
        _record("p1", "bare", correct=True),
        _record("p1", "solvay-full", correct=True),
        _record("p2", "bare", correct=False),
        _record("p2", "solvay-full", correct=True),
    ]
    report = build_comparison(_header(), records)

    assert report.profiles == ["bare", "solvay-full"]
    assert report.problems == ["p1", "p2"]
    # All four (problem, profile) cells are populated.
    assert report.results["p1"]["bare"] is True
    assert report.results["p2"]["bare"] is False


def test_accuracy_per_profile() -> None:
    """bare = 50 %, solvay-full = 100 %."""
    records = [
        _record("p1", "bare", correct=True),
        _record("p1", "solvay-full", correct=True),
        _record("p2", "bare", correct=False),
        _record("p2", "solvay-full", correct=True),
    ]
    report = build_comparison(_header(), records)

    assert report.accuracy["bare"] == pytest.approx(0.5)
    assert report.accuracy["solvay-full"] == pytest.approx(1.0)


def test_format_markdown_contains_table() -> None:
    """format_markdown output contains expected headings and profile columns."""
    records = [
        _record("p1", "bare", correct=True),
        _record("p1", "solvay-full", correct=False),
    ]
    report = build_comparison(_header(), records)
    md = format_markdown(report)

    assert "| Problem" in md
    assert "bare" in md
    assert "solvay-full" in md
    assert "Accuracy" in md


def test_handles_single_profile() -> None:
    """build_comparison and format_markdown do not crash with one record."""
    records = [_record("p1", "bare", correct=True)]
    report = build_comparison(_header(), records)
    md = format_markdown(report)

    assert report.profiles == ["bare"]
    assert report.problems == ["p1"]
    assert "bare" in md
