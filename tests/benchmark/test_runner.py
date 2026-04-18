"""Tests for the matrix runner."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.jsonl import read_run
from solvay.benchmark.profiles import PROFILES, Profile, ProfileResult, register
from solvay.benchmark.runner import MatrixSpec, run_matrix
from solvay.benchmark.schema import Problem


def _fake_problem(pid: str = "p1") -> Problem:
    payload = {
        "id": pid,
        "version": 1,
        "source": {
            "kind": "synthetic",
            "origin": "test",
            "generated_at": "2026-04-18T00:00:00Z",
            "seed": 0,
        },
        "domain": "mechanics",
        "subdomain": None,
        "tags": [],
        "difficulty": "easy",
        "statement": "what is one plus one?",
        "given": {},
        "find": "sum",
        "expected": {
            "kind": "numeric",
            "value": "2",
            "unit": None,
            "tolerance_rel": 0.01,
            "verification": {"method": "numeric_eval", "script": None},
        },
        "contamination": None,
        "notes": None,
    }
    return Problem.model_validate(payload)


@pytest.fixture(autouse=True)
def _register_fake_profile() -> Iterator[None]:
    def runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
        return ProfileResult(
            answer_raw=f"answer={problem.id}:{model}",
            answer_extracted="2",
            elapsed_seconds=0.01,
            tokens={"input": 1, "output": 1, "cache_read": 0},
        )

    register(Profile(name="fake", description="fake", runner=runner, cost_estimate="low"))
    yield
    PROFILES.pop("fake", None)


def test_run_matrix_emits_row_per_cell(tmp_path: Path) -> None:
    problems = [_fake_problem("p1"), _fake_problem("p2")]
    spec = MatrixSpec(
        problems=problems,
        profile_names=["fake"],
        models=["anthropic:claude-sonnet-4-6", "openai:gpt-4o"],
        repeats=2,
    )
    out = tmp_path / "run.jsonl"
    run_matrix(spec, out_path=out, config=BenchConfig())

    _, records = read_run(out)
    assert len(records) == 2 * 1 * 2 * 2
    assert all(r.profile == "fake" for r in records)


def test_run_matrix_correctness_uses_expected_value(tmp_path: Path) -> None:
    spec = MatrixSpec(
        problems=[_fake_problem()],
        profile_names=["fake"],
        models=["anthropic:claude-sonnet-4-6"],
        repeats=1,
    )
    out = tmp_path / "run.jsonl"
    run_matrix(spec, out_path=out, config=BenchConfig())
    _, records = read_run(out)
    assert records[0].correct is True
