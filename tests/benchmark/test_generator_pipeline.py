"""Tests for the generator pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.pipeline import GenerationReport, generate_batch
from solvay.benchmark.generator.skeleton import GenerationSkeleton
from solvay.benchmark.generator.verifier import VerifyOutcome
from solvay.benchmark.schema import Contamination, Problem


def _good_problem(pid: str = "synth-mech-T1") -> Problem:
    return Problem.model_validate(
        {
            "id": pid,
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "solvay-generator-v0.1",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 1,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": ["pendulum"],
            "difficulty": "intermediate",
            "statement": "good problem",
            "given": {"g": "9.81", "L": "1"},
            "find": "omega",
            "expected": {
                "kind": "symbolic",
                "value": "sqrt(g/L)",
                "unit": "rad/s",
                "tolerance_rel": 0.01,
                "verification": {"method": "sympy_equivalence", "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_generate_batch_writes_verified_problems(tmp_path: Path) -> None:
    contamination = Contamination(
        checked_at="2026-04-18T00:00:00Z",
        method="model-recall-probe-v1",
        score=0.1,
        verdict="clean",
    )
    with (
        patch(
            "solvay.benchmark.generator.pipeline.compose_problem",
            side_effect=[_good_problem("synth-mech-T1"), _good_problem("synth-mech-T2")],
        ),
        patch(
            "solvay.benchmark.generator.pipeline.verify_problem",
            return_value=VerifyOutcome(ok=True, method="sympy_equivalence"),
        ),
        patch(
            "solvay.benchmark.generator.pipeline.probe_contamination",
            return_value=contamination,
        ),
    ):
        report = generate_batch(
            skeletons=[
                GenerationSkeleton(domain="mechanics", compose=["pendulum"]),
                GenerationSkeleton(domain="mechanics", compose=["lorentz"]),
            ],
            out_dir=tmp_path,
            config=BenchConfig(),
        )
    assert isinstance(report, GenerationReport)
    assert report.written == 2
    assert report.rejected_verification == 0
    assert (tmp_path / "synth-mech-T1.json").exists()
    assert (tmp_path / "synth-mech-T2.json").exists()


def test_generate_batch_rejects_verification_failures(tmp_path: Path) -> None:
    with (
        patch("solvay.benchmark.generator.pipeline.compose_problem", return_value=_good_problem()),
        patch(
            "solvay.benchmark.generator.pipeline.verify_problem",
            return_value=VerifyOutcome(ok=False, method="sympy_equivalence", reason="bad"),
        ),
    ):
        report = generate_batch(
            skeletons=[GenerationSkeleton(domain="mechanics", compose=["pendulum"])],
            out_dir=tmp_path,
            config=BenchConfig(),
        )
    assert report.written == 0
    assert report.rejected_verification == 1
