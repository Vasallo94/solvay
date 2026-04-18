"""Tests for the benchmark problem schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from solvay.benchmark.schema import Problem, load_problem, load_problems_dir


def _valid_payload() -> dict[str, Any]:
    return {
        "id": "synth-mech-0001",
        "version": 1,
        "source": {
            "kind": "synthetic",
            "origin": "solvay-generator-v0.1",
            "generated_at": "2026-04-18T14:00:00Z",
            "seed": 42,
        },
        "domain": "mechanics",
        "subdomain": "kinematics",
        "tags": ["incline", "newton"],
        "difficulty": "intermediate",
        "statement": "A block slides down a frictionless incline of angle 30 deg.",
        "given": {"m": "2 kg", "theta": "30 deg"},
        "find": "Acceleration of the block.",
        "expected": {
            "kind": "numeric",
            "value": "4.905",
            "unit": "m/s^2",
            "tolerance_rel": 0.01,
            "verification": {"method": "numeric_eval", "script": None},
        },
        "contamination": {
            "checked_at": "2026-04-18T14:05:00Z",
            "method": "model-recall-probe-v1",
            "score": 0.02,
            "verdict": "clean",
        },
        "notes": None,
    }


def test_problem_parses_valid_payload() -> None:
    problem = Problem.model_validate(_valid_payload())
    assert problem.id == "synth-mech-0001"
    assert problem.source.kind == "synthetic"
    assert problem.expected.kind == "numeric"
    assert problem.expected.tolerance_rel == 0.01


def test_problem_rejects_invalid_kind() -> None:
    payload = _valid_payload()
    payload["source"]["kind"] = "made-up"
    with pytest.raises(ValidationError):
        Problem.model_validate(payload)


def test_problem_rejects_invalid_expected_kind() -> None:
    payload = _valid_payload()
    payload["expected"]["kind"] = "essay"
    with pytest.raises(ValidationError):
        Problem.model_validate(payload)


def test_load_problem_roundtrip(tmp_path: Path) -> None:
    payload = _valid_payload()
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    problem = load_problem(path)
    assert problem.id == payload["id"]


def test_load_problems_dir_recursive(tmp_path: Path) -> None:
    mech = tmp_path / "mechanics"
    mech.mkdir()
    payload = _valid_payload()
    (mech / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    payload2 = {**payload, "id": "synth-mech-0002"}
    (mech / "b.json").write_text(json.dumps(payload2), encoding="utf-8")
    problems = load_problems_dir(tmp_path)
    assert sorted(p.id for p in problems) == ["synth-mech-0001", "synth-mech-0002"]
