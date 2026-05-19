"""Tests for the problem composer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.composer import compose_problem
from solvay.benchmark.generator.skeleton import GenerationSkeleton

_VALID_JSON = """
{
  "id": "synth-mech-TEST",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-generator-v0.1",
    "generated_at": "2026-04-18T00:00:00Z",
    "seed": 42
  },
  "domain": "mechanics",
  "subdomain": "kinematics",
  "tags": ["pendulum"],
  "difficulty": "intermediate",
  "statement": "A pendulum...",
  "given": {"L": "1 m", "g": "9.81 m/s^2"},
  "find": "Angular frequency.",
  "expected": {
    "kind": "symbolic",
    "value": "sqrt(g/L)",
    "unit": "rad/s",
    "tolerance_rel": 0.01,
    "verification": {"method": "sympy_equivalence", "script": null}
  },
  "contamination": null,
  "notes": null
}
"""


def test_compose_returns_problem() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.return_value = MagicMock(content=_VALID_JSON)
    with patch("solvay.benchmark.generator.composer.init_chat_model", return_value=fake_chat):
        problem = compose_problem(
            GenerationSkeleton(domain="mechanics", compose=["pendulum"]),
            BenchConfig(),
        )
    assert problem.id == "synth-mech-TEST"
    assert problem.expected.value == "sqrt(g/L)"


def test_compose_retries_on_bad_json() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.side_effect = [
        MagicMock(content="not json"),
        MagicMock(content=_VALID_JSON),
    ]
    with patch("solvay.benchmark.generator.composer.init_chat_model", return_value=fake_chat):
        problem = compose_problem(
            GenerationSkeleton(domain="mechanics", compose=["pendulum"]),
            BenchConfig(composer_max_attempts=3),
        )
    assert problem.id == "synth-mech-TEST"
    assert fake_chat.invoke.call_count == 2
