"""Tests for the bare profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.bare import bare_runner  # noqa: F401
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    return Problem.model_validate(
        {
            "id": "p1",
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
            "statement": "Compute 1+1.",
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
    )


def test_bare_profile_registered() -> None:
    p = get_profile("bare")
    assert p.cost_estimate == "low"


def test_bare_profile_uses_user_only_message() -> None:
    fake_chat = MagicMock()
    response = MagicMock()
    response.content = "2"
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 1}
    fake_chat.invoke.return_value = response
    with patch("solvay.benchmark.profiles.bare.init_chat_model", return_value=fake_chat):
        result = get_profile("bare").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    assert result.answer_raw == "2"
    call_args = fake_chat.invoke.call_args[0][0]
    # bare profile: a single user message, no system prompt
    assert len(call_args) == 1
    assert call_args[0]["role"] == "user"
