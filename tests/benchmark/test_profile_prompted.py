"""Tests for the prompted profile."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.prompted import prompted_runner  # noqa: F401

if TYPE_CHECKING:
    from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_prompted_profile_registered() -> None:
    assert get_profile("prompted").cost_estimate == "low"


def test_prompted_profile_sends_system_prompt() -> None:
    fake_chat = MagicMock()
    response = MagicMock()
    response.content = "2 (dimensionless)"
    response.usage_metadata = {"input_tokens": 20, "output_tokens": 4}
    fake_chat.invoke.return_value = response
    with patch("solvay.benchmark.profiles.prompted.init_chat_model", return_value=fake_chat):
        get_profile("prompted").runner(_problem(), "anthropic:claude-sonnet-4-6", BenchConfig())
    call_args = fake_chat.invoke.call_args[0][0]
    assert call_args[0]["role"] == "system"
    assert "physicist" in call_args[0]["content"].lower()
    assert call_args[1]["role"] == "user"
