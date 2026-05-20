"""Tests for the solvay-noweb profile."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.solvay_noweb import solvay_noweb_runner  # noqa: F401
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_solvay_noweb_registered() -> None:
    assert get_profile("solvay-noweb").cost_estimate == "high"


def test_solvay_noweb_unsets_tavily_during_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str | None] = {}

    def fake_create(scfg):  # type: ignore[no-untyped-def]
        seen["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [MagicMock(content="0")]}
        return agent

    monkeypatch.setenv("TAVILY_API_KEY", "PRESENT")
    with patch(
        "solvay.benchmark.profiles.solvay_noweb.create_solvay_agent",
        side_effect=fake_create,
    ):
        get_profile("solvay-noweb").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    assert seen["TAVILY_API_KEY"] is None
    # Restored after the call:
    assert os.environ.get("TAVILY_API_KEY") == "PRESENT"


def test_solvay_noweb_restores_tavily_when_create_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "PRESENT")
    with (
        patch(
            "solvay.benchmark.profiles.solvay_noweb.create_solvay_agent",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError),
    ):
        get_profile("solvay-noweb").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    assert os.environ.get("TAVILY_API_KEY") == "PRESENT"
