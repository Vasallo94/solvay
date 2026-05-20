"""Tests for the solvay-full profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.solvay_full import solvay_full_runner  # noqa: F401
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_solvay_full_profile_registered() -> None:
    assert get_profile("solvay-full").cost_estimate == "high"


def test_solvay_full_profile_uses_full_agent() -> None:
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [MagicMock(content="2 m/s^2")]}
    with patch(
        "solvay.benchmark.profiles.solvay_full.create_solvay_agent",
        return_value=fake_agent,
    ) as create_full:
        result = get_profile("solvay-full").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    # The injected model must propagate to SolvayConfig.default_model
    cfg = create_full.call_args.args[0]
    assert cfg.default_model == "anthropic:claude-sonnet-4-6"
    assert result.answer_raw == "2 m/s^2"
