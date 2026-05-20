"""Tests for the tooled profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.tooled import tooled_runner  # noqa: F401
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_tooled_profile_registered() -> None:
    assert get_profile("tooled").cost_estimate == "medium"


def test_tooled_profile_wires_local_tools() -> None:
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [MagicMock(content="final: 2")],
    }
    with patch(
        "solvay.benchmark.profiles.tooled.create_react_agent",
        return_value=fake_agent,
    ) as create_react:
        result = get_profile("tooled").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    tools_arg = create_react.call_args.kwargs.get("tools") or create_react.call_args.args[1]
    tool_names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools_arg}
    assert "python_exec" in tool_names
    assert "check_dimensions" in tool_names
    assert result.answer_raw == "final: 2"
