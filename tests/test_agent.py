"""Tests for the agent factory wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from deepagents.backends import CompositeBackend

from solvay.harness import HARNESS_NOTES_PATH, LONG_TERM_MEMORY_PATH


class TestWebSearchTool:
    def test_returns_stub_when_tavily_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without TAVILY_API_KEY the factory must return a callable stub
        rather than crashing with KeyError, so the agent can still run on
        problems that don't need web search."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        from solvay.agent import _create_web_search_tool

        web_search = _create_web_search_tool()
        assert callable(web_search)

        result = web_search("anything")
        assert isinstance(result, dict)
        # The stub must signal unavailability so subagents don't misinterpret
        # empty results as "no information found on the web".
        assert "unavailable" in str(result.get("error", "")).lower()


def test_create_agent_wires_native_memory_backend_and_permissions() -> None:
    from solvay.agent import create_solvay_agent

    with patch("solvay.agent.create_deep_agent") as fake_create:
        fake_create.return_value = object()
        create_solvay_agent()

    kwargs = fake_create.call_args.kwargs
    assert kwargs["memory"] == [LONG_TERM_MEMORY_PATH, HARNESS_NOTES_PATH]
    assert isinstance(kwargs["backend"], CompositeBackend)
    assert kwargs["permissions"]


def test_peer_reviewer_has_physics_checklist_tool() -> None:
    """The peer reviewer subagent must include physics_checklist in its tools."""
    from solvay.config import SolvayConfig
    from solvay.subagents.peer_reviewer import create_peer_reviewer_subagent

    def stub(**kw: object) -> dict[str, object]:
        return {"results": [], "error": "stub"}

    config = SolvayConfig(default_model="fake-model")
    spec = create_peer_reviewer_subagent(config, web_search_tool=stub)

    tool_names = []
    for t in spec["tools"]:
        name = getattr(t, "__name__", None) or getattr(t, "name", str(t))
        tool_names.append(name)

    assert "physics_checklist" in tool_names


def test_no_auto_general_purpose_subagent() -> None:
    """Ensure 'general-purpose' is explicitly included in the subagents list.

    deepagents auto-injects a generic 'general-purpose' subagent when none
    with that name exists, which biases the orchestrator LLM away from the
    solver subagent. By pre-registering our own 'general-purpose' clone of
    the solver we prevent the auto-injection.
    """
    from solvay.agent import create_solvay_agent

    with patch("solvay.agent.create_deep_agent") as fake_create:
        fake_create.return_value = object()
        create_solvay_agent()

    subagents = fake_create.call_args.kwargs["subagents"]
    names = [sa["name"] if isinstance(sa, dict) else sa.name for sa in subagents]
    assert "general-purpose" in names, (
        f"'general-purpose' subagent not found in subagents list: {names}"
    )


def test_solver_wired_from_conversational_module() -> None:
    # The solver factory must come from the conversational module, not solver_loop.
    import solvay.agent as agent_module

    assert agent_module.create_solver_subagent.__module__ == "solvay.subagents.solver"
