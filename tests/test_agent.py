"""Tests for the agent factory wiring."""

from __future__ import annotations

import pytest


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
