"""Researcher subagent: gathers theoretical background."""

from __future__ import annotations

from typing import Any

from solvay.config import SolvayConfig
from solvay.schemas import ResearchBrief
from solvay.subagents import load_prompt


def create_researcher_subagent(
    config: SolvayConfig,
    web_search_tool: Any,
    url_fetch_tool: Any,
) -> dict[str, Any]:
    """Create the researcher subagent definition (dict subagent).

    Args:
        config: Solvay configuration.
        web_search_tool: The Tavily web_search tool function.
        url_fetch_tool: The url_fetch tool function.

    Returns:
        A dict subagent specification for deepagents.
    """
    return {
        "name": "researcher",
        "description": (
            "Gather applicable physics principles, candidate equations, analogies, "
            "and references for a given ProblemSpec."
        ),
        "system_prompt": load_prompt("researcher"),
        "model": config.model_for("researcher"),
        "tools": [web_search_tool, url_fetch_tool],
        "response_format": ResearchBrief,
    }
