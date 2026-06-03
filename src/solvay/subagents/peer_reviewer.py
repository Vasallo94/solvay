"""Peer reviewer subagent: semantic critique of solution drafts."""

from __future__ import annotations

from typing import Any

from solvay.config import SolvayConfig
from solvay.schemas import Verdict
from solvay.subagents import load_prompt
from solvay.tools.physics_checklist import physics_checklist
from solvay.tools.python_exec import python_exec


def create_peer_reviewer_subagent(
    config: SolvayConfig,
    web_search_tool: Any,
) -> dict[str, Any]:
    """Create the peer reviewer subagent definition (dict subagent).

    Args:
        config: Solvay configuration.
        web_search_tool: The Tavily web_search tool function.

    Returns:
        A dict subagent specification for deepagents.
    """
    return {
        "name": "peer_reviewer",
        "description": (
            "Semantic critique: assess whether the approach is correct, complete, "
            "and well-justified. Check for missing terms and unjustified assumptions."
        ),
        "system_prompt": load_prompt("peer_reviewer"),
        "model": config.model_for("peer_reviewer"),
        "tools": [python_exec, web_search_tool, physics_checklist],
        "response_format": Verdict,
    }
