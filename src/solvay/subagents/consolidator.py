"""Consolidator subagent: produces the final polished answer."""

from __future__ import annotations

from typing import Any

from solvay.config import SolvayConfig
from solvay.subagents import load_prompt


def create_consolidator_subagent(config: SolvayConfig) -> dict[str, Any]:
    """Create the consolidator subagent definition (dict subagent).

    Args:
        config: Solvay configuration.

    Returns:
        A dict subagent specification for deepagents.
    """
    return {
        "name": "consolidator",
        "description": (
            "Review all prior outputs (problem spec, research brief, solution "
            "draft, peer review verdict) and produce the final polished answer."
        ),
        "system_prompt": load_prompt("consolidator"),
        "model": config.model_for("consolidator"),
        "tools": [],
    }
