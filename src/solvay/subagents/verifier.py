"""Verifier subagent: mechanical critique of solution drafts."""

from __future__ import annotations

from typing import Any

from solvay.config import SolvayConfig
from solvay.schemas import Verdict
from solvay.subagents import load_prompt
from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec


def create_verifier_subagent(config: SolvayConfig) -> dict[str, Any]:
    """Create the verifier subagent definition (dict subagent).

    Args:
        config: Solvay configuration.

    Returns:
        A dict subagent specification for deepagents.
    """
    return {
        "name": "verifier",
        "description": (
            "Mechanical critique: check dimensional consistency, limit cases, "
            "order of magnitude, and numerical consistency of a SolutionDraft."
        ),
        "system_prompt": load_prompt("verifier"),
        "model": config.model_for("verifier"),
        "tools": [python_exec, check_dimensions],
        "response_format": Verdict,
    }
