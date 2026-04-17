"""Parser subagent: extracts ProblemSpec from natural language."""

from __future__ import annotations

from typing import Any

from solvay.config import SolvayConfig
from solvay.schemas import ProblemSpec
from solvay.subagents import load_prompt


def create_parser_subagent(config: SolvayConfig) -> dict[str, Any]:
    """Create the parser subagent definition (dict subagent).

    The parser has no tools -- it relies on the LLM to parse the problem.

    Args:
        config: Solvay configuration.

    Returns:
        A dict subagent specification for deepagents.
    """
    return {
        "name": "parser",
        "description": (
            "Parse a natural-language physics problem statement into a structured "
            "ProblemSpec with domain, knowns, unknowns, assumptions, and approach hints."
        ),
        "system_prompt": load_prompt("parser"),
        "model": config.model_for("parser"),
        "tools": [],
        "response_format": ProblemSpec,
    }
