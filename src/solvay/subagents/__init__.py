"""Subagent factories for Solvay."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(role: str) -> str:
    """Load the system prompt for a given role.

    Args:
        role: The role name (e.g., "parser", "researcher").

    Returns:
        The prompt text as a string.
    """
    path = _PROMPTS_DIR / f"{role}.md"
    return path.read_text(encoding="utf-8")
