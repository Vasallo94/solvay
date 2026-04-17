"""Consolidator subagent: extracts learning-worthy entries from session notebooks."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from solvay.config import SolvayConfig
from solvay.schemas import JournalEntry, JournalEntryList
from solvay.subagents import load_prompt


def create_consolidator_subagent(config: SolvayConfig) -> dict[str, Any]:
    """Create the consolidator subagent definition (dict subagent).

    The consolidator reviews a session notebook and extracts entries
    worth persisting to the cross-run journal.

    Args:
        config: Solvay configuration.

    Returns:
        A dict subagent specification for deepagents.
    """
    return {
        "name": "consolidator",
        "description": (
            "Review a completed session's lab notebook and extract learning-worthy "
            "entries (mistakes fixed, methodological tricks, pitfalls) for the "
            "persistent journal."
        ),
        "system_prompt": load_prompt("consolidator"),
        "model": config.model_for("consolidator"),
        "tools": [],
        "response_format": JournalEntryList,
    }


def parse_consolidator_response(response_text: str) -> list[JournalEntry]:
    """Parse the consolidator's JSON response into JournalEntry objects.

    Accepts both the current wrapped format ``{"entries": [...]}`` emitted
    under ``JournalEntryList`` and the legacy bare-list ``[...]`` form.
    Best-effort: silently skips items that fail validation, returns an
    empty list on top-level JSON or shape errors. The consolidator runs
    at session-end and partial recovery is preferred over hard failure.

    Args:
        response_text: JSON string with the consolidator output.

    Returns:
        List of JournalEntry objects. Empty list if parsing fails.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError, TypeError:
        return []

    # Unwrap {"entries": [...]} shape from the Pydantic response_format.
    if isinstance(data, dict):
        data = data.get("entries", [])

    if not isinstance(data, list):
        return []

    entries: list[JournalEntry] = []
    for item in data:
        try:
            entries.append(JournalEntry(**item))
        except TypeError, ValidationError:
            continue

    return entries
