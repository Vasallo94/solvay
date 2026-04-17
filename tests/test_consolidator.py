"""Tests for the session-end consolidator subagent."""

from __future__ import annotations

import inspect
import json

from pydantic import BaseModel

from solvay.config import SolvayConfig
from solvay.schemas import JournalEntry
from solvay.subagents.consolidator import (
    create_consolidator_subagent,
    parse_consolidator_response,
)


class TestConsolidatorSubagentSpec:
    def test_response_format_is_pydantic_model(self) -> None:
        """langchain rejects bare GenericAlias (e.g. list[X]) as response_format;
        only Pydantic models, dataclasses, TypedDicts, or JSON schema dicts are
        accepted. Ensure we don't regress back to list[JournalEntry]."""
        spec = create_consolidator_subagent(SolvayConfig())
        rf = spec["response_format"]
        assert inspect.isclass(rf) and issubclass(rf, BaseModel), (
            f"response_format must be a Pydantic model; got {type(rf).__name__}"
        )


class TestParseConsolidatorResponse:
    def test_valid_entries(self) -> None:
        response = json.dumps(
            [
                {
                    "role": "solver",
                    "iteration": 1,
                    "content": (
                        "Lagrangian approach worked better than Newtonian for coupled systems."
                    ),
                },
                {
                    "role": "verifier",
                    "iteration": 2,
                    "content": "Always check units before checking magnitudes.",
                },
            ]
        )
        entries = parse_consolidator_response(response)
        assert len(entries) == 2
        assert all(isinstance(e, JournalEntry) for e in entries)
        assert entries[0].role == "solver"

    def test_empty_list(self) -> None:
        entries = parse_consolidator_response("[]")
        assert entries == []

    def test_invalid_json_returns_empty(self) -> None:
        entries = parse_consolidator_response("not valid json")
        assert entries == []

    def test_partial_valid_entries(self) -> None:
        response = json.dumps(
            [
                {"role": "solver", "iteration": 1, "content": "Valid entry."},
                {"bad_field": "missing required fields"},
            ]
        )
        entries = parse_consolidator_response(response)
        assert len(entries) == 1
        assert entries[0].role == "solver"

    def test_wrapped_entries_format(self) -> None:
        """The response_format wrapper makes the LLM return
        ``{"entries": [...]}`` instead of a bare list. The parser must
        transparently accept both shapes."""
        response = json.dumps(
            {
                "entries": [
                    {"role": "solver", "iteration": 1, "content": "Wrapped entry."},
                ]
            }
        )
        entries = parse_consolidator_response(response)
        assert len(entries) == 1
        assert entries[0].content == "Wrapped entry."
