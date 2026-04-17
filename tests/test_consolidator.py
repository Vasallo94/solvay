"""Tests for the session-end consolidator subagent."""

from __future__ import annotations

import json

from solvay.schemas import JournalEntry
from solvay.subagents.consolidator import parse_consolidator_response


class TestParseConsolidatorResponse:
    def test_valid_entries(self) -> None:
        response = json.dumps([
            {
                "role": "solver",
                "iteration": 1,
                "content": "Lagrangian approach worked better than Newtonian for coupled systems.",
            },
            {
                "role": "verifier",
                "iteration": 2,
                "content": "Always check units before checking magnitudes.",
            },
        ])
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
        response = json.dumps([
            {"role": "solver", "iteration": 1, "content": "Valid entry."},
            {"bad_field": "missing required fields"},
        ])
        entries = parse_consolidator_response(response)
        assert len(entries) == 1
        assert entries[0].role == "solver"
