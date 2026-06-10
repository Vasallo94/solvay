"""Tests for the FINAL ANSWER convention helpers."""

from __future__ import annotations

from solvay.benchmark.answer_format import (
    FINAL_ANSWER_INSTRUCTION,
    append_final_answer_instruction,
    extract_final_answer,
)


class TestExtractFinalAnswer:
    def test_extracts_simple_line(self) -> None:
        text = "Some reasoning.\nFINAL ANSWER: 4.905 m/s^2\n"
        assert extract_final_answer(text) == "4.905 m/s^2"

    def test_case_insensitive_and_markdown_bold(self) -> None:
        text = "Steps...\n**Final Answer:** 42 J"
        assert extract_final_answer(text) == "42 J"

    def test_takes_last_occurrence(self) -> None:
        text = "FINAL ANSWER: 1 m\nMore correction...\nFINAL ANSWER: 2 m"
        assert extract_final_answer(text) == "2 m"

    def test_returns_none_when_absent(self) -> None:
        assert extract_final_answer("just text with 4.9 numbers") is None

    def test_strips_surrounding_whitespace_and_dollar_signs(self) -> None:
        text = "FINAL ANSWER: $g \\sin(\\theta)$"
        assert extract_final_answer(text) == "g \\sin(\\theta)"


class TestAppendInstruction:
    def test_appends_instruction_after_statement(self) -> None:
        statement = "A block slides down an incline."
        combined = append_final_answer_instruction(statement)
        assert combined.startswith(statement)
        assert FINAL_ANSWER_INSTRUCTION in combined
        assert "FINAL ANSWER:" in combined
