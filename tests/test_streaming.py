"""Tests for StreamEvent types and RunCollector."""

from __future__ import annotations

from solvay.streaming import (
    RunCollector,
    RunFinished,
    SchemaProduced,
    SubagentFinished,
    SubagentStarted,
    ToolCallMade,
    ToolCallRecord,
    ToolResultReceived,
)


class TestRunCollector:
    def _collector(self) -> RunCollector:
        return RunCollector(problem="A ball drops from 10m", model="test-model")

    def test_subagent_started_creates_run(self) -> None:
        c = self._collector()
        c.accumulate(SubagentStarted(name="parser", t=0.0))
        assert len(c.subagent_runs) == 1
        assert c.subagent_runs[0].name == "parser"

    def test_tool_call_stored_in_current_subagent(self) -> None:
        c = self._collector()
        c.accumulate(SubagentStarted(name="parser", t=0.0))
        c.accumulate(ToolCallMade(subagent="parser", tool="python_exec", args_preview="h=10"))
        assert len(c.subagent_runs[0].tool_calls) == 1
        assert c.subagent_runs[0].tool_calls[0].tool == "python_exec"

    def test_tool_result_not_stored(self) -> None:
        c = self._collector()
        c.accumulate(SubagentStarted(name="parser", t=0.0))
        c.accumulate(ToolResultReceived(subagent="parser", tool="python_exec", result_preview="h=10"))
        assert c.subagent_runs[0].tool_calls == []

    def test_schema_stored_in_current_subagent(self) -> None:
        c = self._collector()
        c.accumulate(SubagentStarted(name="parser", t=0.0))
        c.accumulate(SchemaProduced(subagent="parser", schema_type="ProblemSpec", data={"domain": "mechanics"}))
        assert len(c.subagent_runs[0].schemas) == 1
        assert c.subagent_runs[0].schemas[0][0] == "ProblemSpec"

    def test_subagent_finished_updates_duration(self) -> None:
        c = self._collector()
        c.accumulate(SubagentStarted(name="parser", t=0.0))
        c.accumulate(SubagentFinished(name="parser", duration_s=8.5))
        assert c.subagent_runs[0].duration_s == 8.5

    def test_run_finished_sets_total_and_answer(self) -> None:
        c = self._collector()
        c.accumulate(RunFinished(total_s=42.0, final_answer="v = 14 m/s"))
        assert c.total_s == 42.0
        assert c.final_answer == "v = 14 m/s"

    def test_multiple_subagents_accumulated_in_order(self) -> None:
        c = self._collector()
        c.accumulate(SubagentStarted(name="parser", t=0.0))
        c.accumulate(SubagentFinished(name="parser", duration_s=5.0))
        c.accumulate(SubagentStarted(name="researcher", t=5.0))
        c.accumulate(SubagentFinished(name="researcher", duration_s=30.0))
        assert [r.name for r in c.subagent_runs] == ["parser", "researcher"]
        assert c.subagent_runs[1].duration_s == 30.0

    def test_tool_call_before_subagent_started_is_ignored(self) -> None:
        c = self._collector()
        c.accumulate(ToolCallMade(subagent="parser", tool="python_exec", args_preview="h=10"))
        assert c.subagent_runs == []

    def test_schema_before_subagent_started_is_ignored(self) -> None:
        c = self._collector()
        c.accumulate(SchemaProduced(subagent="parser", schema_type="ProblemSpec", data={}))
        assert c.subagent_runs == []


# ---------------------------------------------------------------------------
# Fake stream objects for parse_stream tests
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc, field as _field
from typing import Any as _Any


@_dc
class _FakeToolCall:
    tool_name: str
    input: _Any
    output: str | None = None
    error: str | None = None
    output_deltas: list[str] = _field(default_factory=list)


@_dc
class _FakeMsg:
    text: str


@_dc
class _FakeSubagent:
    name: str
    tool_calls: list[_FakeToolCall] = _field(default_factory=list)
    messages: list[_FakeMsg] = _field(default_factory=list)


@_dc
class _FakeStream:
    subagents: list[_FakeSubagent] = _field(default_factory=list)
    messages: list[_FakeMsg] = _field(default_factory=list)


class _FakeAgent:
    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream

    def stream_events(self, *args, **kwargs) -> _FakeStream:
        return self._stream


# ---------------------------------------------------------------------------
# parse_stream tests
# ---------------------------------------------------------------------------

from solvay.streaming import parse_stream  # noqa: E402


class TestParseStream:
    def _make_agent(self, stream: _FakeStream) -> _FakeAgent:
        return _FakeAgent(stream)

    def test_subagent_started_and_finished_emitted(self) -> None:
        stream = _FakeStream(subagents=[_FakeSubagent(name="parser")])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test problem"))
        names = [type(e).__name__ for e in events]
        assert "SubagentStarted" in names
        assert "SubagentFinished" in names
        started = next(e for e in events if isinstance(e, SubagentStarted))
        assert started.name == "parser"

    def test_tool_call_made_and_result_emitted(self) -> None:
        tc = _FakeToolCall(tool_name="python_exec", input={"code": "x=1"}, output='{"stdout": "1"}')
        stream = _FakeStream(subagents=[_FakeSubagent(name="solver", tool_calls=[tc])])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        tool_calls = [e for e in events if isinstance(e, ToolCallMade)]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool == "python_exec"

    def test_schema_produced_when_output_matches_known_schema(self) -> None:
        output = '{"principles": ["F=ma"], "candidate_equations": ["F=ma"], "analogies": [], "citations": []}'
        tc = _FakeToolCall(tool_name="web_search", input="query", output=output)
        stream = _FakeStream(subagents=[_FakeSubagent(name="researcher", tool_calls=[tc])])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        schemas = [e for e in events if isinstance(e, SchemaProduced)]
        assert len(schemas) == 1
        assert schemas[0].schema_type == "ResearchBrief"

    def test_tool_error_yields_result_received_with_error_prefix(self) -> None:
        tc = _FakeToolCall(tool_name="python_exec", input={}, error="ZeroDivisionError")
        stream = _FakeStream(subagents=[_FakeSubagent(name="solver", tool_calls=[tc])])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        results = [e for e in events if isinstance(e, ToolResultReceived)]
        assert len(results) == 1
        assert results[0].result_preview.startswith("ERROR:")

    def test_run_finished_emitted_last(self) -> None:
        stream = _FakeStream(subagents=[_FakeSubagent(name="parser")])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        assert isinstance(events[-1], RunFinished)

    def test_final_answer_captured_from_subagent_message(self) -> None:
        msg = _FakeMsg(text="The answer is 42 m/s because gravity.")
        stream = _FakeStream(subagents=[_FakeSubagent(name="consolidator", messages=[msg])])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        finished = next(e for e in events if isinstance(e, RunFinished))
        assert finished.final_answer == "The answer is 42 m/s because gravity."
