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
