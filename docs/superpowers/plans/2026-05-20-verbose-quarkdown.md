# Verbose Live Output & Quarkdown Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live step-by-step terminal output inside each subagent and always generate a Quarkdown `.qd` report after every `solve` run.

**Architecture:** A new `streaming.py` exposes typed `StreamEvent` dataclasses and a `parse_stream()` generator over `agent.stream_events(version="v3")`; a new `report.py` renders a Quarkdown document from the accumulated `RunCollector` state; `cli.py` wires them together replacing `_stream_verbose`.

**Tech Stack:** deepagents 0.6.2 (`stream_events` v3 API), Python dataclasses, Typer, Quarkdown syntax (plain text generation — no external library needed), pytest + monkeypatch for tests.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/solvay/streaming.py` | Create | `StreamEvent` types, `ToolCallRecord`, `SubagentRun`, `RunCollector`, `parse_stream()` |
| `src/solvay/report.py` | Create | `generate_quarkdown(collector) -> str` and all section renderers |
| `src/solvay/cli.py` | Modify | Replace `_stream_verbose`, add `--output` flag, wire report generation |
| `src/solvay/prompts/consolidator.md` | Modify | Add Quarkdown output format instructions |
| `tests/test_streaming.py` | Create | Unit tests for `RunCollector.accumulate()` and `parse_stream()` |
| `tests/test_report.py` | Create | Unit tests for `generate_quarkdown()` |

---

## Task 1: `StreamEvent` types, `RunCollector`, and helpers (`streaming.py`)

**Files:**
- Create: `src/solvay/streaming.py`
- Create: `tests/test_streaming.py`

- [ ] **Step 1.1: Write failing tests for `RunCollector`**

Create `tests/test_streaming.py`:

```python
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
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```
uv run pytest tests/test_streaming.py -v
```

Expected: `ImportError` — `solvay.streaming` does not exist yet.

- [ ] **Step 1.3: Implement `streaming.py` with types and `RunCollector`**

Create `src/solvay/streaming.py`:

```python
"""Typed StreamEvent protocol and RunCollector for the Solvay pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Union

# ---------------------------------------------------------------------------
# StreamEvent dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubagentStarted:
    name: str
    t: float  # time.monotonic() when dispatched


@dataclass
class ToolCallMade:
    subagent: str
    tool: str
    args_preview: str  # first ~80 chars of args


@dataclass
class ToolResultReceived:
    subagent: str
    tool: str
    result_preview: str  # first ~80 chars of result


@dataclass
class SchemaProduced:
    subagent: str
    schema_type: str  # "ProblemSpec", "ResearchBrief", "SolutionDraft", "Verdict"
    data: dict  # full parsed dict


@dataclass
class SubagentFinished:
    name: str
    duration_s: float


@dataclass
class RunFinished:
    total_s: float
    final_answer: str


StreamEvent = Union[
    SubagentStarted,
    ToolCallMade,
    ToolResultReceived,
    SchemaProduced,
    SubagentFinished,
    RunFinished,
]

# ---------------------------------------------------------------------------
# RunCollector
# ---------------------------------------------------------------------------

_SCHEMA_FIELDS: dict[str, set[str]] = {
    "ProblemSpec": {"statement", "domain", "knowns", "unknowns", "assumptions", "approach_hints"},
    "ResearchBrief": {"principles", "candidate_equations", "analogies", "citations"},
    "SolutionDraft": {"method", "steps", "final_answer", "code_trace"},
    "Verdict": {"approved", "issues", "severity"},
}


@dataclass
class ToolCallRecord:
    tool: str
    args_preview: str


@dataclass
class SubagentRun:
    name: str
    duration_s: float = 0.0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    schemas: list[tuple[str, dict]] = field(default_factory=list)


@dataclass
class RunCollector:
    problem: str
    model: str
    subagent_runs: list[SubagentRun] = field(default_factory=list)
    total_s: float = 0.0
    final_answer: str = ""

    def accumulate(self, event: StreamEvent) -> None:
        if isinstance(event, SubagentStarted):
            self.subagent_runs.append(SubagentRun(name=event.name))
        elif isinstance(event, ToolCallMade):
            if self.subagent_runs:
                self.subagent_runs[-1].tool_calls.append(
                    ToolCallRecord(tool=event.tool, args_preview=event.args_preview)
                )
        elif isinstance(event, SchemaProduced):
            if self.subagent_runs:
                self.subagent_runs[-1].schemas.append((event.schema_type, event.data))
        elif isinstance(event, SubagentFinished):
            for run in reversed(self.subagent_runs):
                if run.name == event.name:
                    run.duration_s = event.duration_s
                    break
        elif isinstance(event, RunFinished):
            self.total_s = event.total_s
            self.final_answer = event.final_answer


# ---------------------------------------------------------------------------
# Helpers (used by parse_stream, defined here to be testable)
# ---------------------------------------------------------------------------


def _truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


def _try_detect_schema(subagent: str, content: str) -> SchemaProduced | None:
    """Return a SchemaProduced event if content is JSON matching a known schema."""
    content = content.strip()
    if not content.startswith("{"):
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for schema_name, required in _SCHEMA_FIELDS.items():
        if required <= data.keys():
            return SchemaProduced(subagent=subagent, schema_type=schema_name, data=data)
    return None


# ---------------------------------------------------------------------------
# parse_stream
# ---------------------------------------------------------------------------


def parse_stream(agent: Any, problem: str):  # -> Iterator[StreamEvent]
    """Yield StreamEvents by consuming agent.stream_events(version='v3').

    Uses the deepagents 0.6+ high-level streaming API:
        stream.subagents  -> yields as each subagent executes
        subagent.tool_calls -> yields as each tool fires
        tool_call.output_deltas -> live output fragments
        tool_call.output  -> full result when complete
        tool_call.error   -> set if the tool raised
    """
    import time

    t0 = time.monotonic()
    final_answer = ""

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": problem}]},
        version="v3",
    )

    for subagent in stream.subagents:
        t_start = time.monotonic()
        yield SubagentStarted(name=subagent.name, t=t_start)

        for tool_call in subagent.tool_calls:
            args_preview = _truncate(str(tool_call.input), 80)
            yield ToolCallMade(
                subagent=subagent.name,
                tool=tool_call.tool_name,
                args_preview=args_preview,
            )

            # Collect full output (output_deltas stream, then .output when done)
            output_parts: list[str] = []
            for delta in tool_call.output_deltas:
                output_parts.append(str(delta))

            full_output = tool_call.output if tool_call.output is not None else "".join(output_parts)
            output_str = full_output if isinstance(full_output, str) else str(full_output)

            if tool_call.error is not None:
                yield ToolResultReceived(
                    subagent=subagent.name,
                    tool=tool_call.tool_name,
                    result_preview=f"ERROR: {_truncate(str(tool_call.error), 75)}",
                )
            else:
                schema_evt = _try_detect_schema(subagent.name, output_str)
                if schema_evt:
                    yield schema_evt
                else:
                    yield ToolResultReceived(
                        subagent=subagent.name,
                        tool=tool_call.tool_name,
                        result_preview=_truncate(output_str, 80),
                    )

        # Check subagent final message for schema (no-tool-call output)
        for msg in subagent.messages:
            text = getattr(msg, "text", "") or ""
            if text.strip().startswith("{"):
                schema_evt = _try_detect_schema(subagent.name, text)
                if schema_evt:
                    yield schema_evt
                    break
            # Capture consolidator plain-text output as final answer candidate
            if text and len(text) > 20:
                final_answer = text

        yield SubagentFinished(name=subagent.name, duration_s=time.monotonic() - t_start)

    # Capture the coordinator's final message if consolidator didn't set it
    for msg in stream.messages:
        text = getattr(msg, "text", "") or ""
        if text and len(text) > 20:
            final_answer = text

    yield RunFinished(total_s=time.monotonic() - t0, final_answer=final_answer)
```

- [ ] **Step 1.4: Run tests — expect pass**

```
uv run pytest tests/test_streaming.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/solvay/streaming.py tests/test_streaming.py
git commit -m "feat: add StreamEvent types, RunCollector, and parse_stream skeleton"
```

---

## Task 2: `parse_stream()` unit tests with a mock agent

**Files:**
- Modify: `tests/test_streaming.py` (add `TestParseStream` class)

The `parse_stream()` function calls `agent.stream_events(version="v3")`. We test it by building a fake agent that returns a fake stream object with the same interface as the deepagents v3 stream.

- [ ] **Step 2.1: Write failing tests for `parse_stream`**

Append to `tests/test_streaming.py`:

```python
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
```

- [ ] **Step 2.2: Run tests — confirm they pass**

```
uv run pytest tests/test_streaming.py -v
```

Expected: all tests in `TestRunCollector` and `TestParseStream` pass.

- [ ] **Step 2.3: Commit**

```bash
git add tests/test_streaming.py
git commit -m "test: add parse_stream unit tests with fake stream objects"
```

---

## Task 3: `generate_quarkdown()` in `report.py`

**Files:**
- Create: `src/solvay/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_report.py`:

```python
"""Tests for Quarkdown report generation."""

from __future__ import annotations

import datetime

from solvay.report import generate_quarkdown
from solvay.streaming import RunCollector, SubagentRun, ToolCallRecord


def _make_collector() -> RunCollector:
    parser_run = SubagentRun(
        name="parser",
        duration_s=8.0,
        tool_calls=[ToolCallRecord(tool="python_exec", args_preview="h = 10")],
        schemas=[
            (
                "ProblemSpec",
                {
                    "statement": "A ball dropped from 10m",
                    "domain": "mechanics",
                    "knowns": {"h": {"value": 10.0, "unit": "m"}},
                    "unknowns": ["v_final"],
                    "assumptions": ["frictionless"],
                    "approach_hints": ["kinematics"],
                },
            )
        ],
    )
    researcher_run = SubagentRun(
        name="researcher",
        duration_s=31.0,
        tool_calls=[ToolCallRecord(tool="web_search", args_preview="kinematic equations")],
        schemas=[
            (
                "ResearchBrief",
                {
                    "principles": ["Conservation of energy"],
                    "candidate_equations": ["v^2 = 2gh"],
                    "analogies": [],
                    "citations": ["HyperPhysics"],
                },
            )
        ],
    )
    consolidator_run = SubagentRun(
        name="consolidator",
        duration_s=9.0,
    )
    return RunCollector(
        problem="A ball dropped from 10m, find final velocity",
        model="anthropic:claude-sonnet-4-6",
        subagent_runs=[parser_run, researcher_run, consolidator_run],
        total_s=96.0,
        final_answer=".box {Answer} type:{tip}\n    v = 14.0 m/s",
    )


class TestGenerateQuarkdown:
    def test_document_header_present(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert ".docname {Solvay Solution Report}" in qd
        assert ".theme {paperwhite} layout:{latex}" in qd

    def test_problem_statement_section(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "## Problem Statement" in qd
        assert "A ball dropped from 10m" in qd

    def test_model_and_domain_in_header(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "claude-sonnet-4-6" in qd
        assert "mechanics" in qd

    def test_parser_section_with_table(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Problem Parsing" in qd
        assert "v_final" in qd
        assert "| Domain" in qd

    def test_assumptions_box(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert ".box {Assumptions} type:{tip}" in qd
        assert "frictionless" in qd

    def test_research_section_with_principles(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Research" in qd
        assert "Conservation of energy" in qd

    def test_final_answer_section(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Final Answer" in qd
        assert "v = 14.0 m/s" in qd

    def test_run_metadata_table(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Run Metadata" in qd
        assert "| parser" in qd
        assert "| **Total**" in qd

    def test_consolidator_not_rendered_as_pipeline_section(self) -> None:
        qd = generate_quarkdown(_make_collector())
        # Consolidator output goes into Final Answer, not its own numbered section
        assert "· Consolidation" not in qd

    def test_tool_calls_listed(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "`python_exec`" in qd
        assert "`web_search`" in qd
```

- [ ] **Step 3.2: Run tests — confirm they fail**

```
uv run pytest tests/test_report.py -v
```

Expected: `ImportError` — `solvay.report` does not exist yet.

- [ ] **Step 3.3: Implement `report.py`**

Create `src/solvay/report.py`:

```python
"""Generate a Quarkdown (.qd) document from a completed RunCollector."""

from __future__ import annotations

import datetime

from solvay.streaming import RunCollector, SubagentRun

_SECTION_TITLES: dict[str, str] = {
    "parser": "Problem Parsing",
    "researcher": "Research",
    "solver": "Solution",
    "verifier": "Verification",
    "peer_reviewer": "Peer Review",
}


def generate_quarkdown(collector: RunCollector) -> str:
    """Render the full Quarkdown report from accumulated pipeline state."""
    parts: list[str] = []

    domain = _domain_from_collector(collector)
    date_str = datetime.date.today().isoformat()
    duration_str = _fmt_duration(collector.total_s)
    title = _title_from_problem(collector.problem)

    parts.append(
        f""".docname {{Solvay Solution Report}}
.doctype {{plain}}
.doclang {{English}}
.theme {{paperwhite}} layout:{{latex}}
.numbering
    - headings: 1

# {title}

**Domain:** {domain} | **Model:** {collector.model} | **Date:** {date_str} | **Duration:** {duration_str}

---

## Problem Statement

{collector.problem}

---

"""
    )

    section_num = 1
    for run in collector.subagent_runs:
        if run.name == "consolidator":
            continue
        parts.append(_render_subagent_section(section_num, run))
        section_num += 1

    # Final answer — written by consolidator in Quarkdown format
    parts.append(f"## {section_num} · Final Answer\n\n")
    parts.append(collector.final_answer or "*No answer recorded.*")
    parts.append("\n\n---\n\n")

    # Run metadata
    parts.append("## Run Metadata\n\n")
    parts.append("| Subagent | Duration |\n|----------|----------|\n")
    for run in collector.subagent_runs:
        parts.append(f"| {run.name} | {_fmt_duration(run.duration_s)} |\n")
    parts.append(f"| **Total** | **{_fmt_duration(collector.total_s)}** |\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_subagent_section(num: int, run: SubagentRun) -> str:
    title = _SECTION_TITLES.get(run.name, run.name.replace("_", " ").title())
    lines = [f"## {num} · {title}\n\n*{run.name} — {_fmt_duration(run.duration_s)}*\n\n"]

    for schema_type, data in run.schemas:
        if schema_type == "ProblemSpec":
            lines.append(_render_problem_spec(data))
        elif schema_type == "ResearchBrief":
            lines.append(_render_research_brief(data))
        elif schema_type == "SolutionDraft":
            lines.append(_render_solution_draft(data))
        elif schema_type == "Verdict":
            lines.append(_render_verdict(data))

    if run.tool_calls:
        tool_list = ", ".join(f"`{tc.tool}`" for tc in run.tool_calls)
        lines.append(f"**Tool calls:** {tool_list}\n\n")

    lines.append("---\n\n")
    return "".join(lines)


def _render_problem_spec(data: dict) -> str:
    domain = data.get("domain", "unknown")
    knowns = data.get("knowns", {})
    unknowns = data.get("unknowns", [])
    assumptions = data.get("assumptions", [])

    knowns_str = (
        ", ".join(
            f"{k} = {v.get('value', '?')} {v.get('unit', '') if isinstance(v, dict) else ''}"
            for k, v in knowns.items()
        )
        or "—"
    )
    unknowns_str = ", ".join(str(u) for u in unknowns) or "—"

    lines = [
        f"| Field | Value |\n|-------|-------|\n",
        f"| Domain | {domain} |\n",
        f"| Knowns | {knowns_str} |\n",
        f"| Unknowns | {unknowns_str} |\n\n",
    ]

    if assumptions:
        lines.append(".box {Assumptions} type:{tip}\n")
        for a in assumptions:
            lines.append(f"    - {a}\n")
        lines.append("\n")

    return "".join(lines)


def _render_research_brief(data: dict) -> str:
    principles = data.get("principles", [])
    equations = data.get("candidate_equations", [])
    citations = data.get("citations", [])
    lines: list[str] = []

    if principles:
        lines.append("**Principles:**\n")
        for p in principles:
            lines.append(f"- {p}\n")
        lines.append("\n")

    if equations:
        lines.append("**Candidate equations:**\n")
        for eq in equations:
            if not eq.strip().startswith("$"):
                eq = f"$ {eq} $"
            lines.append(f"- {eq}\n")
        lines.append("\n")

    if citations:
        lines.append(f"**Sources:** {', '.join(citations)}\n\n")

    return "".join(lines)


def _render_solution_draft(data: dict) -> str:
    method = data.get("method", "")
    steps = data.get("steps", [])
    final_answer = data.get("final_answer", "")
    code_trace = data.get("code_trace", [])
    lines: list[str] = []

    if method:
        lines.append(f"**Method:** {method}\n\n")
    if steps:
        lines.append("**Steps:**\n")
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}\n")
        lines.append("\n")
    if final_answer:
        fa = final_answer
        if isinstance(fa, dict):
            fa = f"{fa.get('value', '?')} {fa.get('unit', '')}".strip()
        lines.append(f"**Final answer:** {fa}\n\n")
    if code_trace:
        lines.append("**Code trace:**\n```python\n")
        for line in code_trace:
            lines.append(f"{line}\n")
        lines.append("```\n\n")

    return "".join(lines)


def _render_verdict(data: dict) -> str:
    approved = data.get("approved", False)
    issues = data.get("issues", [])
    severity = data.get("severity", "none")
    status = "approved ✓" if approved else f"rejected — severity: {severity}"
    lines = [f".box {{Verification}} type:{{tip}}\n    {status}\n"]
    for issue in issues:
        lines.append(f"    - {issue}\n")
    lines.append("\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f} min"


def _title_from_problem(problem: str) -> str:
    first = problem.split(".")[0].split("?")[0].strip()
    return first[:120]


def _domain_from_collector(collector: RunCollector) -> str:
    for run in collector.subagent_runs:
        for schema_type, data in run.schemas:
            if schema_type == "ProblemSpec":
                return str(data.get("domain", "unknown"))
    return "unknown"
```

- [ ] **Step 3.4: Run tests — expect pass**

```
uv run pytest tests/test_report.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 3.5: Run full suite to check no regressions**

```
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 3.6: Commit**

```bash
git add src/solvay/report.py tests/test_report.py
git commit -m "feat: add generate_quarkdown() report renderer"
```

---

## Task 4: Update consolidator prompt for Quarkdown output

**Files:**
- Modify: `src/solvay/prompts/consolidator.md`

- [ ] **Step 4.1: Append Quarkdown output format section**

Open `src/solvay/prompts/consolidator.md` and replace its entire content with:

```markdown
# Solvay Consolidator

You are the consolidator agent. Your job is to take all prior outputs from the
pipeline (problem spec, research brief, solution draft, peer review verdict)
and produce a final, polished answer.

## Input

All outputs from the pipeline so far, including:
- `ProblemSpec` from the parser
- `ResearchBrief` from the researcher
- `SolutionDraft` from the solver (with method, steps, final answer)
- `Verdict` from the peer reviewer

## Output

A clear, well-structured final answer written in **Quarkdown format** that:
1. States the method used
2. Lists the solution steps in logical order
3. Gives the final answer with proper units and significant figures
4. Notes any caveats or assumptions

## Output Format (Quarkdown)

Write your entire response in Quarkdown syntax. This renders to a formatted PDF/HTML report.

**Equations** — wrap all mathematical expressions in `$...$`:
- Inline: `The force is $ F = ma $`
- Display (own line): `$ v = \sqrt{2gh} $`

**Final answer box** — always wrap the final numeric result in:
```
.box {Answer} type:{tip}
    v = 14.0 m/s (downward)
```

**Caveats box** — if the peer reviewer flagged unresolved issues, add:
```
.box {Caveats} type:{warning}
    - Issue description here
```

**Steps** — use a standard numbered Markdown list (renders correctly in Quarkdown):
```
1. Identify knowns: $ h = 10\,\text{m} $, $ g = 9.81\,\text{m/s}^2 $
2. Apply $ v^2 = 2gh $
3. Compute: $ v = \sqrt{2 \times 9.81 \times 10} = 14.0\,\text{m/s} $
```

## Rules

- Synthesize; do not just concatenate the prior outputs.
- If the peer reviewer flagged unresolved issues, include them in a `.box {Caveats}` block.
- Present the answer at a level suitable for a physics student or instructor.
- All output in English.
- Always include the `.box {Answer}` block with the final numeric result and units.
```

- [ ] **Step 4.2: Verify prompt loads without errors**

```
uv run python -c "from solvay.subagents import load_prompt; print(load_prompt('consolidator')[:100])"
```

Expected: prints the first 100 chars of the prompt without error.

- [ ] **Step 4.3: Run test suite**

```
uv run pytest tests/test_prompts.py -v
```

Expected: all prompt tests pass.

- [ ] **Step 4.4: Commit**

```bash
git add src/solvay/prompts/consolidator.md
git commit -m "feat: update consolidator prompt for Quarkdown output format"
```

---

## Task 5: Wire everything into `cli.py`

**Files:**
- Modify: `src/solvay/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 5.1: Write failing CLI tests**

Append to `tests/test_cli.py`:

```python
import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from solvay.cli import app
from solvay.streaming import RunFinished, SubagentFinished, SubagentStarted


def _mock_events():
    return iter([
        SubagentStarted(name="parser", t=0.0),
        SubagentFinished(name="parser", duration_s=5.0),
        SubagentStarted(name="consolidator", t=5.0),
        SubagentFinished(name="consolidator", duration_s=3.0),
        RunFinished(total_s=8.0, final_answer="v = 14.0 m/s"),
    ])


class TestSolveReportGeneration:
    def test_report_always_written_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: _mock_events())
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: ".docname {Test}\n")

        result = CliRunner().invoke(app, ["solve", "test problem"])

        assert result.exit_code == 0, result.output
        qd_files = list(tmp_path.glob("solvay-report-*.qd"))
        assert len(qd_files) == 1
        assert ".docname" in qd_files[0].read_text()

    def test_report_written_to_custom_output_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: _mock_events())
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: ".docname {Test}\n")

        out = tmp_path / "my-report.qd"
        result = CliRunner().invoke(app, ["solve", "test problem", "--output", str(out)])

        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_verbose_flag_shows_tool_call_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from solvay.streaming import ToolCallMade, ToolResultReceived

        events = iter([
            SubagentStarted(name="parser", t=0.0),
            ToolCallMade(subagent="parser", tool="python_exec", args_preview="h=10"),
            ToolResultReceived(subagent="parser", tool="python_exec", result_preview="h=10.0"),
            SubagentFinished(name="parser", duration_s=5.0),
            RunFinished(total_s=5.0, final_answer="v = 14 m/s"),
        ])

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: events)
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: "")

        result = CliRunner().invoke(app, ["solve", "-v", "test problem"])

        assert "python_exec" in result.output
        assert "h=10" in result.output

    def test_non_verbose_hides_tool_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from solvay.streaming import ToolCallMade

        events = iter([
            SubagentStarted(name="parser", t=0.0),
            ToolCallMade(subagent="parser", tool="python_exec", args_preview="secret_arg"),
            SubagentFinished(name="parser", duration_s=5.0),
            RunFinished(total_s=5.0, final_answer="v = 14 m/s"),
        ])

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: events)
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: "")

        result = CliRunner().invoke(app, ["solve", "test problem"])

        assert "secret_arg" not in result.output

    def test_report_saved_line_in_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: _mock_events())
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: "")

        result = CliRunner().invoke(app, ["solve", "test problem"])

        assert "Report saved" in result.output
```

- [ ] **Step 5.2: Run tests — confirm they fail**

```
uv run pytest tests/test_cli.py::TestSolveReportGeneration -v
```

Expected: tests fail (missing `--output` flag, missing `parse_stream` import in cli.py).

- [ ] **Step 5.3: Rewrite `cli.py` with new `solve` flow**

Replace the content of `src/solvay/cli.py`. The key changes are:
1. Remove `_stream_verbose`
2. Add `_print_event` and `_schema_summary`
3. Update `solve` to always use `parse_stream`, add `--output` flag

Full new content of `src/solvay/cli.py`:

```python
"""Typer CLI for Solvay: solve problems and run benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

import os as _os
if not _os.environ.get("LANGSMITH_API_KEY"):
    _os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    _os.environ.setdefault("LANGSMITH_TRACING", "false")

app = typer.Typer(
    name="solvay",
    help="Multi-agent physics problem solver.",
    no_args_is_help=True,
)


def _notebook_content(files: Mapping[str, object]) -> str:
    """Return the run notebook content from native or legacy DeepAgents paths."""
    for path in ("/workspace/lab_notebook.md", "/lab_notebook.md"):
        entry = files.get(path)
        if isinstance(entry, dict):
            content = entry.get("content")
            if isinstance(content, str):
                return content
    return ""


def _schema_summary(schema_type: str, data: dict) -> str:
    if schema_type == "ProblemSpec":
        domain = data.get("domain", "?")
        unknowns = data.get("unknowns", [])
        knowns = list(data.get("knowns", {}).keys())
        return f"domain={domain}, unknowns={unknowns}, knowns={{{', '.join(knowns)}}}"
    if schema_type == "ResearchBrief":
        return (
            f"{len(data.get('principles', []))} principles, "
            f"{len(data.get('candidate_equations', []))} equations, "
            f"{len(data.get('citations', []))} citations"
        )
    if schema_type == "SolutionDraft":
        fa = data.get("final_answer", "?")
        if isinstance(fa, dict):
            fa = f"{fa.get('value', '?')} {fa.get('unit', '')}".strip()
        return f"method={data.get('method', '?')}, steps={len(data.get('steps', []))}, answer={fa}"
    if schema_type == "Verdict":
        return f"approved={data.get('approved', False)}, severity={data.get('severity', 'none')}"
    return str(data)[:80]


def _print_event(event: object, verbose: bool) -> None:
    from solvay.streaming import (
        RunFinished,
        SchemaProduced,
        SubagentFinished,
        SubagentStarted,
        ToolCallMade,
        ToolResultReceived,
    )

    if isinstance(event, SubagentStarted):
        typer.echo(f"  \033[36m⟳\033[0m {event.name}...")
    elif isinstance(event, SubagentFinished):
        typer.echo(f"  \033[32m✓\033[0m {event.name}  ({event.duration_s:.0f}s)")
    elif isinstance(event, RunFinished):
        typer.echo(f"\n\033[2mCompleted in {event.total_s / 60:.1f} min\033[0m\n")
    elif verbose:
        if isinstance(event, ToolCallMade):
            typer.echo(f"    \033[90m→\033[0m {event.tool}: {event.args_preview}")
        elif isinstance(event, ToolResultReceived):
            typer.echo(f"    \033[90m↳\033[0m {event.result_preview}")
        elif isinstance(event, SchemaProduced):
            summary = _schema_summary(event.schema_type, event.data)
            typer.echo(f"    \033[90m→ [{event.schema_type}]\033[0m {summary}")


@app.command()
def solve(
    statement: str | None = typer.Argument(None, help="The physics problem statement."),
    file: Path | None = typer.Option(None, "--file", "-f", help="Read problem from a text file."),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Override the model for all roles. Accepts any string understood "
            "by langchain.chat_models.init_chat_model, e.g. "
            "'anthropic:claude-sonnet-4-6', 'openai:gpt-4o', "
            "'google_genai:gemini-2.5-pro', or 'ollama:qwen3.5'. "
            "Takes precedence over the SOLVAY_MODEL env var."
        ),
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show subagent tool calls and intermediate outputs."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write Quarkdown report to this path (default: solvay-report-<timestamp>.qd).",
    ),
    trace: Path | None = typer.Option(
        None, "--trace", help="Dump full trace as JSON to this file."
    ),
) -> None:
    """Solve a physics problem using the multi-agent system."""
    if file is not None:
        if not file.exists():
            typer.echo(f"Error: file not found: {file}", err=True)
            raise typer.Exit(code=1)
        problem = file.read_text(encoding="utf-8").strip()
    elif statement is not None:
        problem = statement
    else:
        typer.echo("Error: provide a problem statement or --file.", err=True)
        raise typer.Exit(code=1)

    if not problem:
        typer.echo("Error: empty problem statement.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Solving: {problem[:80]}{'...' if len(problem) > 80 else ''}\n")

    from solvay.agent import create_solvay_agent
    from solvay.config import SolvayConfig
    from solvay.report import generate_quarkdown
    from solvay.streaming import RunCollector, parse_stream

    config = SolvayConfig(default_model=model)
    typer.echo(f"Model: {config.model_for('orchestrator')}\n")

    agent = create_solvay_agent(config)
    collector = RunCollector(problem=problem, model=config.model_for("orchestrator"))

    for event in parse_stream(agent, problem):
        _print_event(event, verbose=verbose)
        collector.accumulate(event)

    final_message = collector.final_answer

    typer.echo("=" * 60)
    typer.echo(final_message)
    typer.echo("=" * 60)

    qd = generate_quarkdown(collector)
    out_path = output or Path(f"solvay-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.qd")
    out_path.write_text(qd, encoding="utf-8")
    typer.echo(f"\nReport saved → {out_path}")

    if trace is not None:
        trace.write_text(json.dumps({"answer": final_message}, indent=2), encoding="utf-8")
        typer.echo(f"Trace saved to {trace}")


@app.command()
def chat(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the model for all roles.",
    ),
) -> None:
    """Interactive REPL — chat with the Solvay physics solver."""
    import time

    from solvay.agent import create_solvay_agent
    from solvay.config import SolvayConfig

    config = SolvayConfig(default_model=model)
    typer.echo(f"Solvay  model: {config.model_for('orchestrator')}")
    typer.echo("Type a physics problem. Ctrl+C to exit.\n")

    agent = create_solvay_agent(config)
    messages: list[dict[str, str]] = []

    while True:
        try:
            problem = input("\033[1;32mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nBye!")
            break

        if not problem:
            continue

        messages.append({"role": "user", "content": problem})
        final_content = ""
        seen_subagents: list[str] = []
        subagent_start = time.monotonic()

        try:
            for chunk in agent.stream(
                {"messages": list(messages)},
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            ):
                if not isinstance(chunk, dict):
                    continue

                ns = chunk.get("ns", ())
                data = chunk.get("data", {})

                if not isinstance(data, dict):
                    continue

                for _node, update in data.items():
                    if not isinstance(update, dict):
                        continue

                    raw_msgs = update.get("messages", [])
                    if hasattr(raw_msgs, "value"):
                        raw_msgs = raw_msgs.value
                    if not isinstance(raw_msgs, list):
                        continue

                    for msg in raw_msgs:
                        for tc in getattr(msg, "tool_calls", []):
                            if tc.get("name") == "task":
                                st = tc.get("args", {}).get("subagent_type", "")
                                if st and st not in seen_subagents:
                                    if seen_subagents:
                                        elapsed = time.monotonic() - subagent_start
                                        typer.echo(
                                            f"  \033[32m✓\033[0m {seen_subagents[-1]}"
                                            f"  ({elapsed:.0f}s)"
                                        )
                                    seen_subagents.append(st)
                                    subagent_start = time.monotonic()
                                    typer.echo(f"  \033[36m⟳\033[0m {st}...")

                        if not ns:
                            content = getattr(msg, "content", None)
                            if content and isinstance(content, str):
                                final_content = content

            if seen_subagents:
                elapsed = time.monotonic() - subagent_start
                typer.echo(
                    f"  \033[32m✓\033[0m {seen_subagents[-1]}  ({elapsed:.0f}s)"
                )

        except KeyboardInterrupt:
            typer.echo("\n\033[33mInterrupted.\033[0m\n")
            messages.pop()
            continue

        if final_content:
            typer.echo(f"\n\033[1;34mSolvay:\033[0m\n{final_content}\n")
            messages.append({"role": "assistant", "content": final_content})
        else:
            typer.echo("\033[33m(no response)\033[0m\n")
            messages.pop()


@app.command()
def bench(
    problems: Path = typer.Option(
        "benchmark/problems",
        "--problems",
        "-p",
        help="Root directory containing benchmark problem JSON files.",
    ),
    profiles: str = typer.Option(
        "solvay-full",
        "--profiles",
        help="Comma-separated profile names, or 'all'.",
    ),
    models: str = typer.Option(
        "",
        "--models",
        help=(
            "Comma-separated model strings for langchain.chat_models.init_chat_model, "
            "for example 'ollama:qwen3.5'. Defaults to Solvay's default model."
        ),
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help="Filter problems to this domain subdirectory, e.g. mechanics.",
    ),
    repeat: int = typer.Option(1, "--repeat", min=1, help="Repeats per matrix cell."),
    confirm_cost: bool = typer.Option(
        False,
        "--confirm-cost",
        help="Confirm running a matrix above the benchmark cost threshold.",
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output results JSONL file."),
) -> None:
    """Run the benchmark matrix with the new solvay.benchmark runner."""
    from solvay.benchmark.cli import _import_builtin_profiles
    from solvay.benchmark.config import BenchConfig
    from solvay.benchmark.profiles import PROFILES
    from solvay.benchmark.runner import MatrixSpec, run_matrix
    from solvay.benchmark.schema import load_problems_dir
    from solvay.config import DEFAULT_MODEL

    root = problems / domain if domain else problems
    if not root.exists():
        typer.echo(f"Error: problems directory not found: {root}", err=True)
        raise typer.Exit(code=1)

    _import_builtin_profiles()
    all_problems = load_problems_dir(root)
    if not all_problems:
        typer.echo(f"No problems found under {root}.", err=True)
        raise typer.Exit(code=1)

    profile_names = (
        list(PROFILES) if profiles == "all" else [p.strip() for p in profiles.split(",")]
    )
    for profile_name in profile_names:
        if profile_name not in PROFILES:
            typer.echo(
                f"Unknown profile: {profile_name}. Known: {sorted(PROFILES)}",
                err=True,
            )
            raise typer.Exit(code=2)

    model_list = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
    config = BenchConfig()
    invocations = len(all_problems) * len(profile_names) * len(model_list) * repeat
    if invocations > config.cost_guard_threshold and not confirm_cost:
        typer.echo(
            f"Matrix would make {invocations} invocations "
            f"(threshold {config.cost_guard_threshold}). "
            "Pass --confirm-cost to proceed.",
            err=True,
        )
        raise typer.Exit(code=3)

    out_path = out or Path("benchmark/runs") / "latest.jsonl"
    spec = MatrixSpec(
        problems=all_problems,
        profile_names=profile_names,
        models=model_list,
        repeats=repeat,
    )
    path = run_matrix(spec, out_path=out_path, config=config)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5.4: Run the new CLI tests**

```
uv run pytest tests/test_cli.py::TestSolveReportGeneration -v
```

Expected: all 5 tests pass.

- [ ] **Step 5.5: Run full test suite**

```
uv run pytest tests/ -q
```

Expected: all tests pass (102+ passing).

- [ ] **Step 5.6: Commit**

```bash
git add src/solvay/cli.py tests/test_cli.py
git commit -m "feat: wire parse_stream + generate_quarkdown into solve command"
```

---

## Task 6: Final spec and plan update

- [ ] **Step 6.1: Update spec open questions as resolved**

The open question in the spec about `stream_events` v3 availability is already resolved (deepagents 0.6.2 confirmed). No further spec changes needed.

- [ ] **Step 6.2: Commit spec and plan**

```bash
git add docs/superpowers/specs/2026-05-20-verbose-quarkdown-design.md
git add docs/superpowers/plans/2026-05-20-verbose-quarkdown.md
git commit -m "docs: finalize verbose+quarkdown spec and implementation plan"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `StreamEvent` types (Task 1)
- ✅ `RunCollector.accumulate()` (Task 1)
- ✅ `parse_stream()` with `stream_events(version="v3")` (Task 1)
- ✅ `tool_call.output_deltas` live streaming (Task 1 `parse_stream`)
- ✅ Schema detection via `_try_detect_schema` (Task 1)
- ✅ `generate_quarkdown()` with all section renderers (Task 3)
- ✅ Consolidator prompt Quarkdown instructions (Task 4)
- ✅ `--output` flag with timestamped default (Task 5)
- ✅ `_print_event` verbose/non-verbose distinction (Task 5)
- ✅ Report always generated (Task 5)
- ✅ Verbose=False shows ⟳/✓ but not tool calls (Task 5 tests)

**Type consistency:**
- `ToolCallRecord` defined in Task 1, used in `SubagentRun` (same task), imported in `report.py` (Task 3) ✅
- `RunCollector` constructed in `cli.py` with `problem=` and `model=` kwargs matching `@dataclass` fields ✅
- `parse_stream(agent, problem)` signature matches call site in `cli.py` ✅
- `generate_quarkdown(collector)` — single arg, matches call in `cli.py` ✅

**No placeholders:** All code blocks are complete and runnable. ✅
