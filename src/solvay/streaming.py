"""Typed StreamEvent protocol and RunCollector for the Solvay pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

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
    data: dict[str, Any]  # full parsed dict


@dataclass
class SubagentFinished:
    name: str
    duration_s: float


@dataclass
class RunFinished:
    total_s: float
    final_answer: str


@dataclass
class OrchestratorDispatched:
    subagent_type: str
    description_preview: str  # first ~120 chars
    t: float


StreamEvent = (
    SubagentStarted
    | ToolCallMade
    | ToolResultReceived
    | SchemaProduced
    | SubagentFinished
    | RunFinished
    | OrchestratorDispatched
)

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
    schemas: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


@dataclass
class RunCollector:
    problem: str
    model: str
    subagent_runs: list[SubagentRun] = field(default_factory=list)
    total_s: float = 0.0
    final_answer: str = ""
    orchestrator_dispatches: list[OrchestratorDispatched] = field(default_factory=list)

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
        elif isinstance(event, OrchestratorDispatched):
            self.orchestrator_dispatches.append(event)


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


def parse_stream(agent: Any, problem: str) -> Iterator[StreamEvent]:
    """Yield StreamEvents by consuming agent.stream_events(version='v3').

    Uses the deepagents 0.6+ v3 streaming API:
        stream.interleave("messages", "subagents") -> coordinator messages and
            subagent handles in arrival order; both share the same pump so they
            must be interleaved rather than iterated sequentially.
        subagent.interleave("tool_calls", "messages") -> tool calls and final
            messages from each subagent in arrival order.
        tool_call.output_deltas -> live output fragments
        tool_call.output  -> full result string when complete
        tool_call.error   -> set if the tool raised

    Note: msg.text is a SyncTextProjection, not a str; use str(msg.text).
    """
    import time

    t0 = time.monotonic()
    final_answer = ""
    last_dispatched: str | None = None

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": problem}]},
        version="v3",
    )

    subagent_starts: dict[str, float] = {}

    for channel, item in stream.interleave("messages", "subagents"):
        if channel == "subagents":
            subagent = item
            t_start = time.monotonic()
            subagent_starts[subagent.name] = t_start
            yield SubagentStarted(name=subagent.name, t=t_start)

            for inner_channel, inner_item in subagent.interleave("tool_calls", "messages"):
                if inner_channel == "tool_calls":
                    tool_call = inner_item
                    args_preview = _truncate(str(tool_call.input), 80)
                    yield ToolCallMade(
                        subagent=subagent.name,
                        tool=tool_call.tool_name,
                        args_preview=args_preview,
                    )

                    output_parts: list[str] = []
                    for delta in tool_call.output_deltas:
                        output_parts.append(str(delta))

                    full_output = tool_call.output if tool_call.output else "".join(output_parts)
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

                elif inner_channel == "messages":
                    msg = inner_item
                    # msg.text is SyncTextProjection; str() drives the pump and
                    # returns the full accumulated text.
                    text = str(msg.text)
                    if text.strip().startswith("{"):
                        schema_evt = _try_detect_schema(subagent.name, text)
                        if schema_evt:
                            yield schema_evt
                    if subagent.name == "consolidator" and text and len(text) > 20:
                        final_answer = text

            yield SubagentFinished(
                name=subagent.name,
                duration_s=time.monotonic() - subagent_starts.get(subagent.name, t0),
            )
            last_dispatched = None

        elif channel == "messages":
            msg = item
            for tc in getattr(msg, "tool_calls", []):
                if isinstance(tc, dict) and tc.get("name") == "task":
                    raw_args = tc.get("args", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            raw_args = {}
                    if isinstance(raw_args, dict):
                        st = raw_args.get("subagent_type")
                        if st and st != last_dispatched:
                            last_dispatched = st
                            yield OrchestratorDispatched(
                                subagent_type=st,
                                description_preview=_truncate(
                                    raw_args.get("description", ""), 120
                                ),
                                t=time.monotonic(),
                            )
            text = str(msg.text)
            if text and len(text) > 20 and not final_answer:
                final_answer = text

    yield RunFinished(total_s=time.monotonic() - t0, final_answer=final_answer)
