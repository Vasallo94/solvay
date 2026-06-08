# Solvay v0.7.1: Solver Fix + Orchestrator Observability

**Date**: 2026-06-03
**Status**: Draft
**Scope**: Fix solver CompiledSubAgent input parsing + add OrchestratorDispatched events

---

## Problem Statement

### Problem 1: Solver subagent fails with schema validation error

The solver is the only `CompiledSubAgent` in Solvay. Its LangGraph state (`SolverLoopGraphState`) requires `problem_spec: dict` and `research_brief: dict` as mandatory fields. However, deepagents' task tool prepares subagent state by wrapping the description as a `HumanMessage` in `messages` — it never populates custom state fields.

**Observed behavior** (from `.solvay/memories/solvay/harness_notes.md`):
> Solver agent consistently fails with "description: Input should be a valid string" error...
> Workaround: used general-purpose agent instead.

**Root cause**: `_validate_and_prepare_state()` in deepagents/middleware/subagents.py (line 409) sets `subagent_state["messages"] = [HumanMessage(content=description)]` but does not extract or map structured data from the description string into custom state fields like `problem_spec` and `research_brief`.

### Problem 2: Orchestrator decisions are invisible

The orchestrator calls the `task` tool to dispatch subagents. These calls contain `subagent_type` and `description` arguments — the key decision points in the pipeline. But `parse_stream()` in streaming.py only captures subagent lifecycle events (start/finish) and tool calls within subagents. The orchestrator's own tool calls are in the v3 stream's `"messages"` channel but are discarded (line 234-239 only extracts final answer text).

This made it impossible to diagnose why the parser was called 8 times or why solver was never called — those were orchestrator decisions we couldn't see.

## Fix 1: Solver Input Parsing

### Design

Add a `prepare` node to the solver LangGraph that runs before `solve`. This node:
1. Checks if `problem_spec` and `research_brief` are already populated (direct invocation)
2. If empty, reads the first `HumanMessage` from `messages` and parses it for JSON blocks
3. Extracts `problem_spec` and `research_brief` dicts from the text
4. Populates the state fields for downstream nodes

### State schema change

Make `problem_spec` and `research_brief` optional with `default_factory=dict`:

```python
class SolverLoopGraphState(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    problem_spec: dict[str, Any] = Field(default_factory=dict)
    research_brief: dict[str, Any] = Field(default_factory=dict)
    # ... rest unchanged
```

This prevents Pydantic validation failure when deepagents passes only `messages`.

### Graph topology change

```
Before: START → solve → critique → judge → {solve | END}
After:  START → prepare → solve → critique → judge → {solve | END}
```

### Parsing strategy

The orchestrator passes descriptions like:
```
Problem spec:\n{...JSON...}\n\nResearch:\n{...JSON...}
```

The `prepare` node splits on known markers ("Problem spec:", "Research:") and JSON-parses each block. If parsing fails, it passes the raw description text as `problem_spec["statement"]` so the solver LLM can still work with it.

### Files changed

| File | Change |
|------|--------|
| `src/solvay/subagents/solver_loop.py` | Add `prepare` node, make state fields optional, update graph edges |
| `tests/test_solver_loop.py` | Add tests for `prepare` node parsing |

## Fix 2: OrchestratorDispatched Events

### Design

Add a new `StreamEvent` that captures the orchestrator's `task()` tool calls.

### New dataclass

```python
@dataclass
class OrchestratorDispatched:
    subagent_type: str
    description_preview: str  # first 120 chars
    t: float
```

### Changes to parse_stream()

In the `elif channel == "messages"` branch, inspect message objects for tool_calls before extracting final_answer text:

```python
elif channel == "messages":
    msg = item
    for tc in getattr(msg, "tool_calls", []):
        if tc.get("name") == "task":
            yield OrchestratorDispatched(
                subagent_type=tc["args"].get("subagent_type", "?"),
                description_preview=_truncate(tc["args"].get("description", ""), 120),
                t=time.monotonic(),
            )
    text = str(msg.text)
    if text and len(text) > 20 and not final_answer:
        final_answer = text
```

### CLI display

In verbose mode, show orchestrator dispatches:
```
  ▶ orchestrator → parser
  ⟳ parser...
  ✓ parser  (27s)
  ▶ orchestrator → solver
  ⟳ solver...
```

### RunCollector integration

Add `orchestrator_dispatches: list[OrchestratorDispatched]` to `RunCollector` and include in `accumulate()`. The Quarkdown report can then show the orchestrator's decision sequence.

### Files changed

| File | Change |
|------|--------|
| `src/solvay/streaming.py` | Add `OrchestratorDispatched` dataclass, modify `parse_stream()` |
| `src/solvay/cli.py` | Add display for `OrchestratorDispatched` in `_print_event()` |
| `tests/test_streaming.py` | Add test for orchestrator event capture |

## Testing Strategy

- All existing 157 tests must continue passing
- New tests for `prepare` node: valid JSON, malformed JSON, missing markers, empty description
- New tests for `OrchestratorDispatched`: verify events are yielded from mocked stream
- Integration test: run solver subagent directly with a HumanMessage-only state to verify it no longer crashes

## Non-Goals

- Full --audit flag (future work)
- Full tool I/O expansion from 80 chars (future work)
- LangSmith integration changes
- Changes to the consolidator or report format
