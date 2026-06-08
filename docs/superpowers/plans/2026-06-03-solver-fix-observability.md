# Solver Fix + Orchestrator Observability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the solver CompiledSubAgent crash (Pydantic validation error when deepagents passes only `messages`) and add `OrchestratorDispatched` events so we can see the orchestrator's decisions.

**Architecture:** Two independent changes. The solver fix adds a `prepare` node to the LangGraph that parses `problem_spec`/`research_brief` from the HumanMessage, and makes both fields optional. The observability fix adds a new `OrchestratorDispatched` StreamEvent extracted from the v3 stream's `"messages"` channel.

**Tech Stack:** LangGraph StateGraph, Pydantic BaseModel, deepagents v3 streaming API.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/solvay/subagents/solver_loop.py` | Modify | Add `prepare` node, make state fields optional |
| `tests/test_solver_loop.py` | Modify | Add tests for `prepare` node parsing |
| `src/solvay/streaming.py` | Modify | Add `OrchestratorDispatched` dataclass, capture in `parse_stream()` |
| `src/solvay/cli.py` | Modify | Display `OrchestratorDispatched` in `_print_event()` |
| `tests/test_streaming.py` | Modify | Add tests for orchestrator event capture and RunCollector |

---

## Task 1: Make solver state fields optional and add prepare node

**Files:**
- Modify: `src/solvay/subagents/solver_loop.py:23-54` (state), `56-247` (graph)
- Test: `tests/test_solver_loop.py`

- [ ] **Step 1: Write the failing tests**

Add these test classes to `tests/test_solver_loop.py` at the end of the file:

```python
import re

from langchain_core.messages import HumanMessage

from solvay.subagents.solver_loop import SolverLoopGraphState


class TestPrepareNode:
    """Tests for the prepare node that parses description into state fields."""

    def _make_message_only_input(self, description: str) -> dict[str, object]:
        """Simulate how deepagents task tool passes input to CompiledSubAgents."""
        return {
            "messages": [HumanMessage(content=description)],
        }

    def test_state_accepts_empty_problem_spec_and_research_brief(self) -> None:
        """SolverLoopGraphState must not crash when fields are missing."""
        state = SolverLoopGraphState(messages=[HumanMessage(content="test")])
        assert state.problem_spec == {}
        assert state.research_brief == {}

    def test_prepare_parses_valid_json_blocks(self) -> None:
        spec = '{"statement": "A ball drops", "domain": "mechanics"}'
        brief = '{"principles": ["F=ma"], "candidate_equations": []}'
        description = f"Problem spec:\n{spec}\n\nResearch:\n{brief}"

        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(self._make_message_only_input(description))
        assert result["problem_spec"]["domain"] == "mechanics"
        assert result["research_brief"]["principles"] == ["F=ma"]
        assert result["termination_reason"] == "consensus"

    def test_prepare_handles_malformed_json(self) -> None:
        description = "Problem spec:\n{not valid json}\n\nResearch:\n{also bad}"

        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(self._make_message_only_input(description))
        assert result["problem_spec"].get("raw_description") is not None
        assert result["termination_reason"] == "consensus"

    def test_prepare_skips_when_fields_already_populated(self) -> None:
        inputs = _make_inputs()

        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(inputs)
        assert result["problem_spec"]["domain"] == "mechanics"
        assert result["termination_reason"] == "consensus"

    def test_prepare_handles_empty_messages(self) -> None:
        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke({"messages": []})
        assert result["problem_spec"] == {}
        assert result["termination_reason"] == "consensus"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_solver_loop.py::TestPrepareNode -v`
Expected: FAIL — `SolverLoopGraphState` requires `problem_spec` and `research_brief`

- [ ] **Step 3: Make state fields optional**

In `src/solvay/subagents/solver_loop.py`, change lines 31-32 of `SolverLoopGraphState`:

```python
class SolverLoopGraphState(BaseModel):
    """State for the solver loop LangGraph subgraph."""

    messages: list[Any] = Field(default_factory=list)

    # Inputs — optional because deepagents task tool only passes messages.
    # The prepare node populates these from the HumanMessage description.
    problem_spec: dict[str, Any] = Field(default_factory=dict)
    research_brief: dict[str, Any] = Field(default_factory=dict)

    # Loop control
    iteration: int = 0
    max_iterations: int = 3

    # Current round outputs
    current_draft: dict[str, Any] | None = None
    verifier_verdict: dict[str, Any] | None = None
    reviewer_verdict: dict[str, Any] | None = None

    # Accumulated history
    critique_history: list[dict[str, Any]] = Field(default_factory=list)

    # Blocking signal
    solver_blocked: bool = False
    blocked_topic: str | None = None

    # Final output
    final_draft: dict[str, Any] | None = None
    termination_reason: str | None = None
    unresolved_blockers: bool = False
```

- [ ] **Step 4: Add the prepare node and parsing helper**

In `src/solvay/subagents/solver_loop.py`, add this import at the top (after the existing `import re` if present, or add `import re`):

```python
import re
```

Then add the `_parse_description` helper and `prepare` node inside `build_solver_loop_graph()`, right before the existing `solve` function definition (after line 113 `return {}` of `_invoke_structured`):

```python
    def _parse_description(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract problem_spec and research_brief JSON from a description string.

        Expected format:
            Problem spec:\n{...JSON...}\n\nResearch:\n{...JSON...}
        """
        problem_spec: dict[str, Any] = {}
        research_brief: dict[str, Any] = {}

        spec_match = re.search(
            r"Problem spec:\s*\n(\{.*?\})\s*(?:\n\n|$)",
            text,
            re.DOTALL,
        )
        if spec_match:
            try:
                problem_spec = json.loads(spec_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        brief_match = re.search(
            r"Research:\s*\n(\{.*?\})\s*$",
            text,
            re.DOTALL,
        )
        if brief_match:
            try:
                research_brief = json.loads(brief_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        if not problem_spec:
            problem_spec = {"raw_description": text}

        return problem_spec, research_brief

    def prepare(state: SolverLoopGraphState) -> dict[str, Any]:
        """Parse problem_spec and research_brief from messages if not already set."""
        if state.problem_spec and state.research_brief:
            return {}

        if not state.messages:
            return {}

        first_msg = state.messages[0]
        content = getattr(first_msg, "content", "")
        if not isinstance(content, str):
            content = str(content)

        problem_spec, research_brief = _parse_description(content)

        updates: dict[str, Any] = {}
        if not state.problem_spec:
            updates["problem_spec"] = problem_spec
        if not state.research_brief:
            updates["research_brief"] = research_brief
        return updates
```

- [ ] **Step 5: Update graph topology**

In `build_solver_loop_graph()`, update the graph builder (around line 236) to add the `prepare` node and wire it in:

```python
    builder = StateGraph(SolverLoopGraphState)

    builder.add_node("prepare", prepare)
    builder.add_node("solve", solve)
    builder.add_node("critique", critique)
    builder.add_node("judge", judge)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "solve")
    builder.add_edge("solve", "critique")
    builder.add_edge("critique", "judge")
    builder.add_conditional_edges("judge", route_after_judge, ["solve", END])

    return builder.compile()
```

(Replace the old builder block that had `builder.add_edge(START, "solve")`)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_solver_loop.py -v`
Expected: ALL PASS (existing tests + 5 new TestPrepareNode tests)

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest -x -q`
Expected: 162+ passed (157 existing + 5 new)

- [ ] **Step 8: Commit**

```bash
git add src/solvay/subagents/solver_loop.py tests/test_solver_loop.py
git commit -m "fix: add prepare node to solver graph for description parsing

The solver CompiledSubAgent crashed because deepagents task tool
passes only messages, but SolverLoopGraphState required problem_spec
and research_brief as mandatory dict fields.

Fix: make both fields optional (default_factory=dict) and add a
prepare node that parses them from the HumanMessage description
before the solve loop begins."
```

---

## Task 2: Add OrchestratorDispatched StreamEvent

**Files:**
- Modify: `src/solvay/streaming.py:15-61` (events), `54-61` (StreamEvent union), `89-116` (RunCollector), `234-239` (messages channel)
- Test: `tests/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_streaming.py` at the end of the file:

```python
from solvay.streaming import OrchestratorDispatched


@_dc
class _FakeMsgWithToolCalls:
    text: str
    tool_calls: list[dict[str, _Any]] = _field(default_factory=list)


class TestOrchestratorDispatched:
    def _make_agent(self, stream: _FakeStream) -> _FakeAgent:
        return _FakeAgent(stream)

    def test_orchestrator_dispatch_captured_from_messages(self) -> None:
        msg = _FakeMsgWithToolCalls(
            text="",
            tool_calls=[{
                "name": "task",
                "args": {"subagent_type": "parser", "description": "A ball drops from 10m"},
            }],
        )
        stream = _FakeStream(
            subagents=[_FakeSubagent(name="parser")],
            messages=[msg],
        )
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        dispatches = [e for e in events if isinstance(e, OrchestratorDispatched)]
        assert len(dispatches) == 1
        assert dispatches[0].subagent_type == "parser"

    def test_orchestrator_dispatch_description_truncated(self) -> None:
        long_desc = "x" * 200
        msg = _FakeMsgWithToolCalls(
            text="",
            tool_calls=[{
                "name": "task",
                "args": {"subagent_type": "solver", "description": long_desc},
            }],
        )
        stream = _FakeStream(messages=[msg])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        dispatches = [e for e in events if isinstance(e, OrchestratorDispatched)]
        assert len(dispatches) == 1
        assert len(dispatches[0].description_preview) <= 123  # 120 + "..."

    def test_non_task_tool_calls_ignored(self) -> None:
        msg = _FakeMsgWithToolCalls(
            text="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "test"},
            }],
        )
        stream = _FakeStream(messages=[msg])
        agent = self._make_agent(stream)
        events = list(parse_stream(agent, "test"))
        dispatches = [e for e in events if isinstance(e, OrchestratorDispatched)]
        assert len(dispatches) == 0


class TestRunCollectorOrchestratorDispatched:
    def test_accumulate_orchestrator_dispatched(self) -> None:
        c = RunCollector(problem="test", model="test-model")
        c.accumulate(OrchestratorDispatched(subagent_type="parser", description_preview="test", t=0.0))
        c.accumulate(OrchestratorDispatched(subagent_type="solver", description_preview="test", t=1.0))
        assert len(c.orchestrator_dispatches) == 2
        assert c.orchestrator_dispatches[0].subagent_type == "parser"
        assert c.orchestrator_dispatches[1].subagent_type == "solver"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py::TestOrchestratorDispatched -v`
Expected: FAIL — `ImportError: cannot import name 'OrchestratorDispatched'`

- [ ] **Step 3: Add the OrchestratorDispatched dataclass**

In `src/solvay/streaming.py`, add the new dataclass after `RunFinished` (around line 52):

```python
@dataclass
class OrchestratorDispatched:
    subagent_type: str
    description_preview: str  # first ~120 chars
    t: float
```

Update the `StreamEvent` union type (around line 54) to include it:

```python
StreamEvent = Union[
    SubagentStarted,
    ToolCallMade,
    ToolResultReceived,
    SchemaProduced,
    SubagentFinished,
    RunFinished,
    OrchestratorDispatched,
]
```

- [ ] **Step 4: Add orchestrator_dispatches to RunCollector**

In `src/solvay/streaming.py`, update the `RunCollector` class. Add the field after `final_answer`:

```python
@dataclass
class RunCollector:
    problem: str
    model: str
    subagent_runs: list[SubagentRun] = field(default_factory=list)
    total_s: float = 0.0
    final_answer: str = ""
    orchestrator_dispatches: list[OrchestratorDispatched] = field(default_factory=list)
```

Add handling in the `accumulate` method, after the `RunFinished` check:

```python
        elif isinstance(event, RunFinished):
            self.total_s = event.total_s
            self.final_answer = event.final_answer
        elif isinstance(event, OrchestratorDispatched):
            self.orchestrator_dispatches.append(event)
```

- [ ] **Step 5: Capture orchestrator dispatches in parse_stream()**

In `src/solvay/streaming.py`, modify the `elif channel == "messages"` block (around line 234) to extract tool_calls:

```python
        elif channel == "messages":
            msg = item
            for tc in getattr(msg, "tool_calls", []):
                if isinstance(tc, dict) and tc.get("name") == "task":
                    args = tc.get("args", {})
                    yield OrchestratorDispatched(
                        subagent_type=args.get("subagent_type", "?"),
                        description_preview=_truncate(
                            args.get("description", ""), 120
                        ),
                        t=time.monotonic(),
                    )
            text = str(msg.text)
            if text and len(text) > 20 and not final_answer:
                final_answer = text
```

- [ ] **Step 6: Run streaming tests to verify they pass**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: ALL PASS (existing + 4 new)

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 8: Commit**

```bash
git add src/solvay/streaming.py tests/test_streaming.py
git commit -m "feat: add OrchestratorDispatched StreamEvent for decision audit

Captures the orchestrator's task() tool calls from the v3 stream
messages channel. Each dispatch yields subagent_type, description
preview, and timestamp. Accumulated in RunCollector for reporting.

Makes orchestrator routing decisions visible — previously we could
not see why parser was called 8 times or why solver was skipped."
```

---

## Task 3: Display OrchestratorDispatched in CLI

**Files:**
- Modify: `src/solvay/cli.py:65-89`
- Test: manual verification

- [ ] **Step 1: Update _print_event()**

In `src/solvay/cli.py`, add the import and handler. Update the import block inside `_print_event()` (line 66-73):

```python
def _print_event(event: object, verbose: bool) -> None:
    from solvay.streaming import (
        OrchestratorDispatched,
        RunFinished,
        SchemaProduced,
        SubagentFinished,
        SubagentStarted,
        ToolCallMade,
        ToolResultReceived,
    )

    if isinstance(event, OrchestratorDispatched):
        typer.echo(f"  \033[33m▶\033[0m orchestrator → {event.subagent_type}")
    elif isinstance(event, SubagentStarted):
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
```

Note: `OrchestratorDispatched` is shown **always** (not gated behind `verbose`), since it's a top-level event like SubagentStarted.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 3: Commit**

```bash
git add src/solvay/cli.py
git commit -m "feat: display orchestrator dispatch events in CLI output

Shows '▶ orchestrator → solver' when the orchestrator dispatches
a subagent. Always visible (not gated behind --verbose) since
these are top-level decision events."
```

---

## Final Verification

- [ ] **Run full test suite**: `uv run pytest -v` — all passing
- [ ] **Verify solver can be invoked with messages-only state**: existing TestPrepareNode tests cover this
- [ ] **Verify OrchestratorDispatched events flow through RunCollector**: TestRunCollectorOrchestratorDispatched covers this
