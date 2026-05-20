# Verbose Step-by-Step Output & Quarkdown Report — Design Spec

**Date:** 2026-05-20
**Status:** Approved

---

## Overview

Two new features that share the same streaming infrastructure:

1. **Verbose live output** (`-v`): show tool calls and intermediate schemas inside each subagent as they happen, not just the subagent name and timing.
2. **Quarkdown report** (`.qd`): always generate a structured academic-style paper after each `solve` run, capturing the full pipeline with LaTeX equations, tables, and metadata.

---

## Motivation

During active development and debugging of the multi-agent pipeline, operators need visibility into what each subagent is actually doing — which tools it calls, what data it produced, whether schemas look correct. The existing `-v` flag only shows subagent names and wall-clock times.

The Quarkdown report serves as a deliverable: a self-contained paper documenting the problem, the reasoning pipeline, and the solution, suitable for review or archival.

---

## Architecture

### Files changed

| File | Change |
|------|--------|
| `src/solvay/streaming.py` | New — typed `StreamEvent` union, `parse_stream()` generator, `RunCollector` |
| `src/solvay/report.py` | New — `generate_quarkdown(collector, config) -> str` |
| `src/solvay/cli.py` | Modified — `_stream_verbose` reimplemented, `--output` flag added to `solve` |
| `src/solvay/prompts/consolidator.md` | Modified — Quarkdown formatting instructions |
| `tests/test_streaming.py` | New — unit tests for event parsing and RunCollector |
| `tests/test_report.py` | New — unit tests for Quarkdown generation |

### Data flow

```
agent.stream_events({"messages": [...]}, version="v3")
        │  (deepagents high-level streaming API)
        ▼
parse_stream(agent, problem) → Iterator[StreamEvent]
        │  iterates stream.subagents → tool_calls → output_deltas live
        │
    for event in parse_stream(...):
        ├── print_event(event)        # Feature 1: live terminal output
        └── collector.accumulate(event)
        │
        ▼  (stream complete)
generate_quarkdown(collector, config) → str → <output>.qd   # Feature 2: always
```

---

## Feature 1: Verbose Live Output

### Stream API

`_stream_verbose` currently uses `stream_mode="debug"`, which delivers `task_result` events post-hoc. The reimplementation uses the deepagents `stream_events` v3 API, which exposes a high-level structured interface over the subgraph stream:

```python
stream = agent.stream_events({"messages": [...]}, version="v3")

for subagent in stream.subagents:          # yields as each subagent starts
    subagent.name                          # "parser", "researcher", etc.
    for tool_call in subagent.tool_calls:  # yields as each tool fires
        tool_call.tool_name
        for delta in tool_call.output_deltas:  # live streaming deltas
            ...
        tool_call.output   # full result when complete
        tool_call.error    # set if the tool raised
```

This eliminates the need to parse raw LangGraph namespace tuples (`("tools:<call_id>",)`) and correlate them with subagent names — the v3 API resolves names directly. `subagent.name` maps 1:1 to the deepagents subagent `name` field (e.g., `"parser"`, `"researcher"`).

### Terminal output format

```
Solving: A ball is dropped from 10m...

Model: anthropic:claude-sonnet-4-6

  ⟳ parser...
    → python_exec: "from sympy import symbols..."
    ↳ stdout: "h = 10.0 m, g = 9.81 m/s²" (12ms)
    → [ProblemSpec] domain=mechanics, unknowns=['v_final'], knowns={h:10m, g:9.81m/s²}
  ✓ parser  (8s)

  ⟳ researcher...
    → web_search: "kinematic equations free fall"
    ↳ 3 results: HyperPhysics §2.3, Wikipedia "Kinematics"...
    → [ResearchBrief] 3 principles, 4 equations, 2 citations
  ✓ researcher  (31s)

  ⟳ solver [iter 1]...
    → python_exec: "v = sqrt(2 * 9.81 * 10)"
    ↳ stdout: "v = 14.007 m/s" (8ms)
    → [SolutionDraft] method=kinematic, steps=4, answer=14.0 m/s
  ⟳ verifier...
    → [Verdict] approved=True, severity=none
  ✓ solver  (45s)

  ⟳ consolidator...
  ✓ consolidator  (9s)

Completed in 1.6 min

============================================================
[final answer text]
============================================================
Report saved → report.qd
```

**Tool call lines** (`→`): tool name + first ~80 chars of args, truncated with `...`.

**Tool result lines** (`↳`): first ~80 chars of result. If the result is a known Pydantic schema (ProblemSpec, ResearchBrief, SolutionDraft, Verdict), emit a `[SchemaName]` summary line instead.

**Schema detection**: attempt JSON parse of AI message content; if it matches a known schema by field names, emit `SchemaProduced` event.

---

## Feature 2: Quarkdown Report

### Always generated

The `.qd` report is generated after every `solve` run (not behind a feature flag). One report per run — default filename is timestamped: `solvay-report-YYYYMMDD-HHMMSS.qd` in the current working directory. Configurable via `--output <path>` on the `solve` command. A `--output` value without a `.qd` extension is accepted as-is.

### `--output` flag

```
uv run solvay solve -v "problem" --output my-report.qd
uv run solvay solve "problem"   # writes solvay-report-20260520-143201.qd
```

The Makefile `RUN` variable does not need updating; the timestamped default is sufficient.

### Document structure

```quarkdown
.docname {Solvay Solution Report}
.doctype {plain}
.doclang {English}
.theme {paperwhite} layout:{latex}
.numbering
    - headings: 1

# <first sentence of problem as title>

**Domain:** mechanics | **Model:** claude-sonnet-4-6 | **Date:** 2026-05-20 | **Duration:** 1.6 min

---

## Problem Statement

<full problem text>

---

## 1 · Problem Parsing

*parser — 8s*

| Field    | Value                      |
|----------|----------------------------|
| Domain   | mechanics                  |
| Knowns   | h = 10.0 m, g = 9.81 m/s² |
| Unknowns | v_final                    |

.box {Assumptions} type:{tip}
    - frictionless fall
    - point mass approximation

**Tool calls:** `python_exec` (12ms)

---

## 2 · Research

*researcher — 31s*

**Principles:**
- Conservation of energy: $ E_k = E_p $
- Kinematic equation: $ v^2 = 2gh $

**Tool calls:** `web_search` ×2 → HyperPhysics §2.3, Wikipedia "Kinematics"...

---

## 3 · Solution

*solver — 45s (1 iteration)*

**Method:** Kinematic equations

**Steps:**
1. Identify knowns: $ h = 10\,\text{m},\; g = 9.81\,\text{m/s}^2 $
2. Apply $ v^2 = 2gh $
3. Compute: $ v = \sqrt{2 \times 9.81 \times 10} = 14.0\,\text{m/s} $

**Code trace:**
```python
v = sqrt(2 * 9.81 * 10)  # → 14.007 m/s
```

.box {Verification} type:{tip}
    Verifier: approved — severity: none
    Peer reviewer: approved — minor: check significant figures

---

## 4 · Final Answer

<!-- written by consolidator in Quarkdown format -->

---

## Run Metadata

| Subagent     | Duration |
|--------------|----------|
| parser       | 8s       |
| researcher   | 31s      |
| solver       | 45s      |
| consolidator | 9s       |
| **Total**    | **1.6 min** |
```

### Data sourcing

All sections except `## 4 · Final Answer` are generated **programmatically** by `report.py` from the `RunCollector` state accumulated during streaming — no extra LLM call. The `## 4 · Final Answer` section is the consolidator's raw output, which the consolidator writes directly in Quarkdown format (see below).

---

## Consolidator Prompt Changes

The consolidator prompt (`prompts/consolidator.md`) gains a new `## Output Format` section instructing it to write in Quarkdown:

- Inline equations: `$ E = mc^2 $`
- Display equations on their own line: `$ v = \sqrt{2gh} $`
- `.box {Answer} type:{tip}` wrapping the final numeric answer
- `.box {Caveats} type:{warning}` for any unresolved peer reviewer issues
- Numbered steps using standard Markdown ordered lists (Quarkdown renders them correctly)

The existing content rules (synthesize, physics-student level, English) are unchanged.

The CLI extracts the consolidator's output as-is for the `## 4 · Final Answer` section. The same raw text is also printed to terminal as the final answer block — no syntax stripping. The `.qd` file is the canonical formatted output; the terminal display is informational only.

---

## `StreamEvent` Types (`streaming.py`)

```python
@dataclass
class SubagentStarted:
    name: str
    t: float                    # monotonic timestamp

@dataclass
class ToolCallMade:
    subagent: str
    tool: str
    args_preview: str           # truncated ~80 chars

@dataclass
class ToolResultReceived:
    subagent: str
    tool: str
    result_preview: str         # truncated ~80 chars

@dataclass
class SchemaProduced:
    subagent: str
    schema_type: str            # "ProblemSpec", "ResearchBrief", etc.
    data: dict                  # full parsed dict

@dataclass
class SubagentFinished:
    name: str
    duration_s: float

@dataclass
class RunFinished:
    total_s: float
    final_answer: str

StreamEvent = (
    SubagentStarted | ToolCallMade | ToolResultReceived |
    SchemaProduced | SubagentFinished | RunFinished
)
```

### `RunCollector`

Accumulates events into structured state:

```python
@dataclass
class SubagentRun:
    name: str
    duration_s: float
    tool_calls: list[tuple[str, str, str]]  # (tool, args_preview, result_preview)
    schemas: list[tuple[str, dict]]          # (schema_type, data)

@dataclass
class RunCollector:
    problem: str
    model: str
    subagent_runs: list[SubagentRun]
    total_s: float
    final_answer: str
```

---

## `solve` Command Changes (`cli.py`)

```python
@app.command()
def solve(
    ...,
    verbose: bool = ...,
    output: Path | None = typer.Option(None, "--output", "-o", ...),
    trace: Path | None = ...,
) -> None:
```

If `--output` is omitted, the path defaults to `solvay-report-<timestamp>.qd` computed at runtime. `_stream_verbose` is replaced by a thin loop:

```python
collector = RunCollector(problem=problem, model=config.model_for("orchestrator"), ...)
for event in parse_stream(agent, problem):
    if verbose:
        print_event(event)
    collector.accumulate(event)

qd = generate_quarkdown(collector, config)
out_path = output or Path(f"solvay-report-{timestamp()}.qd")
out_path.write_text(qd, encoding="utf-8")
typer.echo(f"Report saved → {out_path}")
```

When `verbose=False`, `parse_stream()` is still used to drive the report collection, but `print_event` is skipped — only subagent start/finish lines are printed (current behavior). `generate_quarkdown` with a non-verbose collector produces a minimal report: problem statement + final answer sections only, no tool call tables or intermediate schema sections.

---

## Testing

### `tests/test_streaming.py`

- Unit tests for `RunCollector.accumulate()` with synthetic `StreamEvent` sequences
- Assert correct subagent grouping, timing, schema extraction
- No live agent call needed

### `tests/test_report.py`

- Unit tests for `generate_quarkdown()` with a pre-built `RunCollector` fixture
- Assert Quarkdown output contains expected section headers, `.box` blocks, LaTeX markers
- Assert metadata table contains correct values

---

## Open Questions

- **Schema detection heuristic**: Matching `tool_call.output` JSON to Pydantic schemas by field names may produce false positives. A conservative approach (require all required fields to be present) should be used. Known schema field sets: `ProblemSpec` (`statement, domain, knowns, unknowns, assumptions, approach_hints`), `ResearchBrief` (`principles, candidate_equations, analogies, citations`), `SolutionDraft` (`method, steps, final_answer, code_trace`), `Verdict` (`approved, issues, severity`).
- **`stream_events` v3 availability**: Verify that the installed version of deepagents supports `agent.stream_events(version="v3")` before implementing. Fallback: use `stream_mode="messages"` + `subgraphs=True` + `version="v2"` and parse `chunk["ns"]` tuples of the form `("tools:<call_id>",)`, correlating call IDs with subagent names from orchestrator tool calls.
