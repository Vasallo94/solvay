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
agent.stream(stream_mode="updates", subgraphs=True, version="v2")
        │
        ▼
parse_stream(agent, problem) → Iterator[StreamEvent]
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

### Stream mode change

`_stream_verbose` currently uses `stream_mode="debug"`, which delivers `task_result` events post-hoc (after each subagent completes). The reimplementation switches to `stream_mode="updates"` with `subgraphs=True` and `version="v2"`, the same mode used by the `chat` command, which delivers per-node updates including from subgraph namespaces as they happen.

The `ns` (namespace) field on each chunk identifies whether the event originates from the orchestrator or from inside a running subagent. Orchestrator-level events (empty `ns`) are used to detect subagent dispatches (tool calls named `"task"`). Subagent-internal events (non-empty `ns`) expose tool calls and results from within that subagent.

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

The `.qd` report is generated after every `solve` run (not behind a feature flag). Default output path: `report.qd` in the current working directory. Configurable via `--output <path>` on the `solve` command. A `--output` value without a `.qd` extension is accepted as-is (allows `.md` or other formats in future).

### `--output` flag

```
uv run solvay solve -v "problem" --output my-report.qd
uv run solvay solve "problem"            # always writes report.qd
```

The Makefile `RUN` variable does not need updating; `report.qd` is acceptable as a default.

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
    output: Path = typer.Option(Path("report.qd"), "--output", "-o", ...),
    trace: Path | None = ...,
) -> None:
```

`_stream_verbose` is replaced by a thin loop:

```python
collector = RunCollector(problem=problem, model=config.model_for("orchestrator"), ...)
for event in parse_stream(agent, problem):
    print_event(event)
    collector.accumulate(event)

qd = generate_quarkdown(collector, config)
output.write_text(qd, encoding="utf-8")
typer.echo(f"Report saved → {output}")
```

When `verbose=False`, `agent.invoke()` is still used (no streaming), and `generate_quarkdown` receives a minimal collector with only the final answer — producing a report with just the problem statement and final answer sections.

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

- **Namespace format**: The exact structure of `ns` in deepagents' LangGraph event stream is not documented. Implementation will need a short exploratory spike to verify how subagent namespaces are represented before committing to the parsing logic.
- **Schema detection heuristic**: Matching JSON content to Pydantic schemas by field names may produce false positives. A conservative approach (require all required fields to be present) should be used.
- **`verbose=False` report**: The report generated without `-v` will be sparse (no tool calls, no intermediate schemas). This is acceptable for now; a future improvement could run a lightweight structured extraction pass.
