# Solvay — Design Spec

**Date:** 2026-04-15
**Status:** Draft for review
**Project name:** Solvay (Python package: `solvay`)

---

## 1. Context and goals

Solvay is a multi-agent system that solves physics problems by emulating a human research team: a parser, a researcher, a solver, a verifier, and a peer reviewer collaborate through a reflective review loop until they reach consensus (or exhaust a budget).

The **primary goal is experimental**: explore how far [`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) gets with a hard domain (physics) when given real scientific tooling (`sympy`, `scipy`, `astropy`, etc.) and a research-team-style orchestration. Correctness matters, but what we want to observe is **planning, subagent delegation, tool use, and reflective iteration** in action.

**Not** a tutor, not a product, not a benchmark-topping system. A lab.

### Non-goals

- Beating state-of-the-art physics problem benchmarks.
- Production-grade reliability, multi-tenant deployment, or streaming UIs.
- Image/diagram parsing from problem statements (text-only input).
- Training or fine-tuning any model.

---

## 2. Scope

**In scope (v1):**

- Mixed-domain physics problems: mechanics, electromagnetism, quantum, thermodynamics, astrophysics. The agent chooses domain and method.
- University-level problems (Griffiths, Tipler, Marion level) as the calibration target, with a small curated benchmark (~15–20 problems).
- Natural language input, structured answer (method + steps + final quantity with units).
- CLI interface.
- LangSmith tracing for observability.
- Persistent cross-run lab journal.

**Deferred (see `FUTURE.md`):**

- AlphaXiv / arXiv MCP integration.
- Context7 MCP as primary library-docs source (current: Tavily `web_search` covers it).
- Per-role model mix (infrastructure will support it; defaults are single-model).
- Local models via Ollama.
- Consolidation of `lab_journal.md` into distilled `lab_rules.md`.
- QuTiP and domain-specific simulation libraries.
- Web/notebook UI.

---

## 3. Architecture overview

Built on **deepagents** with one key extension: the `solver` role is not a plain subagent but a `CompiledSubAgent` wrapping a LangGraph subgraph that implements the review loop.

```
User (natural language problem)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Main orchestrator (create_deep_agent)                   │
│   built-in tools: write_todos, ls/read_file/write_file/ │
│                   edit_file, task, web_search           │
│                                                         │
│   subagents:                                            │
│     • parser       (dict subagent)                      │
│     • researcher   (dict subagent)                      │
│     • solver       (CompiledSubAgent → loop subgraph)   │
│     • verifier     (dict subagent)                      │
│     • peer_reviewer (dict subagent)                     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Final answer (SolutionDraft + critique history + trace)
```

**Orchestrator flow** (driven by its system prompt, executed via the `task` tool):

```
parser(statement) → ProblemSpec
  └─ researcher(ProblemSpec) → ResearchBrief
       └─ solver(ProblemSpec, ResearchBrief) → SolutionDraft
              [internally iterates solver ↔ verifier ∥ peer_reviewer]
```

The solver subgraph may terminate with `judge_forced` when the inner solver signals it needs deeper physics research. In that case the orchestrator re-invokes `researcher` with the blocked topic and calls the solver once more (maximum 1 re-research per run).

---

## 4. Roles and contracts

All inter-agent deliverables are Pydantic models defined in `src/solvay/schemas.py`. This file is the single source of truth for contracts between roles.

### 4.1 Shared types

```python
class Quantity(BaseModel):
    value: float | str           # symbolic (e.g., "g") or numeric
    unit: str | None             # e.g., "m/s^2"; None if dimensionless

class ProblemSpec(BaseModel):
    statement: str
    domain: Literal["mechanics", "em", "quantum", "thermo", "astro", "other"]
    knowns: dict[str, Quantity]
    unknowns: list[str]
    assumptions: list[str]
    approach_hints: list[str]     # non-binding suggestions from parser

class ResearchBrief(BaseModel):
    principles: list[str]
    candidate_equations: list[str]   # LaTeX
    analogies: list[str]
    citations: list[str]

class SolutionDraft(BaseModel):
    method: str
    steps: list[str]
    final_answer: Quantity | str
    code_trace: list[str]

class Verdict(BaseModel):
    approved: bool
    issues: list[str]
    severity: Literal["blocker", "minor", "none"]
```

### 4.2 Subagents

| Role | Type | Tools | Response format | Purpose |
|---|---|---|---|---|
| `parser` | dict subagent | — (LLM only) | `ProblemSpec` | Parse NL statement; detect domain; extract knowns/unknowns/units. |
| `researcher` | dict subagent | `web_search`, `url_fetch` | `ResearchBrief` | Gather applicable principles, candidate equations, analogies, references. |
| `solver` | `CompiledSubAgent` | `python_exec`, `check_dimensions`, `web_search`, `url_fetch` | `SolutionDraft` | Solve the problem. Internally runs the review loop (§6). |
| `verifier` | dict subagent | `python_exec`, `check_dimensions` | `Verdict` | Mechanical critique: dimensions, limit cases, order of magnitude, numerical consistency. |
| `peer_reviewer` | dict subagent | `python_exec`, `web_search` | `Verdict` | Semantic critique: is the approach correct? missing terms? assumptions justified? better method available? |

**Model configuration:** default `anthropic:claude-sonnet-4-6` for all roles. The agent factory accepts a `models: dict[Role, str]` override for per-role model mix (future use, see `FUTURE.md`).

**Web-search scope by role:**

- `researcher`: physics principles, equations, analogies. Primary consumer.
- `solver`: **library/API usage only** (e.g., "sympy.physics.mechanics Lagrangian example"). Not for physics theory — if solver needs deeper theory, it writes a `BLOCKED` note and stops its iteration (see §6).
- `peer_reviewer`: rarely; only to cross-check known reference values.
- `parser`, `verifier`: no web search.

This scope is enforced **in the subagent prompts**, not structurally.

---

## 5. Tools

### 5.1 Built-in (provided by deepagents)

- **`write_todos`** — planning tool for the orchestrator.
- **`ls`, `read_file`, `write_file`, `edit_file`** — virtual filesystem shared across orchestrator and subagents. Backing store for the lab notebook and artifacts (plots, intermediate files).
- **`task`** — delegation to subagents. Used only by the orchestrator.
- **`web_search`** — Tavily-backed. Signature: `web_search(query, max_results=5, topic="general", include_raw_content=False)`.

### 5.2 Custom

**`python_exec(code: str) -> ExecResult`**

Sandboxed Python runtime with the scientific stack preinstalled. State persists across calls within a run (kernel-like semantics).

- **Isolation:** runs the code in a separate, restricted Python interpreter with no network access and no host filesystem access. Default 30-second wall-time cap; default 512MB memory cap. Both configurable.
- **Preinstalled libraries:** `sympy` (including submodules `sympy.physics.mechanics`, `sympy.physics.quantum`, `sympy.physics.vector`, `sympy.physics.units`, `sympy.physics.optics`), `scipy`, `numpy`, `matplotlib` (non-interactive backend), `astropy`, `uncertainties`, `pint`, `mpmath`.
- **Returns:** `ExecResult(stdout, stderr, last_expr_repr, last_expr_latex, wall_time_ms, artifacts)`.
- **Artifacts:** PNGs/plots generated via matplotlib are saved to the virtual FS under `plots/` and their paths returned in `artifacts`. Agents can reference them from the lab notebook. Claude Sonnet 4.6 is multimodal, so other agents can `read_file` on them.

**`check_dimensions(expression: str, expected_unit: str) -> DimCheckResult`**

Symbolic dimensional verification via `sympy.physics.units.convert_to`. Works on expressions with free symbols (does not require numerical evaluation).

- **Returns:** `DimCheckResult(ok: bool, actual_unit: str, simplified: str, notes: str | None)`.
- Kept as a separate tool (not folded into `python_exec`) because:
  1. It is the most common LLM failure mode in physics, worth surfacing explicitly.
  2. The verifier prompt mandates calling it before approving any answer, which makes it auditable in the LangSmith trace.
  3. It is semantically pure — decoupled from the `python_exec` kernel state.

**`url_fetch(url: str, extract_text: bool = True, max_chars: int = 20000) -> str`**

Generic HTTP fetcher. Uses `httpx` + `trafilatura` for clean text extraction. Replaces a Wikipedia-specific tool — researcher composes `web_search(query)` → `url_fetch(result.url)` for any source.

---

## 6. Solver loop subgraph

Heart of the system. Lives in `src/solvay/subagents/solver_loop.py`. Built as a LangGraph graph compiled via `CompiledSubAgent`.

### 6.1 State

```python
class SolverLoopState(BaseModel):
    # Inputs (set once at entry)
    problem_spec: ProblemSpec
    research_brief: ResearchBrief

    # Loop state
    iteration: int = 0
    max_iterations: int = 3
    current_draft: SolutionDraft | None = None
    verifier_verdict: Verdict | None = None
    reviewer_verdict: Verdict | None = None
    critique_history: list[dict] = []  # per-iter verifier + reviewer + judge

    # Blocking signal (solver asking for deeper research)
    solver_blocked: bool = False
    blocked_topic: str | None = None

    # Output
    final_draft: SolutionDraft | None = None
    termination_reason: Literal[
        "consensus", "budget_exhausted", "judge_forced"
    ] | None = None
    unresolved_blockers: bool = False
```

### 6.2 Nodes

1. **`solve`** — invokes solver LLM with `problem_spec`, `research_brief`, the lab notebook (injected by middleware), and (if `iteration > 0`) explicit critique from the previous iteration. Produces `SolutionDraft` or sets `solver_blocked=True` with `blocked_topic`. Increments `iteration`. Appends to lab notebook.

2. **`critique`** — fan-out node: invokes `verifier` and `peer_reviewer` **in parallel**. Both receive the current draft and the lab notebook. Both return `Verdict`. Rationale for parallel: critique independence. The judge consolidates.

3. **`judge`** — **deterministic Python, no LLM**. Rules:
   - `verifier.approved AND reviewer.approved` → `termination_reason="consensus"`, exit to END.
   - `solver_blocked` → `termination_reason="judge_forced"`, exit to END (orchestrator handles re-research).
   - `iteration >= max_iterations` → `termination_reason="budget_exhausted"`. Sets `unresolved_blockers = any severity == "blocker"`. Exit to END with current draft as best effort.
   - Otherwise → back to `solve`.

Why deterministic judge: guaranteed termination, clean traces, trivial to unit-test, saves tokens, and the physics judgment already happened in verifier/reviewer. The judge is flow control, not reasoning.

### 6.3 Edges

```
START → solve → critique → judge ─→ END
                              └─→ solve
```

### 6.4 Output contract

On exit, `task(name="solver", ...)` returns (through the CompiledSubAgent wrapper):

```python
{
    "final_draft": SolutionDraft,
    "termination_reason": "consensus" | "budget_exhausted" | "judge_forced",
    "iterations_consumed": int,
    "unresolved_blockers": bool,
    "blocked_topic": str | None,   # set only when termination_reason == "judge_forced"
    "critique_history": list[dict],
}
```

### 6.5 Orchestrator handling of `judge_forced`

When the solver returns `termination_reason="judge_forced"`, the orchestrator's prompt instructs it to:

1. Call `researcher` again with focus on `blocked_topic`.
2. Append the new research output to the notebook.
3. Call `solver` one more time.

Maximum **one** re-research per run (hard-coded in the orchestrator prompt, not state-tracked — this is an experiment, not a production safety rail). If the second attempt also returns `judge_forced`, the orchestrator returns the best-effort draft with the unresolved-topic note.

---

## 7. Lab notebook and persistence

The lab notebook is the **informal lateral channel** between agents — complementary to the Pydantic deliverables. It lives in the deepagents virtual filesystem as `lab_notebook.md`.

### 7.1 Format

Append-only markdown log. Convention:

```markdown
## [<role>] iter <N>
<free-form content, max ~4 lines, in English>
```

### 7.2 Usage convention (enforced in every subagent prompt)

- **Before acting:** read `lab_notebook.md`; summarize what is relevant to your current task.
- **After acting:** if you found anything useful for future agents (methodological tips, traps, questionable assumptions, blocked topics), append an entry.
- **Brevity required:** ≤4 lines per entry.

### 7.3 Middleware-assisted injection

A lightweight middleware injects **the last N notebook entries** into each subagent's context when invoked, so the model does not depend on remembering to read the file. N is configurable (default 20).

### 7.4 Cross-run persistence (always on, opt-out via `--no-persist`)

Directory layout under `~/.solvay/`:

```
~/.solvay/
├── lab_journal.md                                       # append-only, across all runs
└── sessions/
    └── 2026-04-15_143022_<problem-slug>.md              # full session notebook
```

- **Session start:** a snapshot of the most recent relevant entries from `lab_journal.md` is loaded into the virtual FS notebook as initial context.
- **Session end:** a lightweight **consolidator subagent** (dict subagent with `response_format = list[JournalEntry]`) reviews the session notebook and extracts learning-worthy entries — mistakes made and how they were fixed, methodological tricks, identified pitfalls. Those entries are appended to `lab_journal.md`.
- **Full session notebook** is saved verbatim under `sessions/` for auditability.

**Future (see `FUTURE.md`):** `solvay consolidate` command → dedup + cluster + summarize `lab_journal.md` into `lab_rules.md`. Once the journal grows, the rules file replaces the journal snapshot as the initial context payload.

---

## 8. CLI

Implemented with `typer`.

```bash
# Solve a problem interactively
solvay solve "A block of mass 2 kg slides down a 30° incline..."

# From file
solvay solve --file problem.txt

# Show the full lab notebook after solving
solvay solve "..." --show-notebook

# Dump full trace as JSON
solvay solve "..." --trace trace.json

# Opt out of cross-run persistence (useful in CI/tests)
solvay solve "..." --no-persist

# Run the benchmark
solvay bench --problems benchmark/problems/ --out results.json
```

Primary `solve` output (stdout):

1. Final answer (method + steps + final quantity with units).
2. Loop summary: termination reason, iterations consumed, unresolved objections (if any).
3. Link/ID to the LangSmith trace for the run.

---

## 9. Observability

**LangSmith**, zero custom code. Activated via environment variables:

```
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=solvay
LANGSMITH_TRACING=true
```

Every `create_deep_agent` call, every subagent, every tool invocation, every LangGraph node is captured automatically with inputs, outputs, token counts, latencies, and errors. Runs are named for readability: `solvay-main / parser / researcher / solver-loop / solve-iter1 / verifier / peer_reviewer / judge / ...`.

---

## 10. Benchmark

**Purpose:** measure and track correctness + cost across a curated problem set. Not CI-blocking — experimental metric, run on demand.

**Problem format** (`benchmark/problems/*.json`):

```json
{
  "id": "griffiths-2.1",
  "source": "Griffiths - Intro to QM, 2nd ed.",
  "domain": "quantum",
  "statement": "A particle in the infinite square well...",
  "expected": {
    "value": "pi*hbar^2*n^2/(2*m*L^2)",
    "unit": "J",
    "tolerance_rel": 0.01
  },
  "tags": ["eigenvalue", "1d", "bound-state"]
}
```

**Runner** (`benchmark/run_bench.py`):

- Runs each problem through the full agent.
- Correctness evaluation:
  1. Try symbolic equivalence via `sympy.simplify(result - expected) == 0`.
  2. Fall back to numerical comparison within `tolerance_rel`.
- Report: overall accuracy, by domain, by tag; average iterations consumed; `budget_exhausted` rate; token/USD cost.

**Initial dataset:** 15–20 hand-curated problems spanning mechanics, EM, quantum, thermo, astro. Sourced from Griffiths, Tipler, Marion. Expandable.

---

## 11. Testing

CI-blocking (`pytest`):

- **`tests/test_schemas.py`** — Pydantic validation, edge cases.
- **`tests/test_tools.py`** — `check_dimensions`, `python_exec` (isolation, timeouts, memory limits, state persistence across calls), `url_fetch`.
- **`tests/test_solver_loop.py`** — the loop subgraph with `FakeListChatModel`. Scenarios:
  - Consensus on iter 1 → terminates OK.
  - 3 iterations without consensus → `budget_exhausted`.
  - Solver blocks → `judge_forced` with reported topic.
  - Minor critique does not block; blocker does.
  - Critique history accumulates correctly.
- **`tests/test_consolidator.py`** — session-end consolidator extracts only learning-worthy entries; skips operational noise.

Non-CI (opt-in via `RUN_LIVE_TESTS=1`):

- **`tests/test_agent_smoke.py`** — one simple mechanics problem end-to-end with a real LLM. Sanity check that the full stack assembles.

---

## 12. Project structure

```
solvay/                                    (repo root; may be renamed from ComplexProblemSolver)
├── pyproject.toml                         # uv-managed, requires-python = ">=3.14"
├── uv.lock
├── .env.example                           # ANTHROPIC_API_KEY, LANGSMITH_*, TAVILY_API_KEY
├── .pre-commit-config.yaml                # ruff check, ruff format, mypy, detect-secrets
├── .editorconfig
├── .github/workflows/ci.yml               # ruff + mypy + pytest (no live tests)
├── README.md
├── CLAUDE.md                              # AI-dev conventions for this repo
├── AGENTS.md                              # compat for Cursor / other IDEs
├── FUTURE.md                              # deferred ideas (Context7, alphaxiv, model mix, Ollama, etc.)
├── .claude/
│   └── settings.local.json
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-15-solvay-design.md
├── src/
│   └── solvay/
│       ├── __init__.py
│       ├── cli.py                         # typer entry point
│       ├── config.py                      # role-model mapping, loop budgets
│       ├── agent.py                       # create_deep_agent wiring
│       ├── schemas.py                     # all Pydantic contracts
│       ├── middleware/
│       │   └── notebook_injector.py       # injects last N notebook entries
│       ├── subagents/
│       │   ├── parser.py
│       │   ├── researcher.py
│       │   ├── verifier.py
│       │   ├── peer_reviewer.py
│       │   ├── solver_loop.py             # CompiledSubAgent with LangGraph subgraph
│       │   └── consolidator.py            # session-end journal extraction
│       ├── tools/
│       │   ├── python_exec.py
│       │   ├── dimensional.py
│       │   └── url_fetch.py
│       ├── persistence.py                 # ~/.solvay/ journal + sessions I/O
│       └── prompts/                       # .md system prompts per role
│           ├── orchestrator.md
│           ├── parser.md
│           ├── researcher.md
│           ├── solver.md
│           ├── verifier.md
│           ├── peer_reviewer.md
│           └── consolidator.md
├── benchmark/
│   ├── problems/
│   └── run_bench.py
└── tests/
    ├── test_schemas.py
    ├── test_tools.py
    ├── test_solver_loop.py
    ├── test_consolidator.py
    └── test_agent_smoke.py
```

### 12.1 Dev standards

**Language rule (project-wide, non-negotiable):** all code, identifiers, comments, docstrings, prompts, schemas, commit messages, branch names, documentation, and log strings are written in **English**. The user converses with Claude Code in Spanish, but no Spanish leaks into the repository.

**Python:** 3.14+, managed with `uv`.

**Style / types:** `ruff` (lint + format) and `mypy` enforced via pre-commit and CI.

**Commits:** small, focused; imperative mood; tied to spec sections or plan steps when possible.

**`CLAUDE.md` contents:**

- Project language rule (English everywhere in code).
- How to add a subagent: define schema → write prompt → register in `agent.py` → add test.
- How to add a tool: write function with typed signature and docstring → register → add test.
- How to run / test / bench: `uv` commands.
- Lab-notebook prompt conventions.
- **Working with deepagents:** canonical docs at https://docs.langchain.com/oss/python/deepagents/overview. For API lookups prefer Context7 (`/websites/reference_langchain`) over web search. Never rely on memory for deepagents API; always verify against current docs.

---

## 13. Key design decisions (recap)

1. **deepagents with `CompiledSubAgent` for the loop.** Simple dict subagents for parser/researcher/verifier/peer_reviewer; LangGraph subgraph for the solver↔critique loop. Keeps orchestration idiomatic while giving the loop deterministic termination.
2. **Deterministic judge.** The loop's terminator is Python logic, not an LLM call. Guaranteed termination, trivially testable, zero token cost.
3. **Parallel critique.** Verifier and peer_reviewer run in parallel each iteration. Independence is the point of having two critics.
4. **Lab notebook as shared lateral channel.** Pydantic contracts for deliverables; free-form markdown notebook for tips, flags, blocks. Persisted across runs by default.
5. **Symbolic dimensional analysis** via `sympy.physics.units` (not `pint`). Matches the symbolic nature of physics derivations.
6. **Generic `url_fetch`** rather than source-specific fetchers. Researcher composes `web_search` → `url_fetch`.
7. **Web search scoped by role** in prompts, not structurally — solver uses it only for library API, researcher for physics.
8. **English-only code, Spanish-only chat.**

---

## 14. Risks and open questions

- **Loop-by-prompt fragility still applies to the orchestrator** (which handles parser → researcher → solver and the `judge_forced` re-research case). If the orchestrator forgets a step, the system degrades. Mitigation: strict system prompt with step numbers, and plans to add a thin LangGraph wrapper at the orchestrator level as a fallback (tracked in `FUTURE.md`).
- **Notebook bloat.** Middleware injects only the last N entries, but a long session still grows the virtual FS file. Acceptable for v1; revisit if it harms latency.
- **Consolidator quality.** The end-of-session consolidator is itself an LLM; noisy or misleading entries could pollute `lab_journal.md`. Mitigation: require `response_format`, keep prompt conservative, allow manual editing of `lab_journal.md`.
- **Benchmark sample size (15–20)** is too small for statistical significance. It is a smoke-quality signal, not a conclusion. Expansion is a follow-up task.
- **Python 3.14 wheel availability.** Should be fine in April 2026 for all listed deps, but the first `uv sync` is the real test. Drop to 3.13 if any blocker.

---

## 15. Deferred to `FUTURE.md`

- Context7 MCP for library-docs (replace Tavily for API lookups).
- AlphaXiv / arXiv MCP for research paper access.
- Per-role model mix (e.g., Haiku for parser/verifier, Sonnet for solver/reviewer).
- Local models (Ollama) as drop-in backends.
- QuTiP and heavier domain libraries.
- `solvay consolidate` command → distilled `lab_rules.md`.
- Web/notebook UI.
- Orchestrator-level LangGraph wrapper for deterministic flow at the top level.
- Benchmark expansion + external datasets (JEEBench, PhysicsQA).
