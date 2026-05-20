# Solvay Native DeepAgents Harness - Design Spec

**Date:** 2026-05-20
**Status:** Draft for review
**Supersedes:** The hand-rolled notebook/persistence parts of
`2026-04-15-solvay-design.md`. The benchmark design remains separate.

---

## 1. Context and motivation

Solvay was originally designed as a DeepAgents-based multi-agent physics lab,
but some harness responsibilities were implemented manually:

- `lab_notebook.md` conventions were injected by custom middleware instead of
  being treated as first-class DeepAgents filesystem state.
- Cross-run memory lived in `persistence.py`, outside the native DeepAgents
  memory/backend model.
- The top-level parser -> researcher -> solver flow depended heavily on a
  prompt protocol instead of making the DeepAgent supervisor and task tool the
  explicit center of the architecture.
- Operational failures discovered by live agent runs had no structured place to
  leave actionable notes for future developers or future coding agents.

The new direction is to make Solvay a native DeepAgents harness first, with
LangGraph used only where deterministic control is valuable.

Official docs that anchor this design:

- https://docs.langchain.com/llms.txt
- https://docs.langchain.com/oss/python/deepagents/overview
- https://docs.langchain.com/oss/python/deepagents/memory
- https://docs.langchain.com/oss/python/deepagents/long-term-memory
- https://docs.langchain.com/oss/python/deepagents/backends
- https://docs.langchain.com/oss/python/deepagents/subagents
- https://docs.langchain.com/oss/python/deepagents/context-engineering

For future work on DeepAgents, LangChain, or LangGraph APIs, agents must consult
the official docs first. Local memory or model recall is not enough.

---

## 2. Goals

1. Make DeepAgents the primary harness abstraction for Solvay: supervisor,
   subagents, filesystem, memory, permissions, context isolation, and backend
   routing.
2. Replace custom notebook/persistence wiring with filesystem-backed memory and
   backend paths that DeepAgents exposes through its native file tools.
3. Preserve the useful Solvay convention of agents leaving short structured
   notes, but store those notes in native filesystem locations.
4. Add an operational "harness notes" channel where runtime agents can record
   tool failures, bad paths, dependency problems, sandbox issues, API mismatch,
   and repair suggestions for future coding agents.
5. Keep deterministic LangGraph only for control loops that need guaranteed
   termination, especially the solver -> verifier/reviewer loop.
6. Prepare the system for first live testing with local Ollama models.

---

## 3. Non-goals

- Build a web UI.
- Replace the benchmark suite in this spec. Benchmark CLI unification remains a
  follow-up task.
- Fully migrate scientific code execution to a DeepAgents sandbox immediately.
  The existing `python_exec` tool can remain until the native sandbox path is
  proven equivalent for scientific workloads.
- Store secrets in memory files.
- Let runtime agents edit developer-facing specs or source code during solve
  runs.

---

## 4. Architecture overview

Solvay becomes a DeepAgent supervisor with native memory and backend routing.
The supervisor coordinates specialized subagents through the DeepAgents `task`
tool and filesystem state.

```
User problem
    |
    v
DeepAgent supervisor
    |-- native todos/planning
    |-- native filesystem tools
    |-- native memory files
    |-- task tool
    |
    +--> parser subagent        -> /workspace/problem_spec.json
    +--> researcher subagent    -> /workspace/research_brief.json
    +--> solver CompiledSubAgent
            |
            +-- LangGraph loop for deterministic control
            +-- solver agent with scientific tools
            +-- verifier agent with scientific checks
            +-- peer reviewer agent with semantic critique
            |
            +-> /workspace/solution_draft.json
            +-> /workspace/lab_notebook.md
            +-> /workspace/artifacts/
    |
    +--> final synthesis
    +--> optional memory update
```

The supervisor owns orchestration. It does not solve physics directly. It reads
the relevant memory files, creates or resets session workspace files, delegates
work, and decides which outcomes should be promoted to long-term memory.

---

## 5. Filesystem and memory layout

Use DeepAgents native filesystem paths as the agent-facing API. Backends decide
where those paths live.

### 5.1 Long-term memory

`/memories/solvay/AGENTS.md`

Purpose: durable scientific and methodological memory for the Solvay runtime.
Examples:

- Physics traps the system repeatedly encounters.
- Good verification habits.
- Model-specific behavior patterns that affect solve quality.
- Stable conventions for how Solvay agents should use tools.

Agents may propose updates to this file, but writes should be controlled. In
early local development, updates can require user approval or be written to a
review queue first.

### 5.2 Operational harness notes

`/memories/solvay/harness_notes.md`

Purpose: durable operational notes from runtime Solvay executions for future
developers and coding agents. This is not physics memory. It is feedback about
the harness itself.

Runtime agents append short reports when they encounter issues such as:

- A tool failed or returned an unexpected schema.
- A filesystem path was missing, ambiguous, or poorly documented.
- A local model behaved differently from hosted models.
- A dependency was missing or incompatible.
- A code execution sandbox blocked a necessary operation.
- Web/search/fetch tools were unavailable or misconfigured.
- A prompt or task contract caused repeated malformed outputs.

Required entry format:

```md
## [harness-note] 2026-05-20T12:34:56Z role=solver severity=warning
Symptom: python_exec rejected code importing <library>.
Context: solving mechanics problem; verifier requested numeric integration.
Likely cause: dependency absent from sandbox environment.
Suggested fix: add dependency to pyproject or document unsupported library.
Artifacts: /workspace/artifacts/run-2026-05-20-123456/
```

Severity values:

- `info`: useful observation, no failure.
- `warning`: degraded behavior, workaround available.
- `error`: task failed or answer quality is materially affected.

This file should be read by coding agents working on Solvay before modifying
the harness. It is a project-level feedback loop from runtime agents to
development agents.

### 5.3 Session workspace

`/workspace/lab_notebook.md`

Purpose: per-task working notes shared among the Solvay runtime agents.
It keeps the original Solvay note style, but now as native filesystem state.

Entry format:

```md
## [solver] iter 2
- Checked dimensional consistency for final expression.
- Previous critique found missing drag term; corrected equation of motion.
```

Rules:

- Entries are short: at most four lines per role per iteration.
- The notebook is for this task only.
- Do not promote task-local scratch notes to long-term memory automatically.
- The supervisor may summarize learnings from this file into a memory update
  proposal after the run.

`/workspace/problem_spec.json`

Parser output.

`/workspace/research_brief.json`

Researcher output.

`/workspace/solution_draft.json`

Current solver output.

`/workspace/critique_history.json`

Verifier and peer-reviewer verdicts across iterations.

`/workspace/artifacts/`

Plots, code snippets, traces, numeric checks, and other run-local artifacts.

---

## 6. Backend routing

Initial local implementation should use a `CompositeBackend`:

- default: `StateBackend()` for ephemeral workspace files.
- `/memories/`: persistent backend, initially local filesystem or DeepAgents
  `StoreBackend`.
- `/skills/`: optional skill backend, later if Solvay grows domain skills.

Target shape:

```python
create_deep_agent(
    model=config.model_for("orchestrator"),
    memory=[
        "/memories/solvay/AGENTS.md",
        "/memories/solvay/harness_notes.md",
    ],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=...),
        },
    ),
    subagents=[...],
    permissions=[...],
)
```

The exact namespace strategy is implementation-specific. For local development,
it can be agent-scoped or project-scoped. For multi-user use, it must include a
user or assistant identity.

---

## 7. Permissions

Solvay should use DeepAgents filesystem permissions to separate mutable scratch
state from protected memory and code.

Recommended policy:

- `/workspace/**`: runtime agents may read and write.
- `/memories/solvay/AGENTS.md`: runtime agents may read; writes require an
  explicit memory-update flow.
- `/memories/solvay/harness_notes.md`: runtime agents may append structured
  notes. Full rewrites require approval.
- `/policies/**` or `/docs/**`: read-only for runtime agents.
- Source code paths: not mounted into the runtime harness unless explicitly
  needed for developer workflows.

If DeepAgents permissions cannot express append-only behavior directly, implement
append-only through a wrapper tool or write notes to
`/workspace/harness_note_proposals.jsonl` and let the supervisor promote them.

---

## 8. Subagents and responsibilities

### 8.1 Supervisor

The supervisor is the main DeepAgent. Responsibilities:

1. Read long-term memory.
2. Initialize session workspace files.
3. Create and update todos.
4. Delegate to parser, researcher, and solver.
5. Keep orchestration context small by trusting subagent summaries.
6. Produce the final answer.
7. Decide whether to propose memory updates.
8. Ensure harness notes are recorded when operational issues occur.

The supervisor does not solve the physics itself.

### 8.2 Parser

Inputs: raw problem statement.

Outputs:

- `ProblemSpec` as structured response.
- `/workspace/problem_spec.json`.
- Optional brief notebook entry.

### 8.3 Researcher

Inputs: `ProblemSpec`.

Outputs:

- `ResearchBrief` as structured response.
- `/workspace/research_brief.json`.
- Citations and relevant equations.
- Harness note if web/search/fetch behavior is degraded.

### 8.4 Solver

The solver remains a `CompiledSubAgent` because its loop needs deterministic
termination. Inside the compiled graph, model work should be done by native
LangChain/DeepAgents-style agents with tools and structured output.

The solver writes:

- `/workspace/solution_draft.json`
- `/workspace/critique_history.json`
- `/workspace/lab_notebook.md`
- `/workspace/artifacts/**`

The solver may set `solver_blocked=true` with a `blocked_topic` when the
research brief is insufficient.

### 8.5 Verifier and peer reviewer

These remain separate critics. They should have narrow tool access:

- Verifier: `python_exec`, dimensional checks, numeric consistency tools.
- Peer reviewer: semantic critique, limited search when needed.

They write verdicts to the critique history and may add short notebook entries.

---

## 9. Harness notes as a reusable pattern

The `harness_notes.md` concept should be designed as a pattern that can later be
standardized across AI harness projects, including coding harnesses.

General pattern:

- Runtime agents see real failures that developers do not see.
- They need a low-friction way to report those failures.
- Reports must be structured enough for future coding agents to act on.
- Reports must be separate from domain memory so operational noise does not
  pollute reasoning memory.

Minimal portable contract:

```md
## [harness-note] <iso-timestamp> role=<role> severity=<info|warning|error>
Symptom: <what failed>
Context: <what the agent was trying to do>
Likely cause: <best current hypothesis>
Suggested fix: <concrete repair or investigation step>
Artifacts: <paths or trace ids, if any>
```

For Solvay, this file is part of DeepAgents memory. For other harnesses, the
same contract could live under `.agent/memory/harness_notes.md`,
`/memories/<project>/harness_notes.md`, or another backend-routed path.

---

## 10. Error handling

When a runtime agent encounters an operational failure:

1. Try the local fallback if one is explicitly documented.
2. If the fallback works, add a `warning` harness note.
3. If the fallback does not work and answer quality is affected, add an `error`
   harness note.
4. Continue only if the final answer can honestly disclose remaining uncertainty.
5. Never store secrets, raw API keys, or private credentials in harness notes.

Examples:

- Ollama model fails to produce valid structured output after retries:
  write a harness note with model name, role, schema, and failure mode.
- `python_exec` lacks a package:
  write package name, attempted import, problem type, and suggested dependency.
- Researcher cannot access web search:
  write tool name, unavailable env var or error class, and fallback used.
- File path confusion:
  write expected path, actual path, and which prompt or tool produced it.

---

## 11. Testing strategy

Unit tests:

- Memory path configuration includes both Solvay memory files.
- Backend routing sends `/memories/**` to persistent backend and workspace files
  to ephemeral state.
- Permissions deny or gate writes to protected paths.
- Harness note formatter produces valid markdown with required fields.
- Harness note writer never includes known secret patterns.
- Solver loop still returns a `messages` final payload for `CompiledSubAgent`.

Integration tests:

- A fake run writes `/workspace/lab_notebook.md` and a harness note proposal.
- A run with a mocked failing tool records a warning note and continues.
- A run with malformed structured output records an error note and terminates
  with an honest failure state.
- Local Ollama smoke test: one simple mechanics problem with persistence disabled
  except for harness notes.

Manual checks:

- Read `harness_notes.md` before making harness changes.
- Run `ruff`, `mypy`, and `pytest`.
- Review memory diffs before promoting any runtime-discovered learning.

---

## 12. Migration plan

This spec should be implemented in phases.

### Phase 1 - Native memory and backend skeleton

- Add a harness configuration module for DeepAgents memory paths and backend
  construction.
- Wire `create_solvay_agent()` to pass `memory=` and `backend=`.
- Keep old persistence code temporarily, but stop using it in the main path.
- Add tests for paths, backend routing, and permissions.

### Phase 2 - Session workspace conventions

- Move `lab_notebook.md` into native workspace filesystem state.
- Make parser/researcher/solver prompts refer to `/workspace/lab_notebook.md`.
- Remove custom notebook injection once native memory and filesystem access are
  sufficient.

### Phase 3 - Harness notes

- Add a small formatter/writer for harness notes or proposals.
- Add prompts telling each runtime role when to leave a harness note.
- Route durable harness notes to `/memories/solvay/harness_notes.md`.
- Add tests for failed tools and bad paths.

### Phase 4 - Supervisor cleanup

- Tighten the supervisor prompt around native DeepAgents behavior:
  filesystem-first, task delegation, memory update proposals, and no direct
  physics solving.
- Keep parser/researcher/solver as specialized subagents.
- Ensure the final answer includes loop summary and honest uncertainty.

### Phase 5 - Harness permissions and sandbox review

- Add read/write permissions for workspace, memory, policies, and source code.
- Evaluate whether DeepAgents sandbox execution can replace or wrap
  `python_exec`.
- Keep `python_exec` if it remains more reliable for scientific computations.

### Phase 6 - Ollama live test readiness

- Add a local-model profile or documented command using `SOLVAY_MODEL=ollama:*`.
- Run a single mechanics problem.
- Inspect `/workspace/lab_notebook.md`, `/memories/solvay/harness_notes.md`, and
  final answer quality.
- Use the resulting notes to drive benchmark and memory follow-ups.

---

## 13. Open questions

1. Should harness notes be appended directly to long-term memory, or should they
   first land in a review queue?
2. Which backend should store `/memories/` locally: `StoreBackend`,
   `FilesystemBackend`, or a composite with project-scoped namespace?
3. Can DeepAgents permissions enforce append-only memory writes, or do we need a
   wrapper tool?
4. Should harness notes be consumed by `CLAUDE.md`/`AGENTS.md` workflows for
   coding agents, or only by Solvay-specific docs?
5. Should runtime agents be allowed to write harness notes during benchmark runs,
   or should benchmarks keep all memory writes disabled for reproducibility?

---

## 14. Success criteria

The migration is successful when:

- `create_solvay_agent()` uses DeepAgents native `memory=` and `backend=`.
- Runtime agents use `/workspace/lab_notebook.md` through native filesystem
  tools.
- Operational failures produce structured harness notes.
- Future coding agents can read `harness_notes.md` before changing Solvay.
- The solver still has deterministic loop termination.
- `ruff`, `mypy`, and `pytest` pass.
- A local Ollama smoke run can produce either a valid answer or an actionable
  harness note explaining why it failed.

