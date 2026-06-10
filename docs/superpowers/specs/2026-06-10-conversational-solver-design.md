# Conversational Solver — Design Spec

**Date:** 2026-06-10
**Status:** Approved
**Supersedes:** the solver-loop subgraph described in section 3 of
`2026-04-15-solvay-design.md`.

---

## 1. Context and goals

The current solver is a `CompiledSubAgent` wrapping a rigid LangGraph subgraph
(solve → critique → judge). Code reading surfaced concrete defects:

1. **The solver loses its own work between iterations.** On re-solve it
   receives the problem, the research brief, and the *last* critique — but not
   the draft being criticized. It re-solves from scratch every round.
2. **Impossible instructions.** The solver/verifier/reviewer prompts order the
   agents to read/write `/workspace/lab_notebook.md`, but the inner
   `create_agent` instances have no filesystem tools.
3. **Fragile parsing.** A structured-output failure degrades to `{}` and
   injects a synthetic `"Failed to parse"` blocker verdict, burning one of the
   three iterations.
4. **Sequential critics.** Verifier and peer reviewer run serially.
5. **Dead wiring.** `SolverLoopConfig.max_iterations` is never passed to the
   graph; `verifier`/`peer_reviewer` are registered as orchestrator subagents
   but absent from the orchestrator workflow.
6. **E2E wiring risk.** `SolverLoopGraphState` has required pydantic fields
   (`problem_spec`, `research_brief`) the deepagents `task` tool cannot
   populate (it only passes `messages`).

**Goal:** replace the subgraph with a conversational solver — a Claude
Code-style agent loop where the solver keeps full message history and the
review cycle happens through tool calls — while staying on deepagents as the
harness. Priorities chosen by the user: loop quality and operational
robustness.

### Non-goals

- Replacing deepagents with a custom main loop (possible future work).
- Touching parser, researcher, or consolidator behavior.
- Benchmark expansion or baseline measurement (separate effort).

---

## 2. Architecture

The solver becomes a plain **dict subagent** of the orchestrator. The review
loop lives in the solver's own conversation, driven by a new `request_review`
tool.

```
Orchestrator (create_deep_agent)
  ├── parser        (dict subagent, unchanged)
  ├── researcher    (dict subagent, unchanged)
  ├── solver        (dict subagent, conversational)
  │     tools: python_exec, check_dimensions, web_search,
  │            url_fetch, request_review
  └── consolidator  (dict subagent, unchanged)
```

- `verifier` and `peer_reviewer` are **removed from the orchestrator's
  subagent list**. They exist only inside `request_review`.
- The solver, as a harness dict subagent, has real filesystem tools, so the
  lab-notebook convention stays for the solver (and remains for parser /
  researcher / orchestrator).

### 2.1 The `request_review` tool

Created by a factory that closes over `SolvayConfig` (budget) and the two
critic runnables.

Behavior per call:

1. Takes the current draft (method, steps, final answer, code trace) as input.
2. Invokes verifier and peer reviewer **in parallel**, each a
   `langchain.agents.create_agent` with `response_format=Verdict`
   (same construction as today: verifier gets `python_exec` +
   `check_dimensions`; reviewer gets `python_exec` + `web_search`).
3. Returns a JSON payload: both verdicts, `reviews_remaining`, and a
   `directive` field:
   - both approved → `"consensus — finalize now"`,
   - budget exhausted → `"budget exhausted — finalize with best effort"`,
   - otherwise → `"address the issues and request review again"`.
4. The call counter lives in the tool closure. Once
   `reviews_remaining == 0` the tool refuses further reviews with the
   budget-exhausted directive — the budget is enforced in code, not prompt.
5. Budget comes from `SolvayConfig.solver_loop.max_iterations` (finally
   wired).

### 2.2 Robustness rules

- **Critic structured-output failure:** retry once; if it fails again, return
  a degraded verdict `{approved: false, issues: ["<role> verdict unavailable:
  <reason>"], severity: "minor"}`. Never a synthetic blocker; never burns the
  solver's context with parse noise.
- **Critic exception** (network, timeout): same degraded-verdict path.
- The pydantic-state wiring risk disappears with the subgraph.

### 2.3 Solver output contract

New schema `SolverReport` (in `schemas.py`), used as the solver subagent's
`response_format`:

```python
class SolverReport(BaseModel):
    solver_blocked: bool = False
    blocked_topic: str | None = None
    draft: SolutionDraft | None = None
    termination_reason: Literal["consensus", "budget_exhausted", "judge_forced"]
    iterations_consumed: int
    open_issues: list[str] = []
```

- `judge_forced` + `blocked_topic` preserves the orchestrator's existing
  re-research flow unchanged (max one re-research per run).
- `open_issues` carries unresolved critic objections on
  `budget_exhausted` (replaces the old `critique_history` blob — the full
  history lives in the solver's transcript and LangSmith traces).

`SolverResponse`, `SolverLoopGraphState`, and (if orphaned) `SolverLoopState`
/ `CritiqueEntry` are removed with the subgraph.

---

## 3. File-level changes

| File | Change |
|---|---|
| `src/solvay/subagents/solver_loop.py` | replaced by `src/solvay/subagents/solver.py`: `create_solver_subagent(config, web_search, url_fetch)` returning a dict subagent, plus `create_request_review_tool(config, web_search)` factory |
| `src/solvay/agent.py` | drop verifier/peer_reviewer from `subagents`; register the new solver dict |
| `src/solvay/schemas.py` | add `SolverReport`; remove `SolverResponse` and orphaned loop-state models |
| `src/solvay/prompts/solver.md` | rewrite: conversational flow — solve, call `request_review`, address issues, repeat until the tool says finalize; keep notebook convention |
| `src/solvay/prompts/verifier.md`, `peer_reviewer.md` | remove notebook/harness-notes instructions (no filesystem tools); keep check methodology |
| `src/solvay/prompts/orchestrator.md` | update solver contract to `SolverReport` |
| `tests/test_solver_loop.py` | replaced by `tests/test_solver.py`: request_review unit tests (parallelism, budget, consensus directive, retry/degradation) with fake critics; solver factory wiring test |
| `tests/test_agent.py`, smoke tests | adapt to the new subagent roster |

---

## 4. Testing

- **Unit:** `request_review` with fake critic runnables — consensus directive,
  budget enforcement (counter in closure), parallel invocation, retry-once and
  degraded-verdict on validation failure/exception.
- **Wiring:** solver factory exposes the expected tools and
  `response_format=SolverReport`; orchestrator registers exactly four
  subagents.
- **E2E verification:** `uv run pytest`, then one real
  `solvay solve` run against the sample mechanics problem to confirm the
  conversational loop and the orchestrator handoff.

---

## 5. Risks

- **Tool inheritance semantics:** the dict subagent's `tools` list overrides
  inherited tools; filesystem tools come from the harness middleware and are
  expected to remain. Verified against current deepagents docs; the smoke run
  is the final check.
- **Solver discipline:** without a hard graph, the solver could skip
  `request_review`. Mitigation: prompt requires at least one review before
  finalizing; `iterations_consumed` in the report makes skipping visible.
