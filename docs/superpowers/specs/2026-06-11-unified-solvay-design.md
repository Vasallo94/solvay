# Unified Solvay — Design Spec

**Date:** 2026-06-11
**Status:** Approved
**Branch base:** current `main` (commit `3fde603`)

## 1. Context

Two lines of work diverged from the 2026-05-20 base:

- **main (2026-06-05..08, Enrique + Opus 4.6):** a robust two-stage benchmark
  grader (`grading.py`, SymPy + LLM-judge), 9 research-grade problems, local
  Ollama model handling (`resolve_model`, `model_kwargs`, `_is_local_model`),
  `SubagentCallLimitMiddleware` anti-loop, a deterministic 6-step orchestrator,
  AFP friction detectors, `regrade.py`, and a security patch.
- **feat/benchmark-suite (2026-06-10..11, Fable):** a conversational solver
  redesign, a parallel (weaker) benchmark grader, and 17 textbook problems —
  all built on the *stale* 2026-05-20 base, unaware of main's progress.

This spec reconciles both into one coherent system **on top of current main**.
Decisions taken with the user:

1. Keep main's deterministic orchestrator and its top-level `peer_reviewer`
   Step 5 (the solver's internal review and the orchestrator's final review
   coexist).
2. Use a **single conversational solver for all model types** (local + API),
   accepting the known risk that local tool-calling may not fire `request_review`.
3. Main's grader wins; Fable's `evaluation.py`/FINAL-ANSWER work is dropped.
   Add Fable's 17 textbook problems.

## 2. Solver: conversational, hardened for local models

`create_solver_subagent` is rewritten to return a conversational solver while
inheriting main's local-model handling.

- **Model:** `resolve_model(config.model_for("solver"), **config.model_kwargs)`
  so Ollama gets the high `num_predict` that caps Qwen thinking-mode runaway.
- **Agent:** `create_agent` with tools
  `[python_exec, check_dimensions, web_search, url_fetch, request_review]` and
  `middleware=[ModelCallLimitMiddleware(run_limit=25)]` to bound loops.
- **Internal review loop:** `request_review(problem, draft)` runs the verifier
  and peer-reviewer critics in parallel under a code-enforced budget
  (`config.solver_loop.max_iterations`), via `run_critic` (retry once, then a
  degraded non-blocking verdict). The internal reviewer critic also gets main's
  `physics_checklist` tool.
- **Tolerant output (local safety):** the solver returns a `SolverReport`. If
  structured output fails (common on Ollama), parse best-effort from the last
  message. If the model finalizes a draft without ever calling
  `request_review`, accept it with `termination_reason="no_review"` and an
  `open_issues` note — never hang.
- **Critics:** verifier tools `[python_exec, check_dimensions]`; peer-reviewer
  tools `[python_exec, web_search, physics_checklist]`. Each critic built with
  `resolve_model(..., **model_kwargs)` and `ModelCallLimitMiddleware`.

### 2.1 Known risk (accepted)

With qwen/Ollama, `request_review` tool-calls may not fire. The tolerant-output
path prevents hangs but such a run has no internal review (visible in
`open_issues`). `solver_loop.py` stays in the repo **uncwired** as a safety net
until local is validated on the M5 Pro; it is then a candidate for deletion.

## 3. Orchestrator: unchanged (drop-in)

The solver returns a payload the orchestrator's Step 4 can read: it exposes
`final_draft` and `termination_reason`. No changes to `orchestrator.md`,
`SubagentCallLimitMiddleware`, the subagent roster, or the `gp_as_solver`
general-purpose override. Only the solver runnable changes.

`SolverReport.draft` is surfaced as `final_draft` in the returned JSON so the
orchestrator prompt's existing contract holds. `termination_reason` values:
`consensus | budget_exhausted | judge_forced | no_review`.

## 4. Benchmark

- **Grader:** main's `grading.py` is authoritative. Drop Fable's
  `evaluation.py`, `answer_format.py`, the FINAL-ANSWER profile changes, the
  `run_bench.py` deletion, and the `--composer/--probe` CLI flags.
- **Problems:** add Fable's 17 textbook problems (mechanics 7, em 3, thermo 3,
  quantum 2, astro 2; 15 numeric + 2 symbolic) under `benchmark/problems/<domain>/`.
  Schema is identical to main's, so they load unchanged.
- **Validation:** run main's grader over the 17 problems with a stub answer
  equal to each `expected.value` and confirm all grade correct (covers the 2
  symbolic problems and unit handling).

## 5. Schemas and cleanup

- Add `SolverReport` to `schemas.py`.
- Keep `SolverResponse`, `CritiqueEntry`, `SolverLoopState` while
  `solver_loop.py` remains (safety net). They are removed together once the
  subgraph is deleted post-local-validation.

## 6. Files

| Action | File |
|---|---|
| Rewrite | `src/solvay/subagents/solver.py` (new conversational factory + `request_review` + `run_critic`) |
| Modify | `src/solvay/agent.py` (point solver wiring at the new factory; keep middleware, roster, gp_as_solver) |
| Add | `src/solvay/schemas.py` → `SolverReport` |
| Rewrite | `src/solvay/prompts/solver.md` (conversational, tolerant-output guidance) |
| Keep | `solver_loop.py`, `orchestrator.md`, `grading.py`, profiles, middleware, critics' prompts/factories |
| Add | `benchmark/problems/{mechanics,em,thermo,quantum,astro}/synth-*.json` (17) |
| Add | `tests/test_solver.py` (conversational solver + request_review) |
| Add | `tests/benchmark/test_handcrafted_problems.py` (grader validates the 17) |

## 7. Verification

`uv run pytest`, `uv run mypy src tests`, `uv run ruff check .`,
`uv run ruff format --check .` (the check that failed CI on the old PR). Plus
the grader-over-17-problems validation in §4.

## 8. Out of scope

- Live API smoke run (no `ANTHROPIC_API_KEY` in dev env) — run on M5 Pro.
- Deleting `solver_loop.py` and the orphaned schemas — deferred to post-local-validation.
- Re-adding `--composer/--probe` CLI flags — only if requested later.
