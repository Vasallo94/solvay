# Unified Solvay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace main's deterministic solver subgraph with a conversational solver (kept as a drop-in `CompiledSubAgent`), hardened for local models, and add 17 textbook benchmark problems — all on top of current `main`, leaving main's orchestrator, middleware, and grader untouched.

**Architecture:** The solver becomes a `create_agent` conversational agent that calls a `request_review` tool (parallel critics, code-enforced budget). A thin output-normalizing wrapper turns its result into the exact JSON payload (`final_draft`, `termination_reason`, …) the orchestrator already reads, so it drops into `agent.py` by changing one import. Benchmark grading stays main's `grading.py`; we only add problem JSONs.

**Tech Stack:** Python 3.14, deepagents, langchain `create_agent` + `ModelCallLimitMiddleware`, pydantic v2, pytest, uv. Spec: `docs/superpowers/specs/2026-06-11-unified-solvay-design.md`.

**Branch:** `feat/conversational-solver` (already created from `main`). Source for ported pieces: `feat/benchmark-suite`.

**Conventions:** All code/comments/commits in English. Run from repo root. Gates: `uv run pytest`, `uv run mypy src tests`, `uv run ruff check .`, `uv run ruff format --check .` (the last one failed the old PR's CI — it must pass).

---

### Task 1: `SolverReport` schema

**Files:**
- Modify: `src/solvay/schemas.py` (add after `SolverResponse`)
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_schemas.py`. First ensure the import block imports `SolverReport` and that `pytest` and `from pydantic import ValidationError` are present (add only if missing).

```python
class TestSolverReport:
    def _draft(self) -> SolutionDraft:
        return SolutionDraft(
            method="Newton",
            steps=["F=ma"],
            final_answer=Quantity(value=4.9, unit="m/s^2"),
            code_trace=["9.81*sin(pi/6)"],
        )

    def test_minimal_consensus_report(self) -> None:
        r = SolverReport(draft=self._draft(), termination_reason="consensus", iterations_consumed=1)
        assert r.solver_blocked is False
        assert r.blocked_topic is None
        assert r.open_issues == []
        assert r.termination_reason == "consensus"

    def test_no_review_and_blocked_allow_missing_draft(self) -> None:
        r = SolverReport(termination_reason="no_review", iterations_consumed=0)
        assert r.draft is None
        b = SolverReport(
            solver_blocked=True, blocked_topic="GR", termination_reason="judge_forced",
            iterations_consumed=1,
        )
        assert b.blocked_topic == "GR"

    def test_termination_reason_constrained(self) -> None:
        with pytest.raises(ValidationError):
            SolverReport(termination_reason="gave_up", iterations_consumed=1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_schemas.py -k SolverReport -v`
Expected: FAIL with `ImportError: cannot import name 'SolverReport'`.

- [ ] **Step 3: Implement**

In `src/solvay/schemas.py`, immediately after the `SolverResponse` class:

```python
class SolverReport(BaseModel):
    """Final report from the conversational solver."""

    solver_blocked: bool = False
    blocked_topic: str | None = None
    draft: SolutionDraft | None = None
    termination_reason: Literal["consensus", "budget_exhausted", "judge_forced", "no_review"]
    iterations_consumed: int
    open_issues: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_schemas.py -k SolverReport -v` → 3 PASS.
Then `uv run mypy src tests` (this repo errors on unused `# type: ignore`; if the `pytest.raises` line needs none, add none).

- [ ] **Step 5: Commit**

```bash
git add src/solvay/schemas.py tests/test_schemas.py
git commit -m "feat: add SolverReport schema for conversational solver"
```

---

### Task 2: `run_critic` helper

**Files:**
- Create: `src/solvay/subagents/solver.py`
- Create: `tests/test_solver.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solver.py`:

```python
"""Tests for the conversational solver and request_review tool."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from solvay.schemas import Verdict
from solvay.subagents.solver import run_critic


def _verdict(approved: bool, severity: str = "none") -> Verdict:
    return Verdict(approved=approved, issues=[] if approved else ["Issue"], severity=severity)


def _critic(verdict: Verdict) -> Runnable[Any, Any]:
    return RunnableLambda(lambda _s: {"structured_response": verdict})


class TestRunCritic:
    def test_returns_verdict_dict(self) -> None:
        assert run_critic(_critic(_verdict(True)), "verifier", "p") == {
            "approved": True, "issues": [], "severity": "none",
        }

    def test_retries_once_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky(_s: Any) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return {"structured_response": _verdict(True)}

        assert run_critic(RunnableLambda(flaky), "verifier", "p")["approved"] is True
        assert calls["n"] == 2

    def test_degrades_after_two_failures(self) -> None:
        def broken(_s: Any) -> dict[str, Any]:
            raise RuntimeError("down")

        r = run_critic(RunnableLambda(broken), "peer_reviewer", "p")
        assert r["approved"] is False
        assert r["severity"] == "minor"
        assert "peer_reviewer verdict unavailable" in r["issues"][0]
        assert "down" in r["issues"][0]

    def test_degrades_on_missing_structured_response(self) -> None:
        empty: Runnable[Any, Any] = RunnableLambda(lambda _s: {"messages": []})
        r = run_critic(empty, "verifier", "p")
        assert r["severity"] == "minor"

    def test_validates_plain_dict_response(self) -> None:
        raw: Runnable[Any, Any] = RunnableLambda(
            lambda _s: {"structured_response": {"approved": True, "issues": [], "severity": "none"}}
        )
        assert run_critic(raw, "verifier", "p") == {"approved": True, "issues": [], "severity": "none"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'solvay.subagents.solver'`.

- [ ] **Step 3: Implement**

Create `src/solvay/subagents/solver.py`:

```python
"""Conversational solver: a request_review-driven review loop as a drop-in subagent."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from solvay.schemas import Verdict


def run_critic(critic: Runnable[Any, Any], role: str, prompt: str) -> dict[str, Any]:
    """Invoke a critic agent and return its Verdict as a dict.

    Retries once on any failure. If both attempts fail, returns a degraded
    non-blocking verdict so a critic outage never poisons the review loop.
    """
    last_error = "no structured verdict returned"
    for _attempt in range(2):
        try:
            result = critic.invoke({"messages": [HumanMessage(content=prompt)]})
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the loop
            last_error = str(exc)
            continue
        structured = result.get("structured_response") if isinstance(result, dict) else None
        if structured is None:
            continue
        try:
            verdict = structured if isinstance(structured, Verdict) else Verdict.model_validate(structured)
        except ValueError as exc:
            last_error = str(exc)
            continue
        return verdict.model_dump()
    return Verdict(
        approved=False,
        issues=[f"{role} verdict unavailable: {last_error}"],
        severity="minor",
    ).model_dump()
```

NOTE: after writing, run `uv run ruff check src/solvay/subagents/solver.py`. If `BLE001` is not enabled (it is not in this repo) the `# noqa: BLE001` triggers `RUF100`; if so, change the comment to `# noqa` with no code, or remove it if ruff stays clean without it. Both `ruff check` and `mypy` must end clean.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_solver.py -v` → 5 PASS. Then `uv run mypy src tests` and `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/subagents/solver.py tests/test_solver.py
git commit -m "feat: add run_critic with retry-once and degraded verdicts"
```

---

### Task 3: `create_request_review_tool` (parallel critics, code-enforced budget)

**Files:**
- Modify: `src/solvay/subagents/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_solver.py` add `import json`, `import threading`, and extend the solver import to:

```python
from solvay.subagents.solver import (
    DIRECTIVE_BUDGET_EXHAUSTED,
    DIRECTIVE_CONSENSUS,
    DIRECTIVE_ITERATE,
    create_request_review_tool,
    run_critic,
)
```

Append:

```python
class TestRequestReview:
    def test_consensus_when_both_approve(self) -> None:
        tool, _state = create_request_review_tool(_critic(_verdict(True)), _critic(_verdict(True)), 3)
        p = json.loads(tool("prob", "draft"))
        assert p["directive"] == DIRECTIVE_CONSENSUS
        assert p["reviews_remaining"] == 2
        assert p["verifier"]["approved"] is True
        assert p["peer_reviewer"]["approved"] is True

    def test_iterate_when_rejected_with_budget(self) -> None:
        tool, _state = create_request_review_tool(_critic(_verdict(False, "blocker")), _critic(_verdict(True)), 3)
        p = json.loads(tool("prob", "draft"))
        assert p["directive"] == DIRECTIVE_ITERATE
        assert p["reviews_remaining"] == 2

    def test_budget_exhausted_on_last_rejected(self) -> None:
        tool, _state = create_request_review_tool(_critic(_verdict(False, "minor")), _critic(_verdict(True)), 1)
        p = json.loads(tool("prob", "draft"))
        assert p["directive"] == DIRECTIVE_BUDGET_EXHAUSTED
        assert p["reviews_remaining"] == 0

    def test_refuses_after_budget_without_calling_critics(self) -> None:
        calls = {"n": 0}

        def counting(_s: Any) -> dict[str, Any]:
            calls["n"] += 1
            return {"structured_response": _verdict(False)}

        c = RunnableLambda(counting)
        tool, state = create_request_review_tool(c, c, 1)
        tool("p", "d1")
        assert calls["n"] == 2
        p = json.loads(tool("p", "d2"))
        assert p["directive"] == DIRECTIVE_BUDGET_EXHAUSTED
        assert p["verifier"] is None and p["peer_reviewer"] is None
        assert calls["n"] == 2
        assert state["used"] == 1

    def test_critics_run_in_parallel(self) -> None:
        # timeout=5: on timeout run_critic degrades to approved=False, which
        # breaks consensus and fails the assertion clearly rather than hanging.
        barrier = threading.Barrier(2, timeout=5)

        def synced(_s: Any) -> dict[str, Any]:
            barrier.wait()
            return {"structured_response": _verdict(True)}

        c = RunnableLambda(synced)
        tool, _state = create_request_review_tool(c, c, 1)
        assert json.loads(tool("p", "d"))["directive"] == DIRECTIVE_CONSENSUS
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_solver.py -v`
Expected: FAIL with `ImportError: cannot import name 'DIRECTIVE_BUDGET_EXHAUSTED'`.

- [ ] **Step 3: Implement**

In `src/solvay/subagents/solver.py` add imports `import json`, `import threading`, `from collections.abc import Callable`, `from concurrent.futures import ThreadPoolExecutor`. Append:

```python
DIRECTIVE_CONSENSUS = "consensus -- finalize now"
DIRECTIVE_BUDGET_EXHAUSTED = "budget exhausted -- finalize with best effort"
DIRECTIVE_ITERATE = "address the issues and request review again"


def create_request_review_tool(
    verifier: Runnable[Any, Any],
    reviewer: Runnable[Any, Any],
    max_reviews: int,
) -> tuple[Callable[[str, str], str], dict[str, int]]:
    """Create the request_review tool plus its shared counter state.

    The counter lives in the returned ``state`` dict so the solver wrapper can
    read ``state["used"]`` for ``iterations_consumed``. The budget is enforced
    in code: the solver cannot exceed it regardless of prompt adherence.
    """
    lock = threading.Lock()
    state = {"used": 0}

    def request_review(problem: str, draft: str) -> str:
        """Submit the current solution draft for independent review.

        Args:
            problem: Short restatement of the problem.
            draft: Full current draft: method, numbered steps, final answer
                with units, key code snippets.

        Returns:
            JSON with verifier and peer_reviewer verdicts, reviews_remaining,
            and a directive telling you whether to finalize or revise.
        """
        with lock:
            if state["used"] >= max_reviews:
                return json.dumps({
                    "directive": DIRECTIVE_BUDGET_EXHAUSTED,
                    "reviews_remaining": 0, "verifier": None, "peer_reviewer": None,
                })
            state["used"] += 1
        prompt = f"Problem:\n{problem}\n\nSolution draft to review:\n{draft}"
        with ThreadPoolExecutor(max_workers=2) as pool:
            vf = pool.submit(run_critic, verifier, "verifier", prompt)
            rf = pool.submit(run_critic, reviewer, "peer_reviewer", prompt)
            verifier_verdict, reviewer_verdict = vf.result(), rf.result()
        remaining = max_reviews - state["used"]
        if verifier_verdict["approved"] and reviewer_verdict["approved"]:
            directive = DIRECTIVE_CONSENSUS
        elif remaining == 0:
            directive = DIRECTIVE_BUDGET_EXHAUSTED
        else:
            directive = DIRECTIVE_ITERATE
        return json.dumps({
            "directive": directive, "reviews_remaining": remaining,
            "verifier": verifier_verdict, "peer_reviewer": reviewer_verdict,
        })

    return request_review, state
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_solver.py -v` → 10 PASS. Then `uv run mypy src tests` and `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/subagents/solver.py tests/test_solver.py
git commit -m "feat: add request_review tool with parallel critics and code-enforced budget"
```

---

### Task 4: critics + conversational agent + output wrapper + factory

**Files:**
- Modify: `src/solvay/subagents/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_solver.py` add `from unittest.mock import patch` and `from langchain_core.messages import AIMessage`. Append:

```python
class TestExtractReport:
    def test_prefers_structured_response(self) -> None:
        from solvay.schemas import SolutionDraft, SolverReport
        from solvay.subagents.solver import extract_report

        draft = SolutionDraft(method="m", steps=["s"], final_answer="g*sin(theta)", code_trace=[])
        report = SolverReport(draft=draft, termination_reason="consensus", iterations_consumed=2)
        result = {"structured_response": report, "messages": [AIMessage(content="ignored")]}
        out = extract_report(result, used=2)
        assert out.termination_reason == "consensus"
        assert out.iterations_consumed == 2

    def test_no_structured_response_falls_back_to_no_review(self) -> None:
        from solvay.subagents.solver import extract_report

        result = {"messages": [AIMessage(content="The acceleration is 4.9 m/s^2")]}
        out = extract_report(result, used=0)
        assert out.termination_reason == "no_review"
        assert out.iterations_consumed == 0
        assert out.draft is not None
        assert "4.9 m/s^2" in out.draft.final_answer
        assert any("no_review" in i or "did not request review" in i for i in out.open_issues)


class TestSolverPayload:
    def test_wrapper_emits_orchestrator_payload(self) -> None:
        import json

        from solvay.config import SolvayConfig
        from solvay.schemas import SolutionDraft, SolverReport
        from solvay.subagents import solver as M

        draft = SolutionDraft(method="Newton", steps=["a=g sin"], final_answer="4.9 m/s^2", code_trace=[])
        report = SolverReport(draft=draft, termination_reason="consensus", iterations_consumed=1)
        fake_agent = RunnableLambda(lambda _s: {"structured_response": report})

        with patch.object(M, "_build_solver_components", return_value=(fake_agent, {"used": 1})):
            sub = M.create_solver_subagent(SolvayConfig(), lambda q: {}, lambda u: "")
        payload = json.loads(sub["runnable"].invoke({"messages": []})["messages"][-1].content)
        assert payload["termination_reason"] == "consensus"
        assert payload["iterations_consumed"] == 1
        assert payload["final_draft"]["method"] == "Newton"

    def test_factory_returns_compiled_subagent_named_solver(self) -> None:
        from solvay.config import SolvayConfig
        from solvay.subagents import solver as M

        fake_agent = RunnableLambda(lambda _s: {"messages": [AIMessage(content="x")]})
        with patch.object(M, "_build_solver_components", return_value=(fake_agent, {"used": 0})):
            sub = M.create_solver_subagent(SolvayConfig(), lambda q: {}, lambda u: "")
        assert sub["name"] == "solver"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_solver.py -k "ExtractReport or SolverPayload" -v`
Expected: FAIL with `ImportError` / `AttributeError` (symbols not defined).

- [ ] **Step 3: Implement**

In `src/solvay/subagents/solver.py` extend imports:

```python
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from solvay.config import SolvayConfig, resolve_model
from solvay.schemas import SolutionDraft, SolverReport, Verdict
from solvay.subagents import load_prompt
from solvay.tools.dimensional import check_dimensions
from solvay.tools.physics_checklist import physics_checklist
from solvay.tools.python_exec import python_exec
```

(Keep the existing `from solvay.schemas import Verdict` merged into the line above; remove the duplicate.) Append:

```python
def _build_solver_components(
    config: SolvayConfig,
    web_search_tool: Any,
    url_fetch_tool: Any,
) -> tuple[Runnable[Any, Any], dict[str, int]]:
    """Build the conversational solver agent and expose the review counter.

    Inherits main's local-model handling: models are resolved with
    ``config.model_kwargs`` (e.g. ``num_predict`` for Ollama) and every agent
    is bounded by ``ModelCallLimitMiddleware``.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware

    mkwargs = config.model_kwargs
    call_limit = [ModelCallLimitMiddleware(run_limit=25)]

    verifier = create_agent(
        resolve_model(config.model_for("verifier"), **mkwargs),
        system_prompt=load_prompt("verifier"),
        tools=[python_exec, check_dimensions],
        response_format=Verdict,
        middleware=call_limit,
        name="verifier",
    )
    reviewer = create_agent(
        resolve_model(config.model_for("peer_reviewer"), **mkwargs),
        system_prompt=load_prompt("peer_reviewer"),
        tools=[python_exec, web_search_tool, physics_checklist],
        response_format=Verdict,
        middleware=call_limit,
        name="peer_reviewer",
    )
    request_review, state = create_request_review_tool(
        verifier, reviewer, config.solver_loop.max_iterations
    )
    solver_agent = create_agent(
        resolve_model(config.model_for("solver"), **mkwargs),
        system_prompt=load_prompt("solver"),
        tools=[python_exec, check_dimensions, web_search_tool, url_fetch_tool, request_review],
        response_format=SolverReport,
        middleware=call_limit,
        name="solver",
    )
    return solver_agent, state


def extract_report(result: Any, used: int) -> SolverReport:
    """Extract a SolverReport from a conversational agent result, best-effort.

    Prefers the structured response. Falls back (common on local models that
    do not emit structured output) to a no_review report built from the last
    message text, so the solver never hangs and the missing review is visible.
    """
    structured = result.get("structured_response") if isinstance(result, dict) else None
    if structured is not None:
        report = structured if isinstance(structured, SolverReport) else SolverReport.model_validate(structured)
        return report.model_copy(update={"iterations_consumed": used})
    text = ""
    if isinstance(result, dict) and result.get("messages"):
        text = str(getattr(result["messages"][-1], "content", ""))
    draft = SolutionDraft(method="(unstructured)", steps=[text] if text else [], final_answer=text, code_trace=[])
    return SolverReport(
        draft=draft,
        termination_reason="no_review",
        iterations_consumed=used,
        open_issues=["solver did not request review (no_review fallback)"],
    )


def create_solver_subagent(
    config: SolvayConfig,
    web_search_tool: Any,
    url_fetch_tool: Any,
) -> dict[str, Any]:
    """Create the conversational solver as a drop-in CompiledSubAgent dict.

    The returned runnable invokes the conversational agent and normalizes its
    output to the JSON payload the orchestrator's Step 4 already reads
    (``final_draft`` + ``termination_reason`` + ``iterations_consumed`` …).
    """
    solver_agent, state = _build_solver_components(config, web_search_tool, url_fetch_tool)

    def _run(inputs: dict[str, Any]) -> dict[str, Any]:
        result = solver_agent.invoke(inputs)
        report = extract_report(result, used=state["used"])
        payload = {
            "final_draft": report.draft.model_dump() if report.draft else None,
            "termination_reason": report.termination_reason,
            "iterations_consumed": report.iterations_consumed,
            "open_issues": report.open_issues,
            "solver_blocked": report.solver_blocked,
            "blocked_topic": report.blocked_topic,
        }
        return {"messages": [AIMessage(content=json.dumps(payload, default=str))]}

    return {
        "name": "solver",
        "description": (
            "Solve a physics problem through a conversational solve-review loop. "
            "Accepts ProblemSpec and ResearchBrief, returns a draft with "
            "termination reason and open issues."
        ),
        "runnable": RunnableLambda(_run),
    }
```

NOTE on the return type: main's `agent.py` registers the solver object directly in `subagents=[...]` AND reads `solver["runnable"]`/`solver["description"]` to build `gp_as_solver`. `deepagents.CompiledSubAgent` is a TypedDict, so a plain dict with `name`/`description`/`runnable` keys is structurally identical and works for both. If `create_deep_agent` rejects the plain dict at runtime (Task 8 step 3 smoke will catch this), wrap the return value with `from deepagents import CompiledSubAgent` → `return CompiledSubAgent(name="solver", description=..., runnable=RunnableLambda(_run))`; the subscript access `solver["runnable"]` still works because it is a TypedDict.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_solver.py -v` (≥14 PASS). Then `uv run mypy src tests` and `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/subagents/solver.py tests/test_solver.py
git commit -m "feat: conversational solver agent with tolerant output wrapper"
```

---

### Task 5: rewrite `solver.md` prompt

**Files:**
- Modify: `src/solvay/prompts/solver.md` (full rewrite)
- Test: `tests/test_prompts.py` (only if it asserts on solver.md — check first)

- [ ] **Step 1: Check existing prompt assertions**

Run: `grep -n "solver" tests/test_prompts.py`
If a test asserts solver.md contains `/workspace/lab_notebook.md` or `/memories/solvay/harness_notes.md`, keep those strings in the rewrite below (they are already included).

- [ ] **Step 2: Rewrite the prompt**

Replace the full contents of `src/solvay/prompts/solver.md` with:

```markdown
# Solvay Solver

You are the solver agent. Solve the physics problem step by step in this
conversation, using your tools to compute and to get your work reviewed.

## Workflow

1. Read `/workspace/lab_notebook.md` for context from earlier agents.
2. Solve the problem:
   - ALWAYS use `python_exec` to verify calculations -- no mental math.
   - ALWAYS call `check_dimensions` on the final answer.
3. Call `request_review(problem, draft)` with a short problem restatement and
   your full draft: method, numbered steps, final answer with units, and key
   code snippets.
4. Act on the returned directive:
   - "consensus -- finalize now": stop and emit your final report.
   - "address the issues and request review again": fix EVERY listed issue,
     noting how you addressed each, then call `request_review` again.
   - "budget exhausted -- finalize with best effort": emit your final report
     and copy unresolved objections into `open_issues`.
5. Get at least one review before finalizing, unless you are blocked on
   missing theory (see below).

## Blocked on missing theory

If you need physics theory missing from the research brief, do not guess:
finalize immediately with `solver_blocked=true`, the missing topic in
`blocked_topic`, and `termination_reason="judge_forced"`.

## Final report (SolverReport)

- `solver_blocked`: false unless blocked on missing theory
- `blocked_topic`: the missing topic when blocked, else null
- `draft`: method, steps, final_answer {value, unit}, code_trace
- `termination_reason`: "consensus" | "budget_exhausted" | "judge_forced"
- `iterations_consumed`: how many times you called `request_review`
- `open_issues`: unresolved critic objections (empty on consensus)

## Other tools

- `web_search(query)` -- ONLY for library/API usage (e.g. "sympy Lagrangian").
  Do NOT search for physics theory.
- `url_fetch(url)` -- read documentation pages.

## Rules

- After finishing: append a brief entry (at most 4 lines) to
  `/workspace/lab_notebook.md`.
- If a tool fails or the harness blocks the task, append a short note to
  `/memories/solvay/harness_notes.md` using:
  - Symptom: ...
  - Context: ...
  - Likely cause: ...
  - Suggested fix: ...
- Never include secrets, API keys, passwords, or credentials in notes.
- All output in English.
```

- [ ] **Step 3: Run prompt tests**

Run: `uv run pytest tests/test_prompts.py -v` → PASS (the rewrite keeps the notebook + harness-notes strings).

- [ ] **Step 4: Commit**

```bash
git add src/solvay/prompts/solver.md
git commit -m "feat: rewrite solver prompt for conversational review loop"
```

---

### Task 6: wire the conversational solver into `agent.py`

**Files:**
- Modify: `src/solvay/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
def test_solver_wired_from_conversational_module() -> None:
    # The solver factory must come from the conversational module, not solver_loop.
    import solvay.agent as agent_module

    assert agent_module.create_solver_subagent.__module__ == "solvay.subagents.solver"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_solver_wired_from_conversational_module -v`
Expected: FAIL — `__module__` is `solvay.subagents.solver_loop` until the import is changed.

- [ ] **Step 3: Change the import only**

In `src/solvay/agent.py`, change the solver import line:

```python
from solvay.subagents.solver_loop import create_solver_subagent
```

to:

```python
from solvay.subagents.solver import create_solver_subagent
```

Leave everything else in `agent.py` unchanged: `create_solver_subagent(config, web_search, url_fetch)` still returns an object supporting `solver["runnable"]` and `solver["description"]`, so the existing `gp_as_solver = CompiledSubAgent(name="general-purpose", description=solver["description"], runnable=solver["runnable"])` and the `subagents=[...]` list keep working. (Main's `solver_loop.create_solver_subagent` is now unused but stays in the repo as the safety net per the spec.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent.py -v` → PASS. Then `uv run mypy src tests` and `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/agent.py tests/test_agent.py
git commit -m "feat: wire conversational solver as drop-in subagent"
```

---

### Task 7: add the 17 textbook problems + grader validation

**Files:**
- Add: `benchmark/problems/{mechanics,em,thermo,quantum,astro}/synth-*.json` (17 files from `feat/benchmark-suite`)
- Test: `tests/benchmark/test_handcrafted_problems.py`

- [ ] **Step 1: Bring the 17 problem files from the other branch**

```bash
git checkout feat/benchmark-suite -- \
  benchmark/problems/mechanics/synth-mech-a101.json \
  benchmark/problems/mechanics/synth-mech-a102.json \
  benchmark/problems/mechanics/synth-mech-a103.json \
  benchmark/problems/mechanics/synth-mech-a104.json \
  benchmark/problems/mechanics/synth-mech-a105.json \
  benchmark/problems/mechanics/synth-mech-a106.json \
  benchmark/problems/mechanics/synth-mech-a107.json \
  benchmark/problems/em/synth-em-b201.json \
  benchmark/problems/em/synth-em-b202.json \
  benchmark/problems/em/synth-em-b203.json \
  benchmark/problems/thermo/synth-thermo-c301.json \
  benchmark/problems/thermo/synth-thermo-c302.json \
  benchmark/problems/thermo/synth-thermo-c303.json \
  benchmark/problems/quantum/synth-quantum-d401.json \
  benchmark/problems/quantum/synth-quantum-d402.json \
  benchmark/problems/astro/synth-astro-e501.json \
  benchmark/problems/astro/synth-astro-e502.json
```

Run: `git status --porcelain benchmark/problems` and confirm 17 new files staged.

- [ ] **Step 2: Write the validation test**

Create `tests/benchmark/test_handcrafted_problems.py`. This proves main's SymPy grader self-scores each handcrafted problem correct when fed its own `expected.value` (covers numeric + the 2 symbolic problems offline, no LLM). First confirm the grader symbol exists: `grep -n "def sympy_grade" src/solvay/benchmark/grading.py`.

```python
"""Main's SymPy grader must self-score every handcrafted problem correct."""

from __future__ import annotations

from pathlib import Path

import pytest

from solvay.benchmark.grading import sympy_grade
from solvay.benchmark.schema import load_problems_dir

_PROBLEMS = [
    p for p in load_problems_dir(Path("benchmark/problems"))
    if p.id.startswith("synth-")
]


def test_seventeen_handcrafted_problems_present() -> None:
    assert len(_PROBLEMS) == 17


@pytest.mark.parametrize("problem", _PROBLEMS, ids=[p.id for p in _PROBLEMS])
def test_grader_self_scores_expected_value(problem) -> None:  # type: ignore[no-untyped-def]
    # Feeding the expected value as the answer must grade True via SymPy alone
    # (no LLM-judge fallback), proving the grader handles this problem's format.
    result = sympy_grade(problem.expected.value, problem.expected)
    assert result is True, f"{problem.id}: sympy_grade returned {result}"
```

- [ ] **Step 3: Run to verify**

Run: `uv run pytest tests/benchmark/test_handcrafted_problems.py -v`
Expected: 18 PASS (1 count + 17 parametrized). If `sympy_grade`'s signature differs (e.g. takes the answer differently), adjust the call to match the real signature found in step 2 — do NOT change the grader. If a specific symbolic problem returns `None` (grader defers to LLM), that is acceptable ONLY if the value is genuinely undecidable by SymPy; otherwise treat as a problem-format bug and fix the problem JSON's `expected.value`/`unit` to a SymPy-gradeable form.

- [ ] **Step 4: Commit**

```bash
git add benchmark/problems tests/benchmark/test_handcrafted_problems.py
git commit -m "feat(bench): add 17 handcrafted textbook problems with grader validation"
```

---

### Task 8: full verification

**Files:** none new.

- [ ] **Step 1: Lint, format, types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
Expected: all clean. `ruff format --check` is the gate that failed the old PR's CI — if it reports files, run `uv run ruff format .`, review the diff, and commit as `style: apply ruff format`.

- [ ] **Step 2: Full suite**

Run: `uv run pytest`
Expected: all PASS, no collection errors.

- [ ] **Step 3: Import-time smoke (agent builds without a live key)**

Run: `uv run python -c "from solvay.agent import create_solvay_agent; create_solvay_agent(); print('agent built')"`
Expected: prints `agent built` (no network call needed to construct the graph). If it fails on a missing key, that is a wiring bug — fix it.

- [ ] **Step 4: Document the local risk and live-smoke gap in the lab notebook**

Append to `lab_notebook.md` (repo root) a 2-line note: conversational solver is now wired for all model types; local (Ollama) tool-calling for `request_review` is unvalidated — run a live `solvay solve` and one `solvay bench` with `ollama:qwen3.5` on the M5 Pro before trusting solvay-profile numbers.

- [ ] **Step 5: Commit (if anything changed)**

```bash
git add -A
git commit -m "chore: post-verification fixes for unified solver"
```
```

---

## Self-review notes

- Spec §2 (conversational solver, local hardening) → Tasks 2-4 (resolve_model, model_kwargs, ModelCallLimitMiddleware, tolerant `extract_report`/no_review).
- Spec §2.1 (keep solver_loop.py as safety net) → Task 6 leaves it unused, not deleted.
- Spec §3 (orchestrator unchanged, drop-in) → Task 6 changes one import; Task 4 emits the `final_draft`/`termination_reason` payload.
- Spec §4 (grader of main + 17 problems) → Task 7; no grader changes.
- Spec §5 (SolverReport added; old schemas kept) → Task 1; old schemas untouched.
- Spec §7 (verification incl. `ruff format --check`) → Task 8.
