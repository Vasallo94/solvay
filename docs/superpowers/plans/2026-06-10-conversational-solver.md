# Conversational Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rigid LangGraph solver subgraph with a conversational solver subagent whose review loop runs through a budget-enforced `request_review` tool.

**Architecture:** The solver becomes a plain deepagents dict subagent that keeps full message history. A new `request_review` tool (closure over the two critic agents and the iteration budget) runs verifier and peer reviewer in parallel and returns verdicts plus a directive. `verifier`/`peer_reviewer` disappear from the orchestrator's subagent roster. Spec: `docs/superpowers/specs/2026-06-10-conversational-solver-design.md`.

**Tech Stack:** Python 3.14, deepagents, langchain `create_agent`, pydantic v2, pytest, uv.

**Conventions:** All code/comments/commits in English (project rule). Run all commands from the repo root. Test commands: `uv run pytest`, `uv run ruff check .`, `uv run mypy src tests`.

---

### Task 1: `SolverReport` schema

**Files:**
- Modify: `src/solvay/schemas.py` (add model after `SolutionDraft`)
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_schemas.py` (note: `pytest` and `ValidationError` are already imported in that file; verify and add imports only if missing — it needs `import pytest` and `from pydantic import ValidationError`). Also add `SolverReport` to the existing `from solvay.schemas import (...)` import block.

```python
class TestSolverReport:
    def _draft(self) -> SolutionDraft:
        return SolutionDraft(
            method="Newtonian mechanics",
            steps=["Apply F=ma along incline"],
            final_answer=Quantity(value=4.9, unit="m/s^2"),
            code_trace=["9.81*sin(pi/6)"],
        )

    def test_minimal_consensus_report(self) -> None:
        report = SolverReport(
            draft=self._draft(),
            termination_reason="consensus",
            iterations_consumed=1,
        )
        assert report.solver_blocked is False
        assert report.blocked_topic is None
        assert report.open_issues == []

    def test_blocked_report_allows_missing_draft(self) -> None:
        report = SolverReport(
            solver_blocked=True,
            blocked_topic="relativistic corrections",
            termination_reason="judge_forced",
            iterations_consumed=1,
        )
        assert report.draft is None

    def test_termination_reason_is_constrained(self) -> None:
        with pytest.raises(ValidationError):
            SolverReport(
                termination_reason="gave_up",  # type: ignore[arg-type]
                iterations_consumed=0,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py -k SolverReport -v`
Expected: FAIL with `ImportError: cannot import name 'SolverReport'`

- [ ] **Step 3: Implement the model**

In `src/solvay/schemas.py`, after the `SolutionDraft` class:

```python
class SolverReport(BaseModel):
    """Final report from the conversational solver subagent."""

    solver_blocked: bool = False
    blocked_topic: str | None = None
    draft: SolutionDraft | None = None
    termination_reason: Literal["consensus", "budget_exhausted", "judge_forced"]
    iterations_consumed: int
    open_issues: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py -k SolverReport -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/solvay/schemas.py tests/test_schemas.py
git commit -m "feat: add SolverReport schema for conversational solver"
```

---

### Task 2: `run_critic` helper (retry once, degrade on failure)

**Files:**
- Create: `src/solvay/subagents/solver.py`
- Create: `tests/test_solver.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solver.py`:

```python
"""Tests for the conversational solver subagent and request_review tool."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from solvay.schemas import Verdict
from solvay.subagents.solver import run_critic


def _verdict(approved: bool, severity: str = "none") -> Verdict:
    return Verdict(
        approved=approved,
        issues=[] if approved else ["Issue found"],
        severity=severity,  # type: ignore[arg-type]
    )


def _critic(verdict: Verdict) -> Runnable[Any, Any]:
    return RunnableLambda(lambda _state: {"structured_response": verdict})


class TestRunCritic:
    def test_returns_verdict_dict_on_success(self) -> None:
        result = run_critic(_critic(_verdict(True)), "verifier", "prompt")
        assert result == {"approved": True, "issues": [], "severity": "none"}

    def test_retries_once_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky(_state: Any) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return {"structured_response": _verdict(True)}

        result = run_critic(RunnableLambda(flaky), "verifier", "prompt")
        assert calls["n"] == 2
        assert result["approved"] is True

    def test_degrades_to_minor_verdict_after_two_failures(self) -> None:
        def broken(_state: Any) -> dict[str, Any]:
            raise RuntimeError("critic down")

        result = run_critic(RunnableLambda(broken), "peer_reviewer", "prompt")
        assert result["approved"] is False
        assert result["severity"] == "minor"
        assert "peer_reviewer verdict unavailable" in result["issues"][0]
        assert "critic down" in result["issues"][0]

    def test_degrades_when_structured_response_missing(self) -> None:
        empty: Runnable[Any, Any] = RunnableLambda(lambda _state: {"messages": []})
        result = run_critic(empty, "verifier", "prompt")
        assert result["approved"] is False
        assert result["severity"] == "minor"
        assert "verifier verdict unavailable" in result["issues"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'solvay.subagents.solver'`

- [ ] **Step 3: Implement `run_critic`**

Create `src/solvay/subagents/solver.py`:

```python
"""Conversational solver subagent: review loop driven by the request_review tool."""

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
            verdict = (
                structured
                if isinstance(structured, Verdict)
                else Verdict.model_validate(structured)
            )
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_solver.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/solvay/subagents/solver.py tests/test_solver.py
git commit -m "feat: add run_critic with retry-once and degraded verdicts"
```

---

### Task 3: `create_request_review_tool` (parallel critics, budget in code, directives)

**Files:**
- Modify: `src/solvay/subagents/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_solver.py`. Add these imports at the top of the file: `import json`, `import threading`, and extend the solver import to

```python
from solvay.subagents.solver import (
    DIRECTIVE_BUDGET_EXHAUSTED,
    DIRECTIVE_CONSENSUS,
    DIRECTIVE_ITERATE,
    create_request_review_tool,
    run_critic,
)
```

Then add:

```python
class TestRequestReview:
    def test_consensus_directive_when_both_approve(self) -> None:
        tool = create_request_review_tool(
            verifier=_critic(_verdict(True)),
            reviewer=_critic(_verdict(True)),
            max_reviews=3,
        )
        payload = json.loads(tool("incline problem", "draft v1"))
        assert payload["directive"] == DIRECTIVE_CONSENSUS
        assert payload["reviews_remaining"] == 2
        assert payload["verifier"]["approved"] is True
        assert payload["peer_reviewer"]["approved"] is True

    def test_iterate_directive_when_rejected_with_budget_left(self) -> None:
        tool = create_request_review_tool(
            verifier=_critic(_verdict(False, "blocker")),
            reviewer=_critic(_verdict(True)),
            max_reviews=3,
        )
        payload = json.loads(tool("incline problem", "draft v1"))
        assert payload["directive"] == DIRECTIVE_ITERATE
        assert payload["reviews_remaining"] == 2
        assert payload["verifier"]["issues"] == ["Issue found"]

    def test_budget_exhausted_directive_on_last_rejected_review(self) -> None:
        tool = create_request_review_tool(
            verifier=_critic(_verdict(False, "minor")),
            reviewer=_critic(_verdict(True)),
            max_reviews=1,
        )
        payload = json.loads(tool("incline problem", "draft v1"))
        assert payload["directive"] == DIRECTIVE_BUDGET_EXHAUSTED
        assert payload["reviews_remaining"] == 0

    def test_refuses_review_after_budget_without_calling_critics(self) -> None:
        calls = {"n": 0}

        def counting_critic(_state: Any) -> dict[str, Any]:
            calls["n"] += 1
            return {"structured_response": _verdict(False)}

        critic = RunnableLambda(counting_critic)
        tool = create_request_review_tool(verifier=critic, reviewer=critic, max_reviews=1)

        tool("p", "draft v1")
        assert calls["n"] == 2  # one verifier + one reviewer call

        payload = json.loads(tool("p", "draft v2"))
        assert payload["directive"] == DIRECTIVE_BUDGET_EXHAUSTED
        assert payload["reviews_remaining"] == 0
        assert payload["verifier"] is None
        assert payload["peer_reviewer"] is None
        assert calls["n"] == 2  # critics NOT invoked again

    def test_critics_run_in_parallel(self) -> None:
        # Both critics block on a 2-party barrier. If they ran sequentially,
        # the barrier would time out, both verdicts would degrade, and the
        # consensus assertion below would fail.
        barrier = threading.Barrier(2, timeout=5)

        def synced_critic(_state: Any) -> dict[str, Any]:
            barrier.wait()
            return {"structured_response": _verdict(True)}

        critic = RunnableLambda(synced_critic)
        tool = create_request_review_tool(verifier=critic, reviewer=critic, max_reviews=1)
        payload = json.loads(tool("p", "draft v1"))
        assert payload["directive"] == DIRECTIVE_CONSENSUS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_solver.py -v`
Expected: FAIL with `ImportError: cannot import name 'DIRECTIVE_BUDGET_EXHAUSTED'`

- [ ] **Step 3: Implement the tool factory**

In `src/solvay/subagents/solver.py`, add `import json`, `from collections.abc import Callable`, and `from concurrent.futures import ThreadPoolExecutor` to the imports, then append:

```python
DIRECTIVE_CONSENSUS = "consensus -- finalize now"
DIRECTIVE_BUDGET_EXHAUSTED = "budget exhausted -- finalize with best effort"
DIRECTIVE_ITERATE = "address the issues and request review again"


def create_request_review_tool(
    verifier: Runnable[Any, Any],
    reviewer: Runnable[Any, Any],
    max_reviews: int,
) -> Callable[[str, str], str]:
    """Create the request_review tool, closing over the critics and budget.

    The review counter lives in this closure: the budget is enforced in code,
    so the solver cannot exceed it regardless of prompt adherence.
    """
    state = {"used": 0}

    def request_review(problem: str, draft: str) -> str:
        """Submit the current solution draft for independent review.

        Args:
            problem: Short restatement of the problem being solved.
            draft: The full current draft: method, numbered steps, final
                answer with units, and key code snippets.

        Returns:
            JSON with the verifier and peer_reviewer verdicts, the number of
            reviews remaining, and a directive telling you whether to
            finalize or revise and request review again.
        """
        if state["used"] >= max_reviews:
            return json.dumps(
                {
                    "directive": DIRECTIVE_BUDGET_EXHAUSTED,
                    "reviews_remaining": 0,
                    "verifier": None,
                    "peer_reviewer": None,
                }
            )
        state["used"] += 1
        prompt = f"Problem:\n{problem}\n\nSolution draft to review:\n{draft}"
        with ThreadPoolExecutor(max_workers=2) as pool:
            verifier_future = pool.submit(run_critic, verifier, "verifier", prompt)
            reviewer_future = pool.submit(run_critic, reviewer, "peer_reviewer", prompt)
            verifier_verdict = verifier_future.result()
            reviewer_verdict = reviewer_future.result()

        remaining = max_reviews - state["used"]
        if verifier_verdict["approved"] and reviewer_verdict["approved"]:
            directive = DIRECTIVE_CONSENSUS
        elif remaining == 0:
            directive = DIRECTIVE_BUDGET_EXHAUSTED
        else:
            directive = DIRECTIVE_ITERATE
        return json.dumps(
            {
                "directive": directive,
                "reviews_remaining": remaining,
                "verifier": verifier_verdict,
                "peer_reviewer": reviewer_verdict,
            }
        )

    return request_review
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_solver.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add src/solvay/subagents/solver.py tests/test_solver.py
git commit -m "feat: add request_review tool with parallel critics and code-enforced budget"
```

---

### Task 4: Solver subagent factory + rewritten solver prompt

**Files:**
- Modify: `src/solvay/subagents/solver.py`
- Modify: `src/solvay/prompts/solver.md` (full rewrite)
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_solver.py` (add `from unittest.mock import patch` to the imports):

```python
class TestCreateSolverSubagent:
    def test_wires_tools_prompt_and_response_format(self) -> None:
        from solvay.config import SolvayConfig
        from solvay.schemas import SolverReport
        from solvay.subagents import solver as solver_module

        def fake_web_search(query: str) -> dict[str, Any]:
            """Fake web search."""
            return {"results": []}

        def fake_url_fetch(url: str) -> str:
            """Fake URL fetch."""
            return ""

        with patch.object(solver_module, "_build_critics") as fake_build:
            fake_build.return_value = (_critic(_verdict(True)), _critic(_verdict(True)))
            subagent = solver_module.create_solver_subagent(
                SolvayConfig(), fake_web_search, fake_url_fetch
            )

        assert subagent["name"] == "solver"
        assert subagent["response_format"] is SolverReport
        assert "request_review" in subagent["system_prompt"]
        tool_names = [t.__name__ for t in subagent["tools"]]
        assert tool_names == [
            "python_exec",
            "check_dimensions",
            "fake_web_search",
            "fake_url_fetch",
            "request_review",
        ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_solver.py::TestCreateSolverSubagent -v`
Expected: FAIL with `AttributeError: ... has no attribute '_build_critics'`

- [ ] **Step 3: Implement `_build_critics` and `create_solver_subagent`**

In `src/solvay/subagents/solver.py`, extend the imports:

```python
from solvay.config import SolvayConfig
from solvay.schemas import SolverReport, Verdict
from solvay.subagents import load_prompt
from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec
```

(replacing the existing bare `from solvay.schemas import Verdict`), then append:

```python
def _build_critics(
    config: SolvayConfig,
    web_search_tool: Any,
) -> tuple[Runnable[Any, Any], Runnable[Any, Any]]:
    """Build the verifier and peer-reviewer critic agents."""
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    verifier = create_agent(
        init_chat_model(config.model_for("verifier")),
        system_prompt=load_prompt("verifier"),
        tools=[python_exec, check_dimensions],
        response_format=Verdict,
        name="verifier",
    )
    reviewer = create_agent(
        init_chat_model(config.model_for("peer_reviewer")),
        system_prompt=load_prompt("peer_reviewer"),
        tools=[python_exec, web_search_tool],
        response_format=Verdict,
        name="peer_reviewer",
    )
    return verifier, reviewer


def create_solver_subagent(
    config: SolvayConfig,
    web_search_tool: Any,
    url_fetch_tool: Any,
) -> dict[str, Any]:
    """Create the conversational solver dict subagent.

    The solver keeps its full message history across review rounds: it sees
    its own drafts, tool calls, and the critiques returned by request_review.
    """
    verifier, reviewer = _build_critics(config, web_search_tool)
    request_review = create_request_review_tool(
        verifier=verifier,
        reviewer=reviewer,
        max_reviews=config.solver_loop.max_iterations,
    )
    return {
        "name": "solver",
        "description": (
            "Solve a physics problem conversationally: compute with python_exec, "
            "check dimensions, submit drafts via request_review, address the "
            "critiques, and return a SolverReport."
        ),
        "system_prompt": load_prompt("solver"),
        "model": config.model_for("solver"),
        "tools": [
            python_exec,
            check_dimensions,
            web_search_tool,
            url_fetch_tool,
            request_review,
        ],
        "response_format": SolverReport,
    }
```

- [ ] **Step 4: Rewrite the solver prompt**

Replace the full contents of `src/solvay/prompts/solver.md` with:

```markdown
# Solvay Solver

You are the solver agent. Solve the physics problem step by step in this
conversation, using your tools to compute and to get your work reviewed.

## Workflow

1. Read `/workspace/lab_notebook.md` for context from previous agents.
2. Solve the problem:
   - ALWAYS use `python_exec` to verify calculations -- no mental math.
   - ALWAYS call `check_dimensions` on the final answer.
3. Call `request_review(problem, draft)` with a short problem restatement and
   your full draft: method, numbered steps, final answer with units, and key
   code snippets.
4. Act on the returned directive:
   - "consensus -- finalize now": stop and emit your final report.
   - "address the issues and request review again": fix EVERY listed issue,
     explicitly noting how you addressed each one, then call
     `request_review` again with the revised draft.
   - "budget exhausted -- finalize with best effort": emit your final report
     and copy the unresolved objections into `open_issues`.
5. You MUST get at least one review before finalizing.

## Blocked on missing theory

If you need physics theory that is missing from the research brief, do not
guess: finalize immediately with `solver_blocked=true`, the missing topic in
`blocked_topic`, and `termination_reason="judge_forced"`. The orchestrator
will run more research and call you again.

## Final report (SolverReport)

- `solver_blocked`: false unless blocked on missing theory
- `blocked_topic`: the missing topic when blocked, else null
- `draft`: method, steps, final_answer {value, unit}, code_trace
- `termination_reason`: "consensus" | "budget_exhausted" | "judge_forced"
- `iterations_consumed`: how many times you called `request_review`
- `open_issues`: unresolved critic objections (empty on consensus)

## Other tools

- `web_search(query)` -- ONLY for library/API usage (e.g., "sympy Lagrangian
  example"). Do NOT search for physics theory.
- `url_fetch(url)` -- read documentation pages.

## Rules

- After finishing: append a brief entry (at most 4 lines) to
  `/workspace/lab_notebook.md`.
- If a tool fails, a path is confusing, dependencies are missing, structured
  output repeatedly fails, or the harness behavior blocks the task, append a
  short note to `/memories/solvay/harness_notes.md` using:
  - Symptom: ...
  - Context: ...
  - Likely cause: ...
  - Suggested fix: ...
  - Artifacts: ...
- Never include secrets, raw API keys, passwords, or private credentials in
  harness notes.
- All output in English.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_solver.py tests/test_prompts.py -v`
Expected: all PASS (test_prompts still passes — solver prompt keeps both notebook and harness-notes references)

- [ ] **Step 6: Commit**

```bash
git add src/solvay/subagents/solver.py src/solvay/prompts/solver.md tests/test_solver.py
git commit -m "feat: add conversational solver subagent factory and rewritten prompt"
```

---

### Task 5: Rewire the orchestrator (`agent.py`) and its prompt

**Files:**
- Modify: `src/solvay/agent.py`
- Modify: `src/solvay/prompts/orchestrator.md`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
def test_orchestrator_registers_exactly_four_subagents() -> None:
    from solvay.agent import create_solvay_agent

    with patch("solvay.agent.create_deep_agent") as fake_create:
        fake_create.return_value = object()
        create_solvay_agent()

    subagents = fake_create.call_args.kwargs["subagents"]
    names = [s["name"] for s in subagents]
    assert names == ["parser", "researcher", "solver", "consolidator"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: the new test FAILS (six subagents today, and the solver entry is a `CompiledSubAgent`, not a dict)

- [ ] **Step 3: Rewire `agent.py`**

In `src/solvay/agent.py`:

Replace the subagent imports

```python
from solvay.subagents.consolidator import create_consolidator_subagent
from solvay.subagents.parser import create_parser_subagent
from solvay.subagents.peer_reviewer import create_peer_reviewer_subagent
from solvay.subagents.researcher import create_researcher_subagent
from solvay.subagents.solver_loop import create_solver_subagent
from solvay.subagents.verifier import create_verifier_subagent
```

with

```python
from solvay.subagents.consolidator import create_consolidator_subagent
from solvay.subagents.parser import create_parser_subagent
from solvay.subagents.researcher import create_researcher_subagent
from solvay.subagents.solver import create_solver_subagent
```

In `create_solvay_agent`, replace the factory calls and roster

```python
    parser = create_parser_subagent(config)
    researcher = create_researcher_subagent(config, web_search, url_fetch)
    solver = create_solver_subagent(config, web_search, url_fetch)
    verifier = create_verifier_subagent(config)
    peer_reviewer = create_peer_reviewer_subagent(config, web_search)
    consolidator = create_consolidator_subagent(config)
```

with

```python
    parser = create_parser_subagent(config)
    researcher = create_researcher_subagent(config, web_search, url_fetch)
    solver = create_solver_subagent(config, web_search, url_fetch)
    consolidator = create_consolidator_subagent(config)
```

and the `subagents=` list with

```python
        subagents=[
            cast(SubAgent, parser),
            cast(SubAgent, researcher),
            cast(SubAgent, solver),
            cast(SubAgent, consolidator),
        ],
```

- [ ] **Step 4: Update the orchestrator prompt**

In `src/solvay/prompts/orchestrator.md`, replace step 3 and step 4 (the block from `3. **Solve**` through the end of step 4) with:

```markdown
3. **Solve** -- call `task(name="solver", ...)` with the ProblemSpec and
   ResearchBrief. The solver iterates internally with a review tool and
   returns a `SolverReport`:
   - `solver_blocked`: bool, `blocked_topic`: str | null
   - `draft`: SolutionDraft (method + steps + final answer), null only if blocked
   - `termination_reason`: "consensus" | "budget_exhausted" | "judge_forced"
   - `iterations_consumed`: int
   - `open_issues`: unresolved critic objections (empty on consensus)

4. **Handle results:**
   - If `termination_reason == "consensus"` then present the final answer.
   - If `termination_reason == "budget_exhausted"` then present the
     best-effort draft and list `open_issues`.
   - If `termination_reason == "judge_forced"`:
     a. Call `researcher` again focused on `blocked_topic`.
     b. Append new research to `/workspace/lab_notebook.md`.
     c. Call `solver` one more time.
     d. If the second attempt also returns `judge_forced`, present whatever
        draft exists (or state that no draft was produced) with an
        unresolved-topic note.
```

And in the `## Output format` section, replace the loop-summary line with:

```markdown
- **Loop summary:** termination reason, iterations consumed, open issues
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py tests/test_solver.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/solvay/agent.py src/solvay/prompts/orchestrator.md tests/test_agent.py
git commit -m "feat: wire conversational solver, drop critic subagents from orchestrator"
```

---

### Task 6: Trim impossible instructions from critic prompts

**Files:**
- Modify: `src/solvay/prompts/verifier.md`
- Modify: `src/solvay/prompts/peer_reviewer.md`
- Modify: `tests/test_prompts.py`

- [ ] **Step 1: Update the prompt-convention tests**

In `tests/test_prompts.py`, the critics no longer have filesystem tools, so they must NOT reference notebook or harness-notes paths. Replace the whole file body with:

```python
"""Tests for prompt-level native harness conventions."""

from __future__ import annotations

from solvay.subagents import load_prompt


def test_runtime_prompts_reference_native_workspace_notebook() -> None:
    for name in ["orchestrator", "parser", "researcher", "solver", "consolidator"]:
        prompt = load_prompt(name)
        assert "/workspace/lab_notebook.md" in prompt


def test_operational_roles_reference_harness_notes() -> None:
    for name in ["orchestrator", "researcher", "solver"]:
        prompt = load_prompt(name)
        assert "/memories/solvay/harness_notes.md" in prompt
        assert "Symptom:" in prompt
        assert "Suggested fix:" in prompt


def test_critic_prompts_do_not_reference_filesystem_paths() -> None:
    """Critics run without filesystem tools; their prompts must not order
    them to read or write files they cannot touch."""
    for name in ["verifier", "peer_reviewer"]:
        prompt = load_prompt(name)
        assert "/workspace/lab_notebook.md" not in prompt
        assert "/memories/solvay/harness_notes.md" not in prompt
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: `test_critic_prompts_do_not_reference_filesystem_paths` FAILS (paths still present)

- [ ] **Step 3: Trim `verifier.md`**

In `src/solvay/prompts/verifier.md`:
- In `## Input`, delete the line `- /workspace/lab_notebook.md for context`.
- In `## Rules`, delete everything from `- Before acting: read /workspace/lab_notebook.md.` through `Artifacts: ...` and the `- Never include secrets...` bullet, keeping:

```markdown
## Rules

- You are a mechanical checker, not a creative thinker. Stick to the checks.
- Mark severity="blocker" only for dimensional errors, wrong-sign results,
  or order-of-magnitude violations. Minor issues (rounding, style) are "minor".
- All output in English.
```

- [ ] **Step 4: Trim `peer_reviewer.md`**

Same surgery in `src/solvay/prompts/peer_reviewer.md`: delete the notebook line from `## Input` and reduce `## Rules` to:

```markdown
## Rules

- You are a physics expert reviewing methodology, not checking arithmetic.
- Mark severity="blocker" only for fundamentally wrong approaches or
  critical missing terms. Style preferences are "minor".
- All output in English.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/solvay/prompts/verifier.md src/solvay/prompts/peer_reviewer.md tests/test_prompts.py
git commit -m "fix: remove filesystem instructions from critic prompts"
```

---

### Task 7: Delete the subgraph and dead code

**Files:**
- Delete: `src/solvay/subagents/solver_loop.py`
- Delete: `src/solvay/subagents/verifier.py`
- Delete: `src/solvay/subagents/peer_reviewer.py`
- Delete: `tests/test_solver_loop.py`
- Modify: `src/solvay/schemas.py` (remove `SolverResponse`, `CritiqueEntry`, `SolverLoopState`)
- Modify: `tests/test_schemas.py` (remove `TestCritiqueEntry`, `TestSolverLoopState` and their imports)

- [ ] **Step 1: Delete files**

```bash
git rm src/solvay/subagents/solver_loop.py src/solvay/subagents/verifier.py \
       src/solvay/subagents/peer_reviewer.py tests/test_solver_loop.py
```

- [ ] **Step 2: Remove dead schemas**

In `src/solvay/schemas.py` delete the `SolverResponse`, `CritiqueEntry`, and `SolverLoopState` class definitions entirely. (`Verdict`, `SolutionDraft`, `SolverReport`, `JournalEntry*`, `ExecResult`, `DimCheckResult` stay.)

- [ ] **Step 3: Remove their tests**

In `tests/test_schemas.py` delete the `TestCritiqueEntry` and `TestSolverLoopState` classes and remove `CritiqueEntry` and `SolverLoopState` from the import block.

- [ ] **Step 4: Verify nothing references the removed symbols**

Run: `rg -n "SolverResponse|SolverLoopState|CritiqueEntry|solver_loop import|subagents.verifier|subagents.peer_reviewer" src tests`
Expected: only `config.py:84` (`solver_loop: SolverLoopConfig` — that field stays) and benchmark generator files matching on their own local `verifier.py` module (different package, untouched).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: all PASS, no collection errors

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove solver-loop subgraph and orphaned schemas"
```

---

### Task 8: Full verification

**Files:** none new.

- [ ] **Step 1: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src tests`
Expected: clean. Fix any findings (imports ordering, unused imports in modified files) and amend the previous commit if trivial, or commit as `chore: fix lint/type findings`.

- [ ] **Step 2: Full test suite**

Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 3: Live smoke run (requires ANTHROPIC_API_KEY in `.env`)**

Run: `RUN_LIVE_TESTS=1 uv run pytest tests/test_agent_smoke.py -v`
Expected: PASS — the answer contains 4.9 m/s². This validates the dict-subagent tool inheritance assumption (spec section 5, risk 1). If it fails because filesystem tools are missing inside the solver, the fallback is to drop the notebook instructions from `solver.md` (and the solver entries from `test_prompts.py` lists) — note it in the lab notebook and surface it to the user.

- [ ] **Step 4: Manual end-to-end check**

Run: `uv run solvay solve "A 2 kg block slides down a frictionless 30 degree incline. Find its acceleration in m/s^2."`
Expected: final answer ≈ 4.9 m/s², loop summary shows a termination reason and ≥1 iteration consumed.

- [ ] **Step 5: Final commit (if anything changed)**

```bash
git add -A
git commit -m "chore: post-verification fixes for conversational solver"
```
