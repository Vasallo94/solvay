"""Conversational solver: a request_review-driven review loop as a drop-in subagent."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from deepagents import CompiledSubAgent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from solvay.config import SolvayConfig, resolve_model
from solvay.schemas import SolutionDraft, SolverReport, Verdict
from solvay.subagents import load_prompt
from solvay.tools.dimensional import check_dimensions
from solvay.tools.physics_checklist import physics_checklist
from solvay.tools.python_exec import python_exec

DIRECTIVE_CONSENSUS = "consensus -- finalize now"
DIRECTIVE_BUDGET_EXHAUSTED = "budget exhausted -- finalize with best effort"
DIRECTIVE_ITERATE = "address the issues and request review again"


def run_critic(critic: Runnable[Any, Any], role: str, prompt: str) -> dict[str, Any]:
    """Invoke a critic agent and return its Verdict as a dict.

    Retries once on any failure. If both attempts fail, returns a degraded
    non-blocking verdict so a critic outage never poisons the review loop.
    """
    last_error = "no structured verdict returned"
    for _attempt in range(2):
        try:
            result = critic.invoke({"messages": [HumanMessage(content=prompt)]})
        except Exception as exc:  # degrade, never crash the loop
            last_error = str(exc)
            continue
        structured = result.get("structured_response") if isinstance(result, dict) else None
        if structured is None:
            continue
        try:
            if isinstance(structured, Verdict):
                verdict = structured
            else:
                verdict = Verdict.model_validate(structured)
        except ValueError as exc:
            last_error = str(exc)
            continue
        return verdict.model_dump()
    return Verdict(
        approved=False,
        issues=[f"{role} verdict unavailable: {last_error}"],
        severity="minor",
    ).model_dump()


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
        return json.dumps(
            {
                "directive": directive,
                "reviews_remaining": remaining,
                "verifier": verifier_verdict,
                "peer_reviewer": reviewer_verdict,
            }
        )

    return request_review, state


def _build_solver_components(
    config: SolvayConfig,
    web_search_tool: Any,
    url_fetch_tool: Any,
) -> tuple[Runnable[Any, Any], dict[str, int]]:
    """Build the conversational solver agent and expose the review counter.

    Unlike the graph-based solver (``solver_loop.py``), this agent uses tools
    and structured output for ALL models, including local ones; degradation on
    Ollama is handled by ``extract_report``'s no_review fallback and ``_run``'s
    invoke guard rather than by stripping tools. Models are resolved with
    ``config.model_kwargs`` (e.g. ``num_predict`` for Ollama) and each agent gets
    its own ``ModelCallLimitMiddleware`` so the call budgets are not shared.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware

    mkwargs = config.model_kwargs

    verifier = create_agent(
        resolve_model(config.model_for("verifier"), **mkwargs),
        system_prompt=load_prompt("verifier"),
        tools=[python_exec, check_dimensions],
        response_format=Verdict,
        middleware=[ModelCallLimitMiddleware(run_limit=25)],
        name="verifier",
    )
    reviewer = create_agent(
        resolve_model(config.model_for("peer_reviewer"), **mkwargs),
        system_prompt=load_prompt("peer_reviewer"),
        tools=[python_exec, web_search_tool, physics_checklist],
        response_format=Verdict,
        middleware=[ModelCallLimitMiddleware(run_limit=25)],
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
        middleware=[ModelCallLimitMiddleware(run_limit=25)],
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
        try:
            report = (
                structured
                if isinstance(structured, SolverReport)
                else SolverReport.model_validate(structured)
            )
            return report.model_copy(update={"iterations_consumed": used})
        except Exception:  # malformed structured output -> fall through to no_review
            pass
    text = ""
    if isinstance(result, dict) and result.get("messages"):
        text = str(getattr(result["messages"][-1], "content", ""))
    draft = SolutionDraft(
        method="(unstructured)",
        steps=[text] if text else [],
        final_answer=text,
        code_trace=[],
    )
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
) -> CompiledSubAgent:
    """Create the conversational solver as a drop-in subagent.

    The returned runnable invokes the conversational agent and normalizes its
    output to the JSON payload the orchestrator's Step 4 already reads
    (``final_draft`` + ``termination_reason`` + ``iterations_consumed`` ...).

    Returns a ``CompiledSubAgent`` so deepagents accepts it in the subagents
    list; subscripting (``sub["name"]``, ``sub["runnable"]``) still works
    because ``CompiledSubAgent`` is a TypedDict-like mapping.
    """
    solver_agent, state = _build_solver_components(config, web_search_tool, url_fetch_tool)

    def _run(inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            result = solver_agent.invoke(inputs)
        except Exception as exc:  # degrade to no_review rather than crash the orchestrator
            result = {"messages": [AIMessage(content=f"solver invocation failed: {exc}")]}
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

    return CompiledSubAgent(
        name="solver",
        description=(
            "Solve a physics problem through a conversational solve-review loop. "
            "Accepts ProblemSpec and ResearchBrief, returns a draft with "
            "termination reason and open issues."
        ),
        runnable=RunnableLambda(_run),
    )
