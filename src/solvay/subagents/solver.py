"""Conversational solver subagent: review loop driven by the request_review tool."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from solvay.config import SolvayConfig
from solvay.schemas import SolverReport, Verdict
from solvay.subagents import load_prompt
from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec


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
    lock = threading.Lock()
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


def _build_critics(
    config: SolvayConfig,
    web_search_tool: Any,
) -> tuple[Runnable[Any, Any], Runnable[Any, Any]]:
    """Build the verifier and peer-reviewer critic agents."""
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    verifier: Runnable[Any, Any] = create_agent(
        init_chat_model(config.model_for("verifier")),
        system_prompt=load_prompt("verifier"),
        tools=[python_exec, check_dimensions],
        response_format=Verdict,
        name="verifier",
    )
    reviewer: Runnable[Any, Any] = create_agent(
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
