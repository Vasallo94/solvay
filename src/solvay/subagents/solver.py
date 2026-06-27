"""Conversational solver: a request_review-driven review loop as a drop-in subagent."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from solvay.schemas import Verdict

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
