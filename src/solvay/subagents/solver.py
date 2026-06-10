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
