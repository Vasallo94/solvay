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
        severity=severity,
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
