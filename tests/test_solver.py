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
            "approved": True,
            "issues": [],
            "severity": "none",
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
            lambda _s: {
                "structured_response": {"approved": True, "issues": [], "severity": "none"}
            }
        )
        assert run_critic(raw, "verifier", "p") == {
            "approved": True,
            "issues": [],
            "severity": "none",
        }
