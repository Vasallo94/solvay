"""Tests for the conversational solver and request_review tool."""

from __future__ import annotations

import json
import threading
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from solvay.schemas import Verdict
from solvay.subagents.solver import (
    DIRECTIVE_BUDGET_EXHAUSTED,
    DIRECTIVE_CONSENSUS,
    DIRECTIVE_ITERATE,
    create_request_review_tool,
    run_critic,
)


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


class TestRequestReview:
    def test_consensus_when_both_approve(self) -> None:
        tool, _state = create_request_review_tool(
            _critic(_verdict(True)), _critic(_verdict(True)), 3
        )
        p = json.loads(tool("prob", "draft"))
        assert p["directive"] == DIRECTIVE_CONSENSUS
        assert p["reviews_remaining"] == 2
        assert p["verifier"]["approved"] is True
        assert p["peer_reviewer"]["approved"] is True

    def test_iterate_when_rejected_with_budget(self) -> None:
        tool, _state = create_request_review_tool(
            _critic(_verdict(False, "blocker")), _critic(_verdict(True)), 3
        )
        p = json.loads(tool("prob", "draft"))
        assert p["directive"] == DIRECTIVE_ITERATE
        assert p["reviews_remaining"] == 2

    def test_budget_exhausted_on_last_rejected(self) -> None:
        tool, _state = create_request_review_tool(
            _critic(_verdict(False, "minor")), _critic(_verdict(True)), 1
        )
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
