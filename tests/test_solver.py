"""Tests for the conversational solver subagent and request_review tool."""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import patch

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

    def test_validates_plain_dict_structured_response(self) -> None:
        raw: Runnable[Any, Any] = RunnableLambda(
            lambda _state: {
                "structured_response": {"approved": True, "issues": [], "severity": "none"}
            }
        )
        result = run_critic(raw, "verifier", "prompt")
        assert result == {"approved": True, "issues": [], "severity": "none"}

    def test_degrades_when_structured_response_missing(self) -> None:
        empty: Runnable[Any, Any] = RunnableLambda(lambda _state: {"messages": []})
        result = run_critic(empty, "verifier", "prompt")
        assert result["approved"] is False
        assert result["severity"] == "minor"
        assert "verifier verdict unavailable" in result["issues"][0]


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
        assert payload["peer_reviewer"]["approved"] is True

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
        # timeout=5: on timeout run_critic degrades to approved=False, which
        # breaks consensus and fails the assertion clearly rather than hanging.
        barrier = threading.Barrier(2, timeout=5)

        def synced_critic(_state: Any) -> dict[str, Any]:
            barrier.wait()
            return {"structured_response": _verdict(True)}

        critic = RunnableLambda(synced_critic)
        tool = create_request_review_tool(verifier=critic, reviewer=critic, max_reviews=1)
        payload = json.loads(tool("p", "draft v1"))
        assert payload["directive"] == DIRECTIVE_CONSENSUS


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
