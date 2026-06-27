"""Tests for the conversational solver and request_review tool."""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage
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

    def test_malformed_structured_response_falls_back_to_no_review(self) -> None:
        from solvay.subagents.solver import extract_report

        result = {"structured_response": {"bad": "dict"}}
        out = extract_report(result, used=1)
        assert out.termination_reason == "no_review"


class TestSolverPayload:
    def test_wrapper_emits_orchestrator_payload(self) -> None:
        import json

        from solvay.config import SolvayConfig
        from solvay.schemas import SolutionDraft, SolverReport
        from solvay.subagents import solver as solver_mod

        draft = SolutionDraft(
            method="Newton",
            steps=["a=g sin"],
            final_answer="4.9 m/s^2",
            code_trace=[],
        )
        report = SolverReport(draft=draft, termination_reason="consensus", iterations_consumed=1)
        fake_agent = RunnableLambda(lambda _s: {"structured_response": report})

        with patch.object(
            solver_mod, "_build_solver_components", return_value=(fake_agent, {"used": 1})
        ):
            sub = solver_mod.create_solver_subagent(SolvayConfig(), lambda q: {}, lambda u: "")
        payload = json.loads(sub["runnable"].invoke({"messages": []})["messages"][-1].content)
        assert payload["termination_reason"] == "consensus"
        assert payload["iterations_consumed"] == 1
        assert payload["final_draft"]["method"] == "Newton"

    def test_factory_returns_dict_named_solver(self) -> None:
        from solvay.config import SolvayConfig
        from solvay.subagents import solver as solver_mod

        fake_agent = RunnableLambda(lambda _s: {"messages": [AIMessage(content="x")]})
        with patch.object(
            solver_mod, "_build_solver_components", return_value=(fake_agent, {"used": 0})
        ):
            sub = solver_mod.create_solver_subagent(SolvayConfig(), lambda q: {}, lambda u: "")
        assert sub["name"] == "solver"
        assert "runnable" in sub

    def test_wrapper_degrades_when_agent_raises(self) -> None:
        import json

        from solvay.config import SolvayConfig
        from solvay.subagents import solver as solver_mod

        fake_agent = RunnableLambda(lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
        with patch.object(
            solver_mod, "_build_solver_components", return_value=(fake_agent, {"used": 0})
        ):
            sub = solver_mod.create_solver_subagent(SolvayConfig(), lambda q: {}, lambda u: "")
        payload = json.loads(sub["runnable"].invoke({"messages": []})["messages"][-1].content)
        assert payload["termination_reason"] == "no_review"
