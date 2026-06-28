"""Tests for the solver-critique loop subgraph."""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from solvay.schemas import (
    ProblemSpec,
    Quantity,
    ResearchBrief,
    SolutionDraft,
    Verdict,
)
from solvay.subagents.solver_loop import SolverLoopGraphState, build_solver_loop_graph


def _make_inputs() -> dict[str, object]:
    """Create minimal inputs for the solver loop."""
    problem_spec = ProblemSpec(
        statement="A 2kg block on a 30-degree frictionless incline",
        domain="mechanics",
        knowns={"m": Quantity(value=2.0, unit="kg")},
        unknowns=["acceleration"],
        assumptions=["frictionless"],
        approach_hints=["Newton's second law"],
    )
    research_brief = ResearchBrief(
        principles=["Newton's second law: F=ma"],
        candidate_equations=[r"a = g \sin\theta"],
        analogies=[],
        citations=["Tipler ch.4"],
    )
    return {
        "problem_spec": problem_spec.model_dump(),
        "research_brief": research_brief.model_dump(),
        "iteration": 0,
        "max_iterations": 3,
        "critique_history": [],
        "solver_blocked": False,
    }


def _draft_json() -> str:
    """Return a valid SolutionDraft JSON string."""
    draft = SolutionDraft(
        method="Newtonian mechanics",
        steps=["Apply F=ma along incline", "a = g*sin(30) = 4.9 m/s^2"],
        final_answer=Quantity(value=4.9, unit="m/s^2"),
        code_trace=["from sympy import sin, pi; 9.81*sin(pi/6)"],
    )
    return json.dumps(draft.model_dump())


def _verdict_json(approved: bool, severity: str = "none") -> str:
    """Return a valid Verdict JSON string."""
    v = Verdict(
        approved=approved,
        issues=[] if approved else ["Issue found"],
        severity=severity,
    )
    return json.dumps(v.model_dump())


class TestSolverLoopConsensus:
    """Solver produces draft, both critics approve: consensus on iter 1."""

    def test_consensus_terminates(self) -> None:
        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(_make_inputs())
        assert result["termination_reason"] == "consensus"
        assert result["final_draft"] is not None
        assert result["iteration"] == 1

    def test_final_message_contains_compiled_subagent_payload(self) -> None:
        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(_make_inputs())
        payload = json.loads(result["messages"][-1].content)
        assert payload["termination_reason"] == "consensus"
        assert payload["iterations_consumed"] == 1
        assert payload["final_draft"]["method"] == "Newtonian mechanics"

    def test_agent_runnables_can_return_structured_responses(self) -> None:
        draft = SolutionDraft.model_validate(json.loads(_draft_json()))
        approved = Verdict.model_validate(json.loads(_verdict_json(True)))

        solver_agent: Runnable[object, object] = RunnableLambda(
            lambda _state: {
                "messages": [AIMessage(content="solver done")],
                "structured_response": draft,
            }
        )
        verifier_agent: Runnable[object, object] = RunnableLambda(
            lambda _state: {
                "messages": [AIMessage(content="verifier done")],
                "structured_response": approved,
            }
        )
        reviewer_agent: Runnable[object, object] = RunnableLambda(
            lambda _state: {
                "messages": [AIMessage(content="reviewer done")],
                "structured_response": approved,
            }
        )

        graph = build_solver_loop_graph(
            solver_model=solver_agent,
            verifier_model=verifier_agent,
            reviewer_model=reviewer_agent,
        )

        result = graph.invoke(_make_inputs())
        assert result["termination_reason"] == "consensus"
        assert result["final_draft"]["method"] == "Newtonian mechanics"


class TestSolverLoopBudgetExhausted:
    """3 iterations without consensus: budget_exhausted."""

    def test_budget_exhausted(self) -> None:
        solver_model = FakeListChatModel(responses=[_draft_json(), _draft_json(), _draft_json()])
        verifier_model = FakeListChatModel(
            responses=[
                _verdict_json(False, "blocker"),
                _verdict_json(False, "blocker"),
                _verdict_json(False, "blocker"),
            ]
        )
        reviewer_model = FakeListChatModel(
            responses=[
                _verdict_json(True),
                _verdict_json(True),
                _verdict_json(True),
            ]
        )

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(_make_inputs())
        assert result["termination_reason"] == "budget_exhausted"
        assert result["unresolved_blockers"] is True
        assert result["iteration"] == 3


class TestSolverLoopJudgeForced:
    """Solver signals blocked: judge_forced."""

    def test_judge_forced(self) -> None:
        blocked_response = json.dumps(
            {
                "solver_blocked": True,
                "blocked_topic": "relativistic corrections",
            }
        )
        solver_model = FakeListChatModel(responses=[blocked_response])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(_make_inputs())
        assert result["termination_reason"] == "judge_forced"
        assert result["blocked_topic"] == "relativistic corrections"


class TestSolverLoopCritiqueHistory:
    """Critique history accumulates across iterations."""

    def test_history_accumulates(self) -> None:
        solver_model = FakeListChatModel(responses=[_draft_json(), _draft_json()])
        verifier_model = FakeListChatModel(
            responses=[
                _verdict_json(False, "minor"),
                _verdict_json(True),
            ]
        )
        reviewer_model = FakeListChatModel(
            responses=[
                _verdict_json(False, "minor"),
                _verdict_json(True),
            ]
        )

        graph = build_solver_loop_graph(
            solver_model=solver_model,
            verifier_model=verifier_model,
            reviewer_model=reviewer_model,
        )

        result = graph.invoke(_make_inputs())
        assert result["termination_reason"] == "consensus"
        assert len(result["critique_history"]) == 2


class TestPrepareNode:
    """Tests for the prepare node that parses description into state fields."""

    def _make_message_only_input(self, description: str) -> dict[str, object]:
        return {"messages": [HumanMessage(content=description)]}

    def test_state_accepts_empty_problem_spec_and_research_brief(self) -> None:
        state = SolverLoopGraphState(messages=[HumanMessage(content="test")])
        assert state.problem_spec == {}
        assert state.research_brief == {}

    def test_prepare_parses_valid_json_blocks(self) -> None:
        spec = '{"statement": "A ball drops", "domain": "mechanics"}'
        brief = '{"principles": ["F=ma"], "candidate_equations": []}'
        description = f"Problem spec:\n{spec}\n\nResearch:\n{brief}"

        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])
        graph = build_solver_loop_graph(
            solver_model=solver_model, verifier_model=verifier_model, reviewer_model=reviewer_model
        )

        result = graph.invoke(self._make_message_only_input(description))
        assert result["problem_spec"]["domain"] == "mechanics"
        assert result["research_brief"]["principles"] == ["F=ma"]
        assert result["termination_reason"] == "consensus"

    def test_prepare_handles_malformed_json(self) -> None:
        description = "Problem spec:\n{not valid json}\n\nResearch:\n{also bad}"
        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])
        graph = build_solver_loop_graph(
            solver_model=solver_model, verifier_model=verifier_model, reviewer_model=reviewer_model
        )

        result = graph.invoke(self._make_message_only_input(description))
        assert result["problem_spec"].get("raw_description") is not None
        assert result["termination_reason"] == "consensus"

    def test_prepare_skips_when_fields_already_populated(self) -> None:
        inputs = _make_inputs()
        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])
        graph = build_solver_loop_graph(
            solver_model=solver_model, verifier_model=verifier_model, reviewer_model=reviewer_model
        )

        result = graph.invoke(inputs)
        assert result["problem_spec"]["domain"] == "mechanics"
        assert result["termination_reason"] == "consensus"

    def test_prepare_handles_empty_messages(self) -> None:
        solver_model = FakeListChatModel(responses=[_draft_json()])
        verifier_model = FakeListChatModel(responses=[_verdict_json(True)])
        reviewer_model = FakeListChatModel(responses=[_verdict_json(True)])
        graph = build_solver_loop_graph(
            solver_model=solver_model, verifier_model=verifier_model, reviewer_model=reviewer_model
        )

        result = graph.invoke({"messages": []})
        assert result.get("problem_spec", {}) == {}
        assert result["termination_reason"] == "consensus"
