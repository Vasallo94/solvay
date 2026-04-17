"""Solver-critique loop as a LangGraph subgraph wrapped in CompiledSubAgent."""

from __future__ import annotations

import json
from typing import Any

from deepagents import CompiledSubAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from solvay.config import SolvayConfig


class SolverLoopGraphState(BaseModel):
    """State for the solver loop LangGraph subgraph."""

    # Inputs
    problem_spec: dict[str, Any]
    research_brief: dict[str, Any]

    # Loop control
    iteration: int = 0
    max_iterations: int = 3

    # Current round outputs
    current_draft: dict[str, Any] | None = None
    verifier_verdict: dict[str, Any] | None = None
    reviewer_verdict: dict[str, Any] | None = None

    # Accumulated history
    critique_history: list[dict[str, Any]] = Field(default_factory=list)

    # Blocking signal
    solver_blocked: bool = False
    blocked_topic: str | None = None

    # Final output
    final_draft: dict[str, Any] | None = None
    termination_reason: str | None = None
    unresolved_blockers: bool = False


def build_solver_loop_graph(
    solver_model: BaseChatModel,
    verifier_model: BaseChatModel,
    reviewer_model: BaseChatModel,
) -> Any:
    """Build and compile the solver loop LangGraph.

    Args:
        solver_model: LLM for the solver node.
        verifier_model: LLM for the verifier node.
        reviewer_model: LLM for the peer reviewer node.

    Returns:
        A compiled LangGraph that can be invoked with SolverLoopGraphState fields.
    """

    def solve(state: SolverLoopGraphState) -> dict[str, Any]:
        """Invoke solver LLM to produce a SolutionDraft or signal blocked."""
        iteration = state.iteration + 1

        prompt_parts = [
            f"Problem: {json.dumps(state.problem_spec)}",
            f"Research: {json.dumps(state.research_brief)}",
        ]

        if state.critique_history:
            last_critique = state.critique_history[-1]
            prompt_parts.append(
                f"Previous critique (address these issues): {json.dumps(last_critique)}"
            )

        prompt = "\n\n".join(prompt_parts)
        response = solver_model.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}

        if parsed.get("solver_blocked"):
            return {
                "iteration": iteration,
                "solver_blocked": True,
                "blocked_topic": parsed.get("blocked_topic"),
            }

        return {
            "iteration": iteration,
            "current_draft": parsed,
            "solver_blocked": False,
        }

    def critique(state: SolverLoopGraphState) -> dict[str, Any]:
        """Invoke verifier and reviewer on the current draft."""
        # Skip critique if solver signaled blocked (no draft to review)
        if state.solver_blocked:
            return {}

        draft = state.current_draft or {}
        problem = state.problem_spec

        critique_prompt = (
            f"Problem: {json.dumps(problem)}\n\nSolution draft to review: {json.dumps(draft)}"
        )

        verifier_resp = verifier_model.invoke([HumanMessage(content=critique_prompt)])
        reviewer_resp = reviewer_model.invoke([HumanMessage(content=critique_prompt)])

        v_content = (
            verifier_resp.content
            if isinstance(verifier_resp.content, str)
            else str(verifier_resp.content)
        )
        r_content = (
            reviewer_resp.content
            if isinstance(reviewer_resp.content, str)
            else str(reviewer_resp.content)
        )

        try:
            verifier_verdict = json.loads(v_content)
        except json.JSONDecodeError:
            verifier_verdict = {
                "approved": False,
                "issues": ["Failed to parse verifier response"],
                "severity": "blocker",
            }

        try:
            reviewer_verdict = json.loads(r_content)
        except json.JSONDecodeError:
            reviewer_verdict = {
                "approved": False,
                "issues": ["Failed to parse reviewer response"],
                "severity": "blocker",
            }

        new_entry = {
            "iteration": state.iteration,
            "verifier": verifier_verdict,
            "reviewer": reviewer_verdict,
        }
        updated_history = [*state.critique_history, new_entry]

        return {
            "verifier_verdict": verifier_verdict,
            "reviewer_verdict": reviewer_verdict,
            "critique_history": updated_history,
        }

    def judge(state: SolverLoopGraphState) -> dict[str, Any]:
        """Deterministic judge: decide whether to continue, stop, or force re-research."""
        verifier = state.verifier_verdict or {}
        reviewer = state.reviewer_verdict or {}
        iteration = state.iteration
        max_iterations = state.max_iterations

        if state.solver_blocked:
            return {
                "final_draft": state.current_draft,
                "termination_reason": "judge_forced",
            }

        if verifier.get("approved") and reviewer.get("approved"):
            return {
                "final_draft": state.current_draft,
                "termination_reason": "consensus",
            }

        if iteration >= max_iterations:
            has_blockers = (
                verifier.get("severity") == "blocker" or reviewer.get("severity") == "blocker"
            )
            return {
                "final_draft": state.current_draft,
                "termination_reason": "budget_exhausted",
                "unresolved_blockers": has_blockers,
            }

        return {}

    def route_after_judge(state: SolverLoopGraphState) -> str:
        """Route after judge: END if terminated, back to solve otherwise."""
        if state.termination_reason is not None:
            return END
        return "solve"

    builder = StateGraph(SolverLoopGraphState)

    builder.add_node("solve", solve)
    builder.add_node("critique", critique)
    builder.add_node("judge", judge)

    builder.add_edge(START, "solve")
    builder.add_edge("solve", "critique")
    builder.add_edge("critique", "judge")
    builder.add_conditional_edges("judge", route_after_judge, ["solve", END])

    return builder.compile()


def create_solver_subagent(config: SolvayConfig) -> CompiledSubAgent:
    """Create the solver CompiledSubAgent wrapping the loop subgraph.

    Args:
        config: Solvay configuration.

    Returns:
        A CompiledSubAgent with the solver loop graph as its runnable.
    """
    from langchain.chat_models import init_chat_model

    solver_model = init_chat_model(config.model_for("solver"))
    verifier_model = init_chat_model(config.model_for("verifier"))
    reviewer_model = init_chat_model(config.model_for("peer_reviewer"))

    graph = build_solver_loop_graph(
        solver_model=solver_model,
        verifier_model=verifier_model,
        reviewer_model=reviewer_model,
    )

    return CompiledSubAgent(
        name="solver",
        description=(
            "Solve a physics problem through an iterative solve-critique loop. "
            "Accepts ProblemSpec and ResearchBrief, returns SolutionDraft with "
            "termination reason and critique history."
        ),
        runnable=graph,
    )
