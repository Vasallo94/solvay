"""Solver-critique loop as a LangGraph subgraph wrapped in CompiledSubAgent."""

from __future__ import annotations

import json
from typing import Any, cast

from deepagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from solvay.config import SolvayConfig
from solvay.schemas import SolutionDraft, SolverResponse, Verdict
from solvay.subagents import load_prompt
from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec


class SolverLoopGraphState(BaseModel):
    """State for the solver loop LangGraph subgraph."""

    # Required by deepagents.CompiledSubAgent. The task tool extracts the final
    # message from this list and returns it to the parent orchestrator.
    messages: list[Any] = Field(default_factory=list)

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
    solver_model: Runnable[Any, Any] | BaseChatModel,
    verifier_model: Runnable[Any, Any] | BaseChatModel,
    reviewer_model: Runnable[Any, Any] | BaseChatModel,
) -> Any:
    """Build and compile the solver loop LangGraph.

    Args:
        solver_model: LLM for the solver node.
        verifier_model: LLM for the verifier node.
        reviewer_model: LLM for the peer reviewer node.

    Returns:
        A compiled LangGraph that can be invoked with SolverLoopGraphState fields.
    """

    def _json_payload(state: SolverLoopGraphState, updates: dict[str, Any]) -> str:
        payload = {
            "final_draft": updates.get("final_draft", state.final_draft),
            "termination_reason": updates.get("termination_reason", state.termination_reason),
            "iterations_consumed": state.iteration,
            "unresolved_blockers": updates.get("unresolved_blockers", state.unresolved_blockers),
            "blocked_topic": state.blocked_topic,
            "critique_history": state.critique_history,
        }
        return json.dumps(payload, default=str)

    def _parse_json_response(content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _invoke_structured(runnable: Runnable[Any, Any] | BaseChatModel, prompt: str) -> Any:
        if isinstance(runnable, BaseChatModel):
            response = runnable.invoke([HumanMessage(content=prompt)])
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            return _parse_json_response(content)

        agent_runnable = cast(Runnable[Any, Any], runnable)
        result = agent_runnable.invoke({"messages": [HumanMessage(content=prompt)]})
        if isinstance(result, dict):
            structured = result.get("structured_response")
            if structured is not None:
                if hasattr(structured, "model_dump"):
                    return structured.model_dump()
                return structured
            messages = result.get("messages")
            if messages:
                last_message = messages[-1]
                return _parse_json_response(getattr(last_message, "content", ""))
        return {}

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
        parsed = _invoke_structured(solver_model, prompt)

        if parsed.get("solver_blocked"):
            return {
                "iteration": iteration,
                "solver_blocked": True,
                "blocked_topic": parsed.get("blocked_topic"),
            }

        try:
            draft = SolutionDraft.model_validate(parsed).model_dump()
        except ValueError:
            draft = parsed

        return {
            "iteration": iteration,
            "current_draft": draft,
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

        verifier_verdict = _invoke_structured(verifier_model, critique_prompt)
        reviewer_verdict = _invoke_structured(reviewer_model, critique_prompt)

        if not isinstance(verifier_verdict, dict) or not verifier_verdict:
            verifier_verdict = {
                "approved": False,
                "issues": ["Failed to parse verifier response"],
                "severity": "blocker",
            }

        if not isinstance(reviewer_verdict, dict) or not reviewer_verdict:
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
            updates: dict[str, Any] = {
                "final_draft": state.current_draft,
                "termination_reason": "judge_forced",
            }
            updates["messages"] = [AIMessage(content=_json_payload(state, updates))]
            return updates

        if verifier.get("approved") and reviewer.get("approved"):
            updates = {
                "final_draft": state.current_draft,
                "termination_reason": "consensus",
            }
            updates["messages"] = [AIMessage(content=_json_payload(state, updates))]
            return updates

        if iteration >= max_iterations:
            has_blockers = (
                verifier.get("severity") == "blocker" or reviewer.get("severity") == "blocker"
            )
            updates = {
                "final_draft": state.current_draft,
                "termination_reason": "budget_exhausted",
                "unresolved_blockers": has_blockers,
            }
            updates["messages"] = [AIMessage(content=_json_payload(state, updates))]
            return updates

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


def create_solver_subagent(
    config: SolvayConfig,
    web_search_tool: Any,
    url_fetch_tool: Any,
) -> CompiledSubAgent:
    """Create the solver CompiledSubAgent wrapping the loop subgraph.

    Args:
        config: Solvay configuration.

    Returns:
        A CompiledSubAgent with the solver loop graph as its runnable.
    """
    from solvay.config import resolve_model

    mkwargs = config.model_kwargs
    solver_model = create_agent(
        resolve_model(config.model_for("solver"), **mkwargs),
        system_prompt=load_prompt("solver"),
        tools=[python_exec, check_dimensions, web_search_tool, url_fetch_tool],
        response_format=SolverResponse,
        name="solver",
    )
    verifier_model = create_agent(
        resolve_model(config.model_for("verifier"), **mkwargs),
        system_prompt=load_prompt("verifier"),
        tools=[python_exec, check_dimensions],
        response_format=Verdict,
        name="verifier",
    )
    reviewer_model = create_agent(
        resolve_model(config.model_for("peer_reviewer"), **mkwargs),
        system_prompt=load_prompt("peer_reviewer"),
        tools=[python_exec, web_search_tool],
        response_format=Verdict,
        name="peer_reviewer",
    )

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
