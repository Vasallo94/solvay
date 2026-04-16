"""Pydantic contracts for all inter-agent deliverables."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Quantity(BaseModel):
    """A physical quantity with optional units."""

    value: float | str
    unit: str | None = None


class ProblemSpec(BaseModel):
    """Parsed physics problem specification."""

    statement: str
    domain: Literal["mechanics", "em", "quantum", "thermo", "astro", "other"]
    knowns: dict[str, Quantity]
    unknowns: list[str]
    assumptions: list[str]
    approach_hints: list[str]


class ResearchBrief(BaseModel):
    """Research output: principles, equations, analogies, citations."""

    principles: list[str]
    candidate_equations: list[str]
    analogies: list[str]
    citations: list[str]


class SolutionDraft(BaseModel):
    """Solver output: method, steps, final answer, code trace."""

    method: str
    steps: list[str]
    final_answer: Quantity | str
    code_trace: list[str]


class Verdict(BaseModel):
    """Verifier or peer-reviewer judgment on a solution draft."""

    approved: bool
    issues: list[str]
    severity: Literal["blocker", "minor", "none"]


class ExecResult(BaseModel):
    """Result from the python_exec tool."""

    stdout: str
    stderr: str
    last_expr_repr: str | None = None
    last_expr_latex: str | None = None
    wall_time_ms: int
    artifacts: list[str] = Field(default_factory=list)


class DimCheckResult(BaseModel):
    """Result from the check_dimensions tool."""

    ok: bool
    actual_unit: str
    simplified: str
    notes: str | None = None


class JournalEntry(BaseModel):
    """A single entry extracted by the consolidator for the persistent journal."""

    role: str
    iteration: int
    content: str


class SolverLoopState(BaseModel):
    """State for the solver-critique LangGraph subgraph."""

    # Inputs (set once at entry)
    problem_spec: ProblemSpec
    research_brief: ResearchBrief

    # Loop state
    iteration: int = 0
    max_iterations: int = 3
    current_draft: SolutionDraft | None = None
    verifier_verdict: Verdict | None = None
    reviewer_verdict: Verdict | None = None
    critique_history: list[dict] = Field(default_factory=list)  # type: ignore[type-arg]

    # Blocking signal
    solver_blocked: bool = False
    blocked_topic: str | None = None

    # Output
    final_draft: SolutionDraft | None = None
    termination_reason: Literal[
        "consensus", "budget_exhausted", "judge_forced"
    ] | None = None
    unresolved_blockers: bool = False
