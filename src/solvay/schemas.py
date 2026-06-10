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


class SolverReport(BaseModel):
    """Final report from the conversational solver subagent."""

    solver_blocked: bool = False
    blocked_topic: str | None = None
    draft: SolutionDraft | None = None
    termination_reason: Literal["consensus", "budget_exhausted", "judge_forced"]
    iterations_consumed: int
    open_issues: list[str] = Field(default_factory=list)


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


class JournalEntryList(BaseModel):
    """Wrapper around ``list[JournalEntry]`` used as the consolidator's
    ``response_format``. langchain rejects bare ``GenericAlias`` schemas, so
    the list must be nested inside a Pydantic model."""

    entries: list[JournalEntry] = Field(default_factory=list)


