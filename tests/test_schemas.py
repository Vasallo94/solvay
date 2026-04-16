"""Tests for Pydantic schemas."""

import pytest
from solvay.schemas import (
    DimCheckResult,
    ExecResult,
    JournalEntry,
    ProblemSpec,
    Quantity,
    ResearchBrief,
    SolutionDraft,
    SolverLoopState,
    Verdict,
)


class TestQuantity:
    def test_numeric_value(self) -> None:
        q = Quantity(value=9.81, unit="m/s^2")
        assert q.value == 9.81
        assert q.unit == "m/s^2"

    def test_symbolic_value(self) -> None:
        q = Quantity(value="g", unit=None)
        assert q.value == "g"
        assert q.unit is None

    def test_dimensionless(self) -> None:
        q = Quantity(value=3.14, unit=None)
        assert q.unit is None


class TestProblemSpec:
    def test_valid_spec(self) -> None:
        spec = ProblemSpec(
            statement="A block slides down an incline",
            domain="mechanics",
            knowns={"m": Quantity(value=2.0, unit="kg")},
            unknowns=["acceleration"],
            assumptions=["frictionless"],
            approach_hints=["Newton's second law"],
        )
        assert spec.domain == "mechanics"
        assert len(spec.knowns) == 1

    def test_invalid_domain_rejected(self) -> None:
        with pytest.raises(Exception):
            ProblemSpec(
                statement="test",
                domain="biology",  # type: ignore[arg-type]
                knowns={},
                unknowns=[],
                assumptions=[],
                approach_hints=[],
            )


class TestResearchBrief:
    def test_valid_brief(self) -> None:
        brief = ResearchBrief(
            principles=["Newton's second law"],
            candidate_equations=[r"F = ma"],
            analogies=[],
            citations=["Griffiths ch.2"],
        )
        assert len(brief.principles) == 1


class TestSolutionDraft:
    def test_valid_draft_with_quantity(self) -> None:
        draft = SolutionDraft(
            method="Newtonian mechanics",
            steps=["Apply F=ma", "Solve for a"],
            final_answer=Quantity(value=4.9, unit="m/s^2"),
            code_trace=["import sympy"],
        )
        assert isinstance(draft.final_answer, Quantity)

    def test_valid_draft_with_string_answer(self) -> None:
        draft = SolutionDraft(
            method="Symbolic",
            steps=["Derive expression"],
            final_answer="g*sin(theta)",
            code_trace=[],
        )
        assert draft.final_answer == "g*sin(theta)"


class TestVerdict:
    def test_approved_verdict(self) -> None:
        v = Verdict(approved=True, issues=[], severity="none")
        assert v.approved

    def test_blocker_verdict(self) -> None:
        v = Verdict(approved=False, issues=["Wrong units"], severity="blocker")
        assert not v.approved
        assert v.severity == "blocker"


class TestExecResult:
    def test_basic_result(self) -> None:
        r = ExecResult(
            stdout="42",
            stderr="",
            last_expr_repr="42",
            last_expr_latex=None,
            wall_time_ms=15,
            artifacts=[],
        )
        assert r.wall_time_ms == 15


class TestDimCheckResult:
    def test_ok_result(self) -> None:
        r = DimCheckResult(
            ok=True,
            actual_unit="m/s^2",
            simplified="meter/second**2",
            notes=None,
        )
        assert r.ok


class TestJournalEntry:
    def test_valid_entry(self) -> None:
        e = JournalEntry(
            role="solver",
            iteration=2,
            content="Used Lagrangian approach instead of Newtonian.",
        )
        assert e.role == "solver"


class TestSolverLoopState:
    def test_defaults(self) -> None:
        state = SolverLoopState(
            problem_spec=ProblemSpec(
                statement="test",
                domain="mechanics",
                knowns={},
                unknowns=["a"],
                assumptions=[],
                approach_hints=[],
            ),
            research_brief=ResearchBrief(
                principles=[], candidate_equations=[], analogies=[], citations=[]
            ),
        )
        assert state.iteration == 0
        assert state.max_iterations == 3
        assert state.current_draft is None
        assert state.termination_reason is None
