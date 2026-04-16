"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from solvay.schemas import (
    CritiqueEntry,
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
        with pytest.raises(ValidationError):
            ProblemSpec(
                statement="test",
                domain="biology",
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


class TestCritiqueEntry:
    def _draft(self) -> SolutionDraft:
        return SolutionDraft(
            method="Newtonian mechanics",
            steps=["Apply F=ma"],
            final_answer=Quantity(value=4.9, unit="m/s^2"),
            code_trace=[],
        )

    def test_valid_entry_with_both_verdicts(self) -> None:
        entry = CritiqueEntry(
            iteration=1,
            draft=self._draft(),
            verifier_verdict=Verdict(approved=True, issues=[], severity="none"),
            reviewer_verdict=Verdict(
                approved=False, issues=["Check sign"], severity="minor"
            ),
        )
        assert entry.iteration == 1
        assert isinstance(entry.draft, SolutionDraft)
        assert entry.verifier_verdict is not None and entry.verifier_verdict.approved
        assert (
            entry.reviewer_verdict is not None
            and entry.reviewer_verdict.severity == "minor"
        )

    def test_verdicts_default_to_none(self) -> None:
        entry = CritiqueEntry(iteration=0, draft=self._draft())
        assert entry.verifier_verdict is None
        assert entry.reviewer_verdict is None

    def test_missing_required_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CritiqueEntry(iteration=0)  # type: ignore[call-arg]


class TestSolverLoopState:
    def _state(self) -> SolverLoopState:
        return SolverLoopState(
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

    def test_defaults(self) -> None:
        state = self._state()
        assert state.iteration == 0
        assert state.max_iterations == 3
        assert state.current_draft is None
        assert state.termination_reason is None
        assert state.critique_history == []

    def test_critique_history_accepts_critique_entries(self) -> None:
        draft = SolutionDraft(
            method="m",
            steps=["s"],
            final_answer="x",
            code_trace=[],
        )
        entry = CritiqueEntry(
            iteration=1,
            draft=draft,
            verifier_verdict=Verdict(approved=True, issues=[], severity="none"),
            reviewer_verdict=None,
        )
        state = self._state()
        state.critique_history.append(entry)
        assert len(state.critique_history) == 1
        assert isinstance(state.critique_history[0], CritiqueEntry)
        assert state.critique_history[0].draft.method == "m"

    def test_critique_history_coerces_dict_input(self) -> None:
        draft = SolutionDraft(
            method="m", steps=["s"], final_answer="x", code_trace=[]
        )
        state = SolverLoopState(
            problem_spec=ProblemSpec(
                statement="t",
                domain="mechanics",
                knowns={},
                unknowns=[],
                assumptions=[],
                approach_hints=[],
            ),
            research_brief=ResearchBrief(
                principles=[], candidate_equations=[], analogies=[], citations=[]
            ),
            critique_history=[
                {
                    "iteration": 0,
                    "draft": draft,
                    "verifier_verdict": None,
                    "reviewer_verdict": None,
                }
            ],
        )
        assert isinstance(state.critique_history[0], CritiqueEntry)
        assert state.critique_history[0].iteration == 0
