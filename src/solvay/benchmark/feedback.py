"""AFP friction detectors for the Solvay benchmark grading pipeline.

Runs post-matrix analysis on RunRecords to surface grading anomalies
as AFP field reports (drafts for human review).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from afp.enums import Confidence, FaultDomain, FrictionType, Reproducibility, Severity
from afp.models import FieldReport
from afp.sinks.draft import DraftSink

from solvay.benchmark.jsonl import RunRecord
from solvay.benchmark.schema import Problem

_SUBJECT_BASE = "pkg:github/Vasallo94/solvay@0.1.0"


def _subject_uri(problem_id: str) -> str:
    return f"{_SUBJECT_BASE}#benchmark/problems/{problem_id}"


def detect_grading_fallback(
    records: list[RunRecord],
    problems: dict[str, Problem],
) -> list[FieldReport]:
    """Detect when SymPy grading failed and fell back to LLM-judge.

    This signals parsing issues in the grader (e.g. SymPy namespace
    collisions, unparseable LaTeX, or broken expected values).
    """
    reports: list[FieldReport] = []
    seen: set[str] = set()

    for rec in records:
        if rec.grading_method != "llm-judge" or rec.error:
            continue
        if rec.problem_id in seen:
            continue
        seen.add(rec.problem_id)

        problem = problems.get(rec.problem_id)
        expected_value = problem.expected.value if problem else "unknown"

        reports.append(FieldReport.create(
            subject_uri=_subject_uri(rec.problem_id),
            goal="Grade model answer using SymPy symbolic/numeric equivalence",
            expectation="sympy_grade returns True or False (definitive result)",
            observed=(
                f"sympy_grade returned None for problem '{rec.problem_id}', "
                f"falling back to LLM-judge. Expected value: '{expected_value}'"
            ),
            friction_type=FrictionType.BUG,
            fault_domain=FaultDomain.TOOL,
            severity=Severity.DEGRADED,
            confidence=Confidence.HIGH,
            reproducibility=Reproducibility.DETERMINISTIC,
            dedupe_key=f"grading-fallback:{rec.problem_id}",
            tool_call_name="sympy_grade",
            evidence=[
                f"problem_id: {rec.problem_id}",
                f"expected.value: {expected_value}",
                f"expected.kind: {problem.expected.kind if problem else 'unknown'}",
                f"profile: {rec.profile}",
                f"grading_method: {rec.grading_method}",
            ],
        ))

    return reports


def detect_inconsistent_grading(
    records: list[RunRecord],
) -> list[FieldReport]:
    """Detect when the same answer gets different grades across profiles.

    Groups records by problem_id, extracts the effective answer per profile,
    and flags when identical answers receive conflicting correct/incorrect
    verdicts — a strong signal of a grading bug.
    """
    by_problem: dict[str, list[RunRecord]] = defaultdict(list)
    for rec in records:
        if not rec.error:
            by_problem[rec.problem_id].append(rec)

    reports: list[FieldReport] = []

    for problem_id, recs in by_problem.items():
        answer_grades: dict[str, set[bool]] = defaultdict(set)
        answer_profiles: dict[str, list[str]] = defaultdict(list)

        for rec in recs:
            answer_key = (rec.answer_extracted or rec.answer_raw).strip()[:200]
            answer_grades[answer_key].add(rec.correct)
            answer_profiles[answer_key].append(f"{rec.profile}={rec.correct}")

        for answer_key, grades in answer_grades.items():
            if len(grades) <= 1:
                continue
            profiles_detail = answer_profiles[answer_key]
            reports.append(FieldReport.create(
                subject_uri=_subject_uri(problem_id),
                goal="Grade model answers consistently across profiles",
                expectation="Identical answers receive identical grades",
                observed=(
                    f"Same answer for '{problem_id}' graded differently: "
                    f"{', '.join(profiles_detail)}"
                ),
                friction_type=FrictionType.WRONG_OUTPUT,
                fault_domain=FaultDomain.TOOL,
                severity=Severity.BLOCKED,
                confidence=Confidence.HIGH,
                reproducibility=Reproducibility.DETERMINISTIC,
                dedupe_key=f"inconsistent-grading:{problem_id}",
                tool_call_name="evaluate_correct",
                evidence=[
                    f"problem_id: {problem_id}",
                    f"answer (first 200 chars): {answer_key}",
                    *[f"  {p}" for p in profiles_detail],
                ],
            ))

    return reports


def detect_unanimous_disagreement(
    records: list[RunRecord],
    problems: dict[str, Problem],
) -> list[FieldReport]:
    """Detect when ALL profiles agree on an answer but are all marked wrong.

    When every profile for a given problem is marked incorrect, the expected
    value itself may be wrong — especially if the models show consensus.
    """
    by_problem: dict[str, list[RunRecord]] = defaultdict(list)
    for rec in records:
        if not rec.error:
            by_problem[rec.problem_id].append(rec)

    reports: list[FieldReport] = []

    for problem_id, recs in by_problem.items():
        if len(recs) < 2:
            continue
        if any(rec.correct for rec in recs):
            continue

        problem = problems.get(problem_id)
        expected_value = problem.expected.value if problem else "unknown"

        reports.append(FieldReport.create(
            subject_uri=_subject_uri(problem_id),
            goal="Verify benchmark expected values against model consensus",
            expectation="At least one profile solves the problem correctly",
            observed=(
                f"All {len(recs)} profiles marked incorrect for '{problem_id}'. "
                f"Expected: '{expected_value}'. "
                f"This may indicate a wrong expected value."
            ),
            friction_type=FrictionType.WRONG_OUTPUT,
            fault_domain=FaultDomain.AMBIGUOUS_CONTRACT,
            severity=Severity.DEGRADED,
            confidence=Confidence.MEDIUM,
            reproducibility=Reproducibility.DETERMINISTIC,
            dedupe_key=f"unanimous-disagreement:{problem_id}",
            tool_call_name="evaluate_correct",
            evidence=[
                f"problem_id: {problem_id}",
                f"expected: {expected_value}",
                f"profiles tested: {len(recs)}",
                *[f"  {r.profile}: correct={r.correct}, method={r.grading_method}"
                  for r in recs],
            ],
        ))

    return reports


def analyze_run(
    records: list[RunRecord],
    problems: dict[str, Problem],
) -> list[FieldReport]:
    """Run all friction detectors and return deduplicated reports."""
    all_reports: list[FieldReport] = []
    all_reports.extend(detect_grading_fallback(records, problems))
    all_reports.extend(detect_inconsistent_grading(records))
    all_reports.extend(detect_unanimous_disagreement(records, problems))

    seen_keys: set[str] = set()
    deduped: list[FieldReport] = []
    for report in all_reports:
        key = report.dedupe_key or report.report_id
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(report)
    return deduped


def submit_feedback(
    reports: list[FieldReport],
    project_dir: Path,
) -> int:
    """Submit reports to the AFP draft sink and print review signal.

    Returns the number of reports submitted.
    """
    if not reports:
        return 0

    sink = DraftSink(base_dir=project_dir)
    for report in reports:
        sink.submit(report.to_dict())

    print(
        f"AFP-REVIEW: {len(reports)} friction report(s) pending "
        f"human review -> afp drafts list --dir {project_dir}",
        file=sys.stderr,
    )
    return len(reports)
