"""Tests for Quarkdown report generation."""

from __future__ import annotations

import datetime

from solvay.report import generate_quarkdown
from solvay.streaming import RunCollector, SubagentRun, ToolCallRecord


def _make_collector() -> RunCollector:
    parser_run = SubagentRun(
        name="parser",
        duration_s=8.0,
        tool_calls=[ToolCallRecord(tool="python_exec", args_preview="h = 10")],
        schemas=[
            (
                "ProblemSpec",
                {
                    "statement": "A ball dropped from 10m",
                    "domain": "mechanics",
                    "knowns": {"h": {"value": 10.0, "unit": "m"}},
                    "unknowns": ["v_final"],
                    "assumptions": ["frictionless"],
                    "approach_hints": ["kinematics"],
                },
            )
        ],
    )
    researcher_run = SubagentRun(
        name="researcher",
        duration_s=31.0,
        tool_calls=[ToolCallRecord(tool="web_search", args_preview="kinematic equations")],
        schemas=[
            (
                "ResearchBrief",
                {
                    "principles": ["Conservation of energy"],
                    "candidate_equations": ["v^2 = 2gh"],
                    "analogies": [],
                    "citations": ["HyperPhysics"],
                },
            )
        ],
    )
    consolidator_run = SubagentRun(
        name="consolidator",
        duration_s=9.0,
    )
    return RunCollector(
        problem="A ball dropped from 10m, find final velocity",
        model="anthropic:claude-sonnet-4-6",
        subagent_runs=[parser_run, researcher_run, consolidator_run],
        total_s=96.0,
        final_answer=".box {Answer} type:{tip}\n    v = 14.0 m/s",
    )


class TestGenerateQuarkdown:
    def test_document_header_present(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert ".docname {Solvay Solution Report}" in qd
        assert ".theme {paperwhite} layout:{latex}" in qd

    def test_problem_statement_section(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "## Problem Statement" in qd
        assert "A ball dropped from 10m" in qd

    def test_model_and_domain_in_header(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "claude-sonnet-4-6" in qd
        assert "mechanics" in qd

    def test_parser_section_with_table(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Problem Parsing" in qd
        assert "v_final" in qd
        assert "| Domain" in qd

    def test_assumptions_box(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert ".box {Assumptions} type:{tip}" in qd
        assert "frictionless" in qd

    def test_research_section_with_principles(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Research" in qd
        assert "Conservation of energy" in qd

    def test_final_answer_section(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Final Answer" in qd
        assert "v = 14.0 m/s" in qd

    def test_run_metadata_table(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "Run Metadata" in qd
        assert "| parser" in qd
        assert "| **Total**" in qd

    def test_consolidator_not_rendered_as_pipeline_section(self) -> None:
        qd = generate_quarkdown(_make_collector())
        # Consolidator output goes into Final Answer, not its own numbered section
        assert "· Consolidation" not in qd

    def test_tool_calls_listed(self) -> None:
        qd = generate_quarkdown(_make_collector())
        assert "`python_exec`" in qd
        assert "`web_search`" in qd
