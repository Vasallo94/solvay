"""Generate a Quarkdown (.qd) document from a completed RunCollector."""

from __future__ import annotations

import datetime
import re

from solvay.streaming import RunCollector, SubagentRun

_SECTION_TITLES: dict[str, str] = {
    "parser": "Problem Parsing",
    "researcher": "Research",
    "solver": "Solution",
    "verifier": "Verification",
    "peer_reviewer": "Peer Review",
}


def generate_quarkdown(collector: RunCollector) -> str:
    """Render the full Quarkdown report from accumulated pipeline state."""
    parts: list[str] = []

    domain = _domain_from_collector(collector)
    date_str = datetime.date.today().isoformat()
    duration_str = _fmt_duration(collector.total_s)
    title = _title_from_problem(collector.problem)

    parts.append(
        f""".docname {{Solvay Solution Report}}
.doctype {{plain}}
.doclang {{English}}
.theme {{paperwhite}} layout:{{latex}}
.numbering
    - headings: 1

# {title}

**Domain:** {domain} | **Model:** {collector.model} | **Date:** {date_str} | **Duration:** {duration_str}

---

## Problem Statement

{collector.problem}

---

"""
    )

    for run in collector.subagent_runs:
        if run.name == "consolidator":
            continue
        parts.append(_render_subagent_section(run))

    # Final answer — written by consolidator in Quarkdown format
    parts.append("## Final Answer\n\n")
    parts.append(_fix_quarkdown_math(collector.final_answer) if collector.final_answer else "*No answer recorded.*")
    parts.append("\n\n---\n\n")

    # Run metadata
    parts.append("## Run Metadata\n\n")
    parts.append("| Subagent | Duration |\n|----------|----------|\n")
    for run in collector.subagent_runs:
        parts.append(f"| {run.name} | {_fmt_duration(run.duration_s)} |\n")
    parts.append(f"| **Total** | **{_fmt_duration(collector.total_s)}** |\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_subagent_section(run: SubagentRun) -> str:
    title = _SECTION_TITLES.get(run.name, run.name.replace("_", " ").title())
    lines = [f"## {title}\n\n*{run.name} — {_fmt_duration(run.duration_s)}*\n\n"]

    for schema_type, data in run.schemas:
        if schema_type == "ProblemSpec":
            lines.append(_render_problem_spec(data))
        elif schema_type == "ResearchBrief":
            lines.append(_render_research_brief(data))
        elif schema_type == "SolutionDraft":
            lines.append(_render_solution_draft(data))
        elif schema_type == "Verdict":
            lines.append(_render_verdict(data))

    if run.tool_calls:
        tool_list = ", ".join(f"`{tc.tool}`" for tc in run.tool_calls)
        lines.append(f"**Tool calls:** {tool_list}\n\n")

    lines.append("---\n\n")
    return "".join(lines)


def _render_problem_spec(data: dict) -> str:
    domain = data.get("domain", "unknown")
    knowns = data.get("knowns", {})
    unknowns = data.get("unknowns", [])
    assumptions = data.get("assumptions", [])

    knowns_str = (
        ", ".join(
            (f"{k} = {v.get('value', '?')} {v.get('unit', '')}".strip() if isinstance(v, dict) else f"{k} = {v}")
            for k, v in knowns.items()
        )
        or "—"
    )
    unknowns_str = ", ".join(str(u) for u in unknowns) or "—"

    lines = [
        f"| Field | Value |\n|-------|-------|\n",
        f"| Domain | {domain} |\n",
        f"| Knowns | {knowns_str} |\n",
        f"| Unknowns | {unknowns_str} |\n\n",
    ]

    if assumptions:
        lines.append(".box {Assumptions} type:{tip}\n")
        for a in assumptions:
            lines.append(f"    - {a}\n")
        lines.append("\n")

    return "".join(lines)


def _render_research_brief(data: dict) -> str:
    principles = data.get("principles", [])
    equations = data.get("candidate_equations", [])
    citations = data.get("citations", [])
    lines: list[str] = []

    if principles:
        lines.append("**Principles:**\n")
        for p in principles:
            lines.append(f"- {p}\n")
        lines.append("\n")

    if equations:
        lines.append("**Candidate equations:**\n")
        for eq in equations:
            if not eq.strip().startswith("$"):
                eq = f"$ {eq} $"
            lines.append(f"- {eq}\n")
        lines.append("\n")

    if citations:
        lines.append(f"**Sources:** {', '.join(citations)}\n\n")

    return "".join(lines)


def _render_solution_draft(data: dict) -> str:
    method = data.get("method", "")
    steps = data.get("steps", [])
    final_answer = data.get("final_answer", "")
    code_trace = data.get("code_trace", [])
    lines: list[str] = []

    if method:
        lines.append(f"**Method:** {method}\n\n")
    if steps:
        lines.append("**Steps:**\n")
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}\n")
        lines.append("\n")
    if final_answer:
        fa = final_answer
        if isinstance(fa, dict):
            fa = f"{fa.get('value', '?')} {fa.get('unit', '')}".strip()
        lines.append(f"**Final answer:** {fa}\n\n")
    if code_trace:
        lines.append("**Code trace:**\n```python\n")
        for line in code_trace:
            lines.append(f"{line}\n")
        lines.append("```\n\n")

    return "".join(lines)


def _render_verdict(data: dict) -> str:
    approved = data.get("approved", False)
    issues = data.get("issues", [])
    severity = data.get("severity", "none")
    status = "approved ✓" if approved else f"rejected — severity: {severity}"
    lines = [f".box type:{{tip}}\n    **Verification:** {status}\n"]
    for issue in issues:
        lines.append(f"    - {issue}\n")
    lines.append("\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fix_quarkdown_math(text: str) -> str:
    """Convert LaTeX math delimiters to valid Quarkdown syntax.

    Quarkdown math delimiters:
      - $ expr $  — single expression (inline or display), spaces required
      - $$$       — multiline block fence (triple dollar, not double)

    LLMs emit standard LaTeX: $expr$ (inline) and $$....$$ (display/block).
    Processing order: longest match first to avoid partial replacements.
    """
    # 1. Multiline LaTeX blocks: $$ alone on a line, content lines, $$ alone.
    #    → Quarkdown $$$ fenced block.
    text = re.sub(
        r'^\$\$\s*\n(.*?)\n\s*\$\$$',
        lambda m: '$$$\n' + m.group(1) + '\n$$$',
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    # 2. Single-line LaTeX display: $$ expr $$ → $ expr $
    text = re.sub(
        r'(?<!\$)\$\$\s*([^$\n]+?)\s*\$\$(?!\$)',
        lambda m: '$ ' + m.group(1).strip() + ' $',
        text,
    )

    # 3. Inline LaTeX: $expr$ (no surrounding spaces) → $ expr $
    text = re.sub(
        r'(?<!\$)\$([^$\n]+?)\$(?!\$)',
        lambda m: '$ ' + m.group(1).strip() + ' $',
        text,
    )

    # 4. .box {Title} type:{X} → .box type:{X} \n    **Title**
    #    The LLM writes positional title arg which Quarkdown rejects (body is required).
    def _fix_box(m: re.Match) -> str:
        title = m.group(1).strip()
        box_type = m.group(2).strip()
        return f'.box type:{{{box_type}}}\n    **{title}**'

    text = re.sub(
        r'\.box\s+\{([^}]+)\}\s+type:\{([^}]+)\}',
        _fix_box,
        text,
    )

    return text


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f} min"


def _title_from_problem(problem: str) -> str:
    first = problem.split(".")[0].split("?")[0].strip()
    return first[:120]


def _domain_from_collector(collector: RunCollector) -> str:
    for run in collector.subagent_runs:
        for schema_type, data in run.schemas:
            if schema_type == "ProblemSpec":
                return str(data.get("domain", "unknown"))
    return "unknown"
