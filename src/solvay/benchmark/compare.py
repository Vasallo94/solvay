"""Build and format a comparison report from a benchmark run file."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from solvay.benchmark.jsonl import RunHeader, RunRecord


@dataclass
class ComparisonReport:
    header: RunHeader
    profiles: list[str]                             # sorted
    problems: list[str]                             # sorted
    results: dict[str, dict[str, bool | None]]      # {problem_id: {profile: correct}}
    accuracy: dict[str, float]                      # {profile: 0.0–1.0}
    avg_time: dict[str, float]                      # {profile: seconds}
    avg_tokens: dict[str, float]                    # {profile: output token count}


def build_comparison(header: RunHeader, records: list[RunRecord]) -> ComparisonReport:
    """Aggregate *records* into a :class:`ComparisonReport`.

    For each (problem, profile) pair only the **first** matching record is used;
    subsequent repeats are ignored so the table stays square.
    """
    # Discover unique profiles and problems (preserve insertion order, sort later).
    profiles_seen: set[str] = set()
    problems_seen: set[str] = set()

    # {problem_id: {profile: correct}}
    results: dict[str, dict[str, bool | None]] = defaultdict(dict)

    # Accumulators for summary stats.
    time_acc: dict[str, list[float]] = defaultdict(list)
    token_acc: dict[str, list[float]] = defaultdict(list)
    correct_acc: dict[str, list[bool]] = defaultdict(list)

    for rec in records:
        profiles_seen.add(rec.profile)
        problems_seen.add(rec.problem_id)

        # First record for this (problem, profile) wins.
        if rec.profile not in results[rec.problem_id]:
            results[rec.problem_id][rec.profile] = rec.correct
            time_acc[rec.profile].append(rec.elapsed_seconds)
            token_acc[rec.profile].append(float(rec.tokens.get("output", 0)))
            correct_acc[rec.profile].append(rec.correct)

    profiles = sorted(profiles_seen)
    problems = sorted(problems_seen)

    # Fill missing (problem, profile) slots with None.
    for pid in problems:
        for prof in profiles:
            results[pid].setdefault(prof, None)

    accuracy = {
        prof: (sum(correct_acc[prof]) / len(correct_acc[prof]) if correct_acc[prof] else 0.0)
        for prof in profiles
    }
    avg_time = {
        prof: (sum(time_acc[prof]) / len(time_acc[prof]) if time_acc[prof] else 0.0)
        for prof in profiles
    }
    avg_tokens = {
        prof: (sum(token_acc[prof]) / len(token_acc[prof]) if token_acc[prof] else 0.0)
        for prof in profiles
    }

    return ComparisonReport(
        header=header,
        profiles=profiles,
        problems=problems,
        results=dict(results),
        accuracy=accuracy,
        avg_time=avg_time,
        avg_tokens=avg_tokens,
    )


def format_markdown(report: ComparisonReport) -> str:
    """Render *report* as a Markdown string with two tables."""
    lines: list[str] = []

    lines.append("# Solvay Benchmark Comparison")
    lines.append("")
    lines.append(
        f"**Run ID:** {report.header.run_id} &nbsp;|&nbsp; "
        f"**Problems:** {len(report.problems)} &nbsp;|&nbsp; "
        f"**Date:** {report.header.started_at}"
    )
    lines.append("")

    # ── Per-problem table ────────────────────────────────────────────────────
    lines.append("## Per-problem results")
    lines.append("")

    header_cells = ["| Problem"] + [f" {p} " for p in report.profiles] + ["|"]
    lines.append("|".join(header_cells))

    sep_cells = ["|---"] + [" :---: " for _ in report.profiles] + ["|"]
    lines.append("|".join(sep_cells))

    for pid in report.problems:
        row = [f"| `{pid}`"]
        for prof in report.profiles:
            val = report.results.get(pid, {}).get(prof)
            if val is None:
                cell = " - "
            elif val:
                cell = " ✓ "
            else:
                cell = " ✗ "
            row.append(cell)
        row.append("|")
        lines.append("|".join(row))

    lines.append("")

    # ── Summary table ────────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")

    sum_header = ["| Metric"] + [f" {p} " for p in report.profiles] + ["|"]
    lines.append("|".join(sum_header))

    sum_sep = ["|---"] + [" ---: " for _ in report.profiles] + ["|"]
    lines.append("|".join(sum_sep))

    # Accuracy row
    acc_row = ["| Accuracy (%)"]
    for prof in report.profiles:
        acc_row.append(f" {report.accuracy[prof] * 100:.1f}% ")
    acc_row.append("|")
    lines.append("|".join(acc_row))

    # Avg time row
    time_row = ["| Avg time (s)"]
    for prof in report.profiles:
        time_row.append(f" {report.avg_time[prof]:.2f} ")
    time_row.append("|")
    lines.append("|".join(time_row))

    # Avg output tokens row
    tok_row = ["| Avg output tokens"]
    for prof in report.profiles:
        tok_row.append(f" {report.avg_tokens[prof]:.0f} ")
    tok_row.append("|")
    lines.append("|".join(tok_row))

    lines.append("")

    return "\n".join(lines)
