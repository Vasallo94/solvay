"""Re-grade a benchmark JSONL file with the current grading code.

Usage: uv run python benchmark/regrade.py benchmark/runs/solvay-full-2026-06-05.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.grading import evaluate_correct
from solvay.benchmark.schema import Expected, Problem


def load_problems() -> dict[str, dict]:
    problems = {}
    for f in Path("benchmark/problems").rglob("*.json"):
        p = json.loads(f.read_text())
        problems[p["id"]] = p
    return problems


def regrade(run_file: Path) -> None:
    problems = load_problems()
    config = BenchConfig(grader_model="azure_openai:gpt-5.5-codex")
    lines = run_file.read_text().strip().split("\n")

    for line in lines:
        d = json.loads(line)
        if d.get("kind") != "record":
            continue

        pid = d["problem_id"]
        answer = d.get("answer_raw", "")
        prob_data = problems[pid]
        exp = prob_data["expected"]
        expected = Expected(
            kind=exp["kind"],
            value=exp["value"],
            unit=exp.get("unit"),
            tolerance_rel=exp.get("tolerance_rel", 0.01),
            verification={"method": "sympy_equivalence", "script": None},
        )

        old_correct = d["correct"]
        old_method = d.get("grading_method", "?")

        new_correct, new_method = evaluate_correct(
            answer, expected, Problem.model_validate(prob_data), config
        )

        changed = "CHANGED" if old_correct != new_correct else ""
        print(
            f"{pid:30s}  old={old_correct!s:5s}({old_method:10s})  "
            f"new={new_correct!s:5s}({new_method:10s})  {changed}"
        )


if __name__ == "__main__":
    regrade(Path(sys.argv[1]))
