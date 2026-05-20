"""Benchmark runner: evaluate Solvay against curated physics problems."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import sympy


def run_benchmark(
    problems_dir: Path,
    results_file: Path | None = None,
) -> dict[str, Any]:
    """Run all benchmark problems and evaluate correctness.

    Args:
        problems_dir: Directory containing problem JSON files.
        results_file: Optional path to save results.

    Returns:
        Dict with overall and per-problem results.
    """
    from solvay.agent import create_solvay_agent
    from solvay.config import SolvayConfig

    config = SolvayConfig()
    agent = create_solvay_agent(config)

    problem_files = sorted(problems_dir.glob("*.json"))
    if not problem_files:
        return {"error": "No problem files found", "problems_dir": str(problems_dir)}

    results: list[dict[str, Any]] = []
    total_correct = 0

    for pf in problem_files:
        problem = json.loads(pf.read_text(encoding="utf-8"))
        print(f"\n{'=' * 60}")
        print(f"Problem: {problem['id']} ({problem['domain']})")
        print(f"  {problem['statement'][:80]}...")

        start = time.time()
        try:
            agent_result = agent.invoke(
                {"messages": [{"role": "user", "content": problem["statement"]}]}
            )
            elapsed = time.time() - start
            final_msg = agent_result["messages"][-1].content

            correct = evaluate_answer(final_msg, problem["expected"])
            if correct:
                total_correct += 1

            results.append(
                {
                    "id": problem["id"],
                    "domain": problem["domain"],
                    "correct": correct,
                    "elapsed_seconds": round(elapsed, 1),
                    "answer_excerpt": final_msg[:200],
                }
            )
        except Exception as exc:
            elapsed = time.time() - start
            results.append(
                {
                    "id": problem["id"],
                    "domain": problem["domain"],
                    "correct": False,
                    "elapsed_seconds": round(elapsed, 1),
                    "error": str(exc),
                }
            )

    summary = {
        "total": len(results),
        "correct": total_correct,
        "accuracy": round(total_correct / len(results), 3) if results else 0,
        "by_domain": _group_by_domain(results),
        "results": results,
    }

    return summary


def evaluate_answer(
    agent_output: str,
    expected: dict[str, Any],
) -> bool:
    """Evaluate whether the agent's answer matches the expected value.

    Tries symbolic equivalence first, then falls back to numerical comparison.

    Args:
        agent_output: The agent's final output text.
        expected: Dict with 'value', 'unit', and 'tolerance_rel'.

    Returns:
        True if the answer is correct within tolerance.
    """
    expected_value = expected["value"]
    tolerance = expected.get("tolerance_rel", 0.01)

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", agent_output)
    if not numbers:
        return False

    try:
        expected_expr = sympy.sympify(expected_value)
        expected_float = float(expected_expr.evalf())

        for num_str in numbers:
            try:
                candidate = float(num_str)
                if expected_float != 0:
                    rel_error = abs(candidate - expected_float) / abs(expected_float)
                    if rel_error <= tolerance:
                        return True
                elif abs(candidate) <= tolerance:
                    return True
            except ValueError:
                continue
    except Exception:
        pass

    return False


def _group_by_domain(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group results by domain."""
    domains: dict[str, dict[str, Any]] = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"total": 0, "correct": 0}
        domains[d]["total"] += 1
        if r.get("correct"):
            domains[d]["correct"] += 1
    for d in domains:
        domains[d]["accuracy"] = (
            round(domains[d]["correct"] / domains[d]["total"], 3) if domains[d]["total"] > 0 else 0
        )
    return domains
