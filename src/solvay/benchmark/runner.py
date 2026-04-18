"""Matrix runner over (problems x profiles x models x repeats)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import sympy

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.fingerprint import compute_fingerprint
from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.schema import Expected, Problem
from solvay.config import SolvayConfig


@dataclass(frozen=True)
class MatrixSpec:
    problems: list[Problem]
    profile_names: list[str]
    models: list[str]
    repeats: int = 1


def _git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, dirty


def _solvay_version() -> str:
    try:
        return importlib.metadata.version("solvay")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _evaluate_correct(answer_raw: str, expected: Expected) -> bool:
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer_raw)
    if not numbers:
        return False
    try:
        target = float(sympy.sympify(expected.value).evalf())
    except (ValueError, TypeError, sympy.SympifyError):
        return False
    for num in numbers:
        try:
            candidate = float(num)
        except ValueError:
            continue
        if target == 0:
            if abs(candidate) <= expected.tolerance_rel:
                return True
        else:
            if abs(candidate - target) / abs(target) <= expected.tolerance_rel:
                return True
    return False


def run_matrix(
    spec: MatrixSpec,
    out_path: Path,
    config: BenchConfig,
    solvay_config: SolvayConfig | None = None,
) -> Path:
    """Execute the matrix and stream results to a JSONL file."""
    scfg = solvay_config or SolvayConfig()
    commit, dirty = _git_commit()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S")
    header = RunHeader(
        run_id=run_id,
        solvay_version=_solvay_version(),
        git_commit=commit,
        git_dirty=dirty,
        config_fingerprint=compute_fingerprint(scfg),
        started_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    out_path = Path(out_path)
    with RunWriter(out_path, header) as writer:
        for problem in spec.problems:
            for name in spec.profile_names:
                profile = get_profile(name)
                for model in spec.models:
                    for repeat_idx in range(spec.repeats):
                        res = profile.runner(problem, model, config)
                        answer_for_eval = (
                            res.answer_extracted
                            if res.answer_extracted is not None
                            else res.answer_raw
                        )
                        correct = (
                            False
                            if res.error
                            else _evaluate_correct(answer_for_eval, problem.expected)
                        )
                        writer.write(
                            RunRecord(
                                problem_id=problem.id,
                                profile=name,
                                model=model,
                                repeat_idx=repeat_idx,
                                answer_raw=res.answer_raw,
                                answer_extracted=res.answer_extracted,
                                correct=correct,
                                elapsed_seconds=res.elapsed_seconds,
                                tokens=res.tokens,
                                trace_id=res.trace_id,
                                error=res.error,
                            )
                        )
    return out_path
