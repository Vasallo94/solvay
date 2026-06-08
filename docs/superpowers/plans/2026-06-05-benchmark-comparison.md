# Benchmark Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hard physics problems, improve grading (SymPy + LLM-judge), and add a `compare` command so we can measure Solvay's multi-agent pipeline against raw LLM baselines.

**Architecture:** Three independent components. (1) Improved grading replaces the regex-only `_evaluate_correct` with SymPy equivalence + LLM-as-judge fallback, adding `grading_method` to RunRecord. (2) Compare command reads a JSONL run file and produces a markdown comparison table. (3) Problem files are hand-written JSON following the existing Problem schema.

**Tech Stack:** SymPy, langchain init_chat_model, Pydantic, Typer CLI, JSONL.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/solvay/benchmark/grading.py` | Create | SymPy grader + LLM-judge grader |
| `src/solvay/benchmark/runner.py` | Modify | Use new grading, pass `grading_method` to RunRecord |
| `src/solvay/benchmark/jsonl.py` | Modify | Add `grading_method` field to RunRecord |
| `src/solvay/benchmark/config.py` | Modify | Add `grader_model` field |
| `src/solvay/benchmark/compare.py` | Create | Comparison report from JSONL |
| `src/solvay/benchmark/cli.py` | Modify | Add `compare` command |
| `tests/benchmark/test_grading.py` | Create | Tests for SymPy + LLM grading |
| `tests/benchmark/test_compare.py` | Create | Tests for comparison report |
| `benchmark/problems/` | Create | 23 problem JSON files |

---

## Task 1: Create grading module with SymPy + LLM-judge

**Files:**
- Create: `src/solvay/benchmark/grading.py`
- Create: `tests/benchmark/test_grading.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmark/test_grading.py`:

```python
"""Tests for the two-stage grading system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from solvay.benchmark.schema import Expected, Verification


def _numeric_expected(value: str, unit: str | None = None, tol: float = 0.01) -> Expected:
    return Expected(
        kind="numeric",
        value=value,
        unit=unit,
        tolerance_rel=tol,
        verification=Verification(method="numeric_eval", script=None),
    )


def _symbolic_expected(value: str) -> Expected:
    return Expected(
        kind="symbolic",
        value=value,
        unit=None,
        tolerance_rel=0.01,
        verification=Verification(method="sympy_equivalence", script=None),
    )


class TestSympyGrade:
    def test_numeric_exact_match(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade("The answer is 4.905 m/s^2", _numeric_expected("4.905"))
        assert result is True

    def test_numeric_within_tolerance(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade("The acceleration is 4.91 m/s^2", _numeric_expected("4.905"))
        assert result is True

    def test_numeric_outside_tolerance(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade("The answer is 5.5", _numeric_expected("4.905"))
        assert result is False

    def test_symbolic_equivalent_expressions(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade(
            "The torque is 32*pi*sigma*B0**2*R**5*omega/315",
            _symbolic_expected("32*pi*sigma*B0**2*R**5*omega/315"),
        )
        assert result is True

    def test_symbolic_different_form_same_value(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade(
            "v = sqrt(2*g*h)",
            _symbolic_expected("(2*g*h)**(1/2)"),
        )
        assert result is True

    def test_unparseable_returns_none(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade(
            "The answer involves complex boundary conditions that cannot be expressed simply.",
            _numeric_expected("4.905"),
        )
        assert result is None

    def test_no_numbers_returns_none(self) -> None:
        from solvay.benchmark.grading import sympy_grade

        result = sympy_grade("I don't know the answer.", _numeric_expected("4.905"))
        assert result is None


class TestLlmGrade:
    def test_correct_verdict(self) -> None:
        from solvay.benchmark.grading import llm_grade
        from solvay.benchmark.config import BenchConfig
        from solvay.benchmark.schema import Problem

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"correct": true, "reason": "Values match"}'
        )

        problem = _fake_problem()
        config = BenchConfig(grader_model="fake-model")

        with patch("solvay.benchmark.grading.init_chat_model", return_value=mock_llm):
            result = llm_grade("4.905 m/s^2", problem.expected, problem, config)

        assert result is True

    def test_incorrect_verdict(self) -> None:
        from solvay.benchmark.grading import llm_grade
        from solvay.benchmark.config import BenchConfig

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"correct": false, "reason": "Wrong value"}'
        )

        problem = _fake_problem()
        config = BenchConfig(grader_model="fake-model")

        with patch("solvay.benchmark.grading.init_chat_model", return_value=mock_llm):
            result = llm_grade("999", problem.expected, problem, config)

        assert result is False

    def test_unparseable_llm_response_returns_false(self) -> None:
        from solvay.benchmark.grading import llm_grade
        from solvay.benchmark.config import BenchConfig

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I can't compare these")

        problem = _fake_problem()
        config = BenchConfig(grader_model="fake-model")

        with patch("solvay.benchmark.grading.init_chat_model", return_value=mock_llm):
            result = llm_grade("4.905", problem.expected, problem, config)

        assert result is False


class TestEvaluateCorrect:
    def test_uses_sympy_when_parseable(self) -> None:
        from solvay.benchmark.grading import evaluate_correct
        from solvay.benchmark.config import BenchConfig

        problem = _fake_problem()
        config = BenchConfig()
        correct, method = evaluate_correct("The answer is 2.0", problem.expected, problem, config)
        assert correct is True
        assert method == "sympy"

    def test_falls_back_to_llm_when_sympy_fails(self) -> None:
        from solvay.benchmark.grading import evaluate_correct
        from solvay.benchmark.config import BenchConfig

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"correct": true, "reason": "Match"}'
        )

        problem = _fake_problem()
        config = BenchConfig(grader_model="fake-model")

        with patch("solvay.benchmark.grading.init_chat_model", return_value=mock_llm):
            correct, method = evaluate_correct(
                "The answer involves a complex expression blah blah",
                problem.expected,
                problem,
                config,
            )

        assert method == "llm-judge"


def _fake_problem() -> "Problem":
    from solvay.benchmark.schema import Problem

    return Problem.model_validate({
        "id": "test-001",
        "version": 1,
        "source": {"kind": "synthetic", "origin": "test", "generated_at": "2026-01-01T00:00:00Z", "seed": 0},
        "domain": "mechanics",
        "statement": "Find the acceleration.",
        "given": {"m": "2 kg"},
        "find": "acceleration",
        "expected": {
            "kind": "numeric",
            "value": "2",
            "unit": "m/s^2",
            "tolerance_rel": 0.01,
            "verification": {"method": "numeric_eval", "script": None},
        },
    })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/benchmark/test_grading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'solvay.benchmark.grading'`

- [ ] **Step 3: Implement the grading module**

Create `src/solvay/benchmark/grading.py`:

```python
"""Two-stage grading: SymPy equivalence first, LLM-as-judge fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import sympy

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Expected, Problem

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_SYMBOLIC_RE = re.compile(
    r"[a-zA-Z_]\w*(?:\s*[\*\+\-/\^]\s*[a-zA-Z_\d\.\(\)]+)+"
)


def sympy_grade(answer_raw: str, expected: Expected) -> bool | None:
    """Grade using SymPy. Returns True/False if parseable, None otherwise."""
    if expected.kind == "numeric":
        return _grade_numeric(answer_raw, expected)
    return _grade_symbolic(answer_raw, expected)


def _grade_numeric(answer_raw: str, expected: Expected) -> bool | None:
    numbers = _NUMBER_RE.findall(answer_raw)
    if not numbers:
        return None

    try:
        target = float(sympy.sympify(expected.value).evalf())
    except (ValueError, TypeError, sympy.SympifyError):
        return None

    for num_str in numbers:
        try:
            candidate = float(num_str)
        except ValueError:
            continue
        if target == 0:
            if abs(candidate) <= expected.tolerance_rel:
                return True
        else:
            if abs(candidate - target) / abs(target) <= expected.tolerance_rel:
                return True

    return False


def _grade_symbolic(answer_raw: str, expected: Expected) -> bool | None:
    try:
        expected_expr = sympy.sympify(expected.value)
    except (ValueError, TypeError, sympy.SympifyError):
        return None

    candidates = _SYMBOLIC_RE.findall(answer_raw)
    candidates.extend(_NUMBER_RE.findall(answer_raw))

    for candidate_str in candidates:
        candidate_str = candidate_str.replace("^", "**")
        try:
            candidate_expr = sympy.sympify(candidate_str)
        except (ValueError, TypeError, sympy.SympifyError):
            continue
        try:
            diff = sympy.simplify(candidate_expr - expected_expr)
            if diff == 0:
                return True
        except Exception:
            continue

    return None if not candidates else False


def llm_grade(
    answer_raw: str,
    expected: Expected,
    problem: Problem,
    config: BenchConfig,
) -> bool:
    """Grade using an LLM as judge. Returns True if the LLM considers the answer correct."""
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(config.grader_model)

    unit_str = f" {expected.unit}" if expected.unit else ""
    prompt = (
        f"Compare this physics answer to the expected answer.\n\n"
        f"Problem: {problem.statement}\n"
        f"Student answer: {answer_raw}\n"
        f"Expected: {expected.value}{unit_str}\n\n"
        f"Are they mathematically equivalent? "
        f'Return JSON: {{"correct": true/false, "reason": "..."}}'
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        parsed = json.loads(content)
        return bool(parsed.get("correct", False))
    except (json.JSONDecodeError, ValueError):
        return False


def evaluate_correct(
    answer_raw: str,
    expected: Expected,
    problem: Problem,
    config: BenchConfig,
) -> tuple[bool, str]:
    """Two-stage grading. Returns (correct, grading_method)."""
    sympy_result = sympy_grade(answer_raw, expected)
    if sympy_result is not None:
        return sympy_result, "sympy"
    return llm_grade(answer_raw, expected, problem, config), "llm-judge"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/benchmark/test_grading.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 6: Commit**

```bash
git add src/solvay/benchmark/grading.py tests/benchmark/test_grading.py
git commit -m "feat: add two-stage grading (SymPy + LLM-judge)

SymPy equivalence check handles numeric and symbolic answers
deterministically. Falls back to LLM-as-judge for unparseable
answers. Returns grading_method for analysis."
```

---

## Task 2: Wire grading into runner + update RunRecord

**Files:**
- Modify: `src/solvay/benchmark/jsonl.py:28-39`
- Modify: `src/solvay/benchmark/config.py:10-25`
- Modify: `src/solvay/benchmark/runner.py:46-119`
- Modify: `tests/benchmark/test_runner.py`

- [ ] **Step 1: Add `grader_model` to BenchConfig**

Edit `src/solvay/benchmark/config.py`:

```python
"""Benchmark-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass

from solvay.config import DEFAULT_MODEL


@dataclass(frozen=True)
class BenchConfig:
    """Configuration for the benchmark suite."""

    composer_model: str = DEFAULT_MODEL
    probe_model: str = DEFAULT_MODEL
    error_classifier_model: str = DEFAULT_MODEL
    grader_model: str = DEFAULT_MODEL
    composer_max_attempts: int = 3
    cost_guard_threshold: int = 100
```

- [ ] **Step 2: Add `grading_method` to RunRecord**

Edit `src/solvay/benchmark/jsonl.py`, add the field to `RunRecord` after `error`:

```python
@dataclass(frozen=True)
class RunRecord:
    problem_id: str
    profile: str
    model: str
    repeat_idx: int
    answer_raw: str
    answer_extracted: str | None
    correct: bool
    elapsed_seconds: float
    tokens: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None
    error: str | None = None
    grading_method: str = "regex"
```

- [ ] **Step 3: Update runner to use new grading**

Edit `src/solvay/benchmark/runner.py`. Replace the `_evaluate_correct` function and update `run_matrix`:

```python
"""Matrix runner over (problems x profiles x models x repeats)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import subprocess
from dataclasses import dataclass
from pathlib import Path

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.fingerprint import compute_fingerprint
from solvay.benchmark.grading import evaluate_correct
from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.schema import Problem
from solvay.config import SolvayConfig


@dataclass(frozen=True)
class MatrixSpec:
    problems: list[Problem]
    profile_names: list[str]
    models: list[str]
    repeats: int = 1


def _git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, dirty


def _solvay_version() -> str:
    try:
        return importlib.metadata.version("solvay")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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
                        if res.error:
                            correct, grading_method = False, "error"
                        else:
                            correct, grading_method = evaluate_correct(
                                answer_for_eval, problem.expected, problem, config
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
                                grading_method=grading_method,
                            )
                        )
    return out_path
```

- [ ] **Step 4: Update test for new RunRecord field**

Edit `tests/benchmark/test_runner.py`. The existing tests should still pass because `grading_method` has a default. Add a new test:

```python
def test_run_matrix_records_grading_method(tmp_path: Path) -> None:
    spec = MatrixSpec(
        problems=[_fake_problem()],
        profile_names=["fake"],
        models=["anthropic:claude-sonnet-4-6"],
        repeats=1,
    )
    out = tmp_path / "run.jsonl"
    run_matrix(spec, out_path=out, config=BenchConfig())
    _, records = read_run(out)
    assert records[0].grading_method in ("sympy", "llm-judge", "error", "regex")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/benchmark/test_runner.py tests/benchmark/test_grading.py -v`
Expected: PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 7: Commit**

```bash
git add src/solvay/benchmark/runner.py src/solvay/benchmark/jsonl.py src/solvay/benchmark/config.py tests/benchmark/test_runner.py
git commit -m "feat: wire two-stage grading into benchmark runner

Replace regex-only _evaluate_correct with SymPy + LLM-judge.
Add grading_method field to RunRecord and grader_model to BenchConfig."
```

---

## Task 3: Create compare command

**Files:**
- Create: `src/solvay/benchmark/compare.py`
- Create: `tests/benchmark/test_compare.py`
- Modify: `src/solvay/benchmark/cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmark/test_compare.py`:

```python
"""Tests for the benchmark comparison report."""

from __future__ import annotations

from solvay.benchmark.jsonl import RunHeader, RunRecord


def _header() -> RunHeader:
    return RunHeader(
        run_id="test-run",
        solvay_version="0.1.0",
        git_commit="abc1234",
        git_dirty=False,
        config_fingerprint="sha256:test",
        started_at="2026-06-05T00:00:00Z",
    )


def _record(
    problem_id: str, profile: str, correct: bool, elapsed: float = 1.0, model: str = "test-model"
) -> RunRecord:
    return RunRecord(
        problem_id=problem_id,
        profile=profile,
        model=model,
        repeat_idx=0,
        answer_raw="test",
        answer_extracted=None,
        correct=correct,
        elapsed_seconds=elapsed,
        tokens={"input": 100, "output": 200},
        grading_method="sympy",
    )


class TestComparisonReport:
    def test_builds_from_records(self) -> None:
        from solvay.benchmark.compare import build_comparison

        records = [
            _record("p1", "bare", False, 1.0),
            _record("p1", "solvay-full", True, 10.0),
            _record("p2", "bare", True, 2.0),
            _record("p2", "solvay-full", True, 15.0),
        ]
        report = build_comparison(_header(), records)

        assert len(report.profiles) == 2
        assert "bare" in report.profiles
        assert "solvay-full" in report.profiles
        assert len(report.problems) == 2

    def test_accuracy_per_profile(self) -> None:
        from solvay.benchmark.compare import build_comparison

        records = [
            _record("p1", "bare", False),
            _record("p2", "bare", True),
            _record("p1", "solvay-full", True),
            _record("p2", "solvay-full", True),
        ]
        report = build_comparison(_header(), records)

        assert report.accuracy["bare"] == 0.5
        assert report.accuracy["solvay-full"] == 1.0

    def test_format_markdown_contains_table(self) -> None:
        from solvay.benchmark.compare import build_comparison, format_markdown

        records = [
            _record("p1", "bare", False),
            _record("p1", "prompted", True),
        ]
        report = build_comparison(_header(), records)
        md = format_markdown(report)

        assert "| Problem" in md
        assert "bare" in md
        assert "prompted" in md
        assert "Accuracy" in md

    def test_handles_single_profile(self) -> None:
        from solvay.benchmark.compare import build_comparison, format_markdown

        records = [_record("p1", "bare", True)]
        report = build_comparison(_header(), records)
        md = format_markdown(report)

        assert "bare" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/benchmark/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement compare module**

Create `src/solvay/benchmark/compare.py`:

```python
"""Comparison report from a benchmark JSONL run."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from solvay.benchmark.jsonl import RunHeader, RunRecord


@dataclass
class ComparisonReport:
    header: RunHeader
    profiles: list[str]
    problems: list[str]
    results: dict[str, dict[str, bool | None]] = field(default_factory=dict)
    accuracy: dict[str, float] = field(default_factory=dict)
    avg_time: dict[str, float] = field(default_factory=dict)
    avg_tokens: dict[str, float] = field(default_factory=dict)


def build_comparison(header: RunHeader, records: list[RunRecord]) -> ComparisonReport:
    """Build a comparison report from run records."""
    profiles_set: set[str] = set()
    problems_set: set[str] = set()
    results: dict[str, dict[str, bool | None]] = defaultdict(dict)
    correct_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    time_sums: dict[str, float] = defaultdict(float)
    token_sums: dict[str, int] = defaultdict(int)

    for r in records:
        profiles_set.add(r.profile)
        problems_set.add(r.problem_id)
        results[r.problem_id][r.profile] = r.correct
        total_counts[r.profile] += 1
        if r.correct:
            correct_counts[r.profile] += 1
        time_sums[r.profile] += r.elapsed_seconds
        token_sums[r.profile] += r.tokens.get("output", 0)

    profiles = sorted(profiles_set)
    problems = sorted(problems_set)

    accuracy = {
        p: correct_counts[p] / total_counts[p] if total_counts[p] > 0 else 0.0
        for p in profiles
    }
    avg_time = {
        p: time_sums[p] / total_counts[p] if total_counts[p] > 0 else 0.0
        for p in profiles
    }
    avg_tokens = {
        p: token_sums[p] / total_counts[p] if total_counts[p] > 0 else 0.0
        for p in profiles
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
    """Format a comparison report as a markdown string."""
    lines: list[str] = []

    lines.append("# Solvay Benchmark Comparison")
    lines.append("")
    lines.append(
        f"Run: {report.header.run_id} | "
        f"Problems: {len(report.problems)} | "
        f"Date: {report.header.started_at[:10]}"
    )
    lines.append("")

    prof_headers = " | ".join(report.profiles)
    lines.append(f"| Problem | {prof_headers} |")
    sep = " | ".join("---" for _ in report.profiles)
    lines.append(f"| --- | {sep} |")

    for pid in report.problems:
        cells: list[str] = []
        for prof in report.profiles:
            result = report.results.get(pid, {}).get(prof)
            if result is True:
                cells.append("✓")
            elif result is False:
                cells.append("✗")
            else:
                cells.append("-")
        row = " | ".join(cells)
        lines.append(f"| {pid} | {row} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | {prof_headers} |")
    lines.append(f"| --- | {sep} |")

    acc_cells = " | ".join(f"{report.accuracy[p]:.0%}" for p in report.profiles)
    lines.append(f"| Accuracy | {acc_cells} |")

    time_cells = " | ".join(f"{report.avg_time[p]:.1f}s" for p in report.profiles)
    lines.append(f"| Avg time | {time_cells} |")

    tok_cells = " | ".join(f"{report.avg_tokens[p]:.0f}" for p in report.profiles)
    lines.append(f"| Avg output tokens | {tok_cells} |")

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/benchmark/test_compare.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Add `compare` command to CLI**

Edit `src/solvay/benchmark/cli.py`. Add the command after the `generate_cmd`:

```python
@app.command("compare")
def compare_cmd(
    run_file: Annotated[
        Path, typer.Argument(help="Path to a benchmark JSONL run file.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write markdown report to file."),
    ] = None,
) -> None:
    """Compare profiles from a benchmark run and produce a markdown table."""
    from solvay.benchmark.compare import build_comparison, format_markdown
    from solvay.benchmark.jsonl import read_run

    if not run_file.exists():
        typer.echo(f"File not found: {run_file}", err=True)
        raise typer.Exit(1)

    header, records = read_run(run_file)
    if not records:
        typer.echo("No records found in run file.", err=True)
        raise typer.Exit(1)

    report = build_comparison(header, records)
    md = format_markdown(report)
    typer.echo(md)

    if output:
        output.write_text(md, encoding="utf-8")
        typer.echo(f"\nReport saved to {output}")
```

- [ ] **Step 6: Run full suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 7: Commit**

```bash
git add src/solvay/benchmark/compare.py src/solvay/benchmark/cli.py tests/benchmark/test_compare.py
git commit -m "feat: add benchmark compare command

Reads a JSONL run file and produces a markdown comparison table
showing per-problem results and summary metrics (accuracy, time,
tokens) across all profiles."
```

---

## Task 4: Create hand-crafted hard problems (stress test set)

**Files:**
- Create: `benchmark/problems/em/quadrupole-braking.json`
- Create: `benchmark/problems/waves/acoustic-hawking.json`
- Create: `benchmark/problems/quantum/phonon-casimir.json`
- Create: `benchmark/problems/em/hartmann-flow.json`
- Create: `benchmark/problems/em/synchrotron-radiation.json`
- Create: `benchmark/problems/quantum/floquet-tunneling.json`
- Create: `benchmark/problems/em/electromagnetic-sail.json`
- Create: `benchmark/problems/quantum/thermal-casimir.json`

- [ ] **Step 1: Create domain directories**

Run:
```bash
mkdir -p benchmark/problems/em benchmark/problems/waves benchmark/problems/quantum benchmark/problems/thermo
```

- [ ] **Step 2: Create the 3 tested stress-test problems**

Create `benchmark/problems/em/quadrupole-braking.json`:

```json
{
  "id": "novel-em-001",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-stress-test-2026-06-02",
    "generated_at": "2026-06-02T00:00:00Z",
    "seed": null
  },
  "domain": "em",
  "subdomain": "eddy-currents",
  "tags": ["quadrupole", "eddy-current", "braking", "boundary-conditions", "multi-part"],
  "difficulty": "hard",
  "statement": "A solid conducting sphere of radius R, electrical conductivity σ, and mass m spins with initial angular velocity ω₀ about the z-axis. The sphere sits at the center of a magnetic quadrupole field B_ext = (B₀/R)(x ŷ + y x̂). Working to lowest order in Rm = μ₀σωR² ≪ 1: (a) Find the eddy current density J. (b) Find the braking torque N. (c) Find ω(t). (d) Verify energy conservation.",
  "given": {
    "R": "sphere radius",
    "sigma": "conductivity",
    "m": "mass",
    "omega_0": "initial angular velocity",
    "B_0": "field strength constant"
  },
  "find": "Braking torque proportionality constant and characteristic time",
  "expected": {
    "kind": "symbolic",
    "value": "32*pi*sigma*B0**2*R**5/315",
    "unit": "N·m/(rad/s)",
    "tolerance_rel": 0.01,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": {
    "checked_at": "2026-06-02T00:00:00Z",
    "method": "novel-problem-design",
    "score": 0.05,
    "verdict": "clean"
  },
  "notes": "Requires electric field correction for J·n=0 at surface. Most models miss the boundary condition and get 16π/105 instead of 32π/315."
}
```

Create `benchmark/problems/waves/acoustic-hawking.json`:

```json
{
  "id": "novel-waves-001",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-stress-test-2026-06-02",
    "generated_at": "2026-06-02T00:00:00Z",
    "seed": null
  },
  "domain": "waves",
  "subdomain": "analogue-gravity",
  "tags": ["hawking", "acoustic", "black-hole", "sonic-horizon", "multi-domain"],
  "difficulty": "hard",
  "statement": "A 2D draining bathtub vortex has radial velocity v(r) = -Q/(2πr). The speed of sound is c_s. (a) Find the sonic horizon radius r_s. (b) Derive the acoustic metric. (c) Compute the Hawking temperature T_H. (d) Estimate T_H for Q=10⁻⁴ m²/s, c_s=1500 m/s.",
  "given": {
    "Q": "volumetric flow rate per unit height",
    "c_s": "speed of sound"
  },
  "find": "Hawking temperature of acoustic black hole",
  "expected": {
    "kind": "symbolic",
    "value": "hbar*c_s/(k_B*Q)",
    "unit": "K",
    "tolerance_rel": 0.01,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": {
    "checked_at": "2026-06-02T00:00:00Z",
    "method": "novel-problem-design",
    "score": 0.1,
    "verdict": "clean"
  },
  "notes": "Multi-domain: fluid dynamics + wave mechanics + GR analogy + quantum mechanics."
}
```

Create `benchmark/problems/quantum/phonon-casimir.json`:

```json
{
  "id": "novel-qm-001",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-stress-test-2026-06-02",
    "generated_at": "2026-06-02T00:00:00Z",
    "seed": null
  },
  "domain": "quantum",
  "subdomain": "condensed-matter",
  "tags": ["casimir", "phonon", "lattice", "green-function", "t-matrix"],
  "difficulty": "hard",
  "statement": "Two isotopic impurities (mass M ≠ m) in an infinite 1D monatomic lattice (mass m, spring constant k, spacing a) at sites 0 and N. Using the T-matrix formalism and Krein-Friedel-Lloyd formula: (a) Find the single-impurity T-matrix. (b) Derive the two-impurity secular equation. (c) Find the zero-point energy scaling ΔE(d). (d) Derive the force F(d) and compare with QED Casimir.",
  "given": {
    "m": "host atom mass",
    "M": "impurity mass",
    "k": "spring constant",
    "a": "lattice spacing",
    "d": "Na, separation"
  },
  "find": "Power-law exponent n in ΔE ~ d^(-n) and force scaling",
  "expected": {
    "kind": "numeric",
    "value": "3",
    "unit": null,
    "tolerance_rel": 0.01,
    "verification": {
      "method": "numeric_eval",
      "script": null
    }
  },
  "contamination": {
    "checked_at": "2026-06-02T00:00:00Z",
    "method": "novel-problem-design",
    "score": 0.05,
    "verdict": "clean"
  },
  "notes": "Requires Wick rotation to imaginary frequencies. Correct exponent n=3 (not n=1 as Friedel oscillations suggest). Force scales as d^(-4)."
}
```

- [ ] **Step 3: Create 5 new novel hard problems**

Create `benchmark/problems/em/hartmann-flow.json`:

```json
{
  "id": "novel-em-002",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-novel-2026-06-05",
    "generated_at": "2026-06-05T00:00:00Z",
    "seed": null
  },
  "domain": "em",
  "subdomain": "magnetohydrodynamics",
  "tags": ["mhd", "hartmann", "navier-stokes", "heat-transfer", "multi-domain"],
  "difficulty": "hard",
  "statement": "A conducting fluid (conductivity σ, viscosity μ, density ρ) flows between parallel plates separated by distance 2L in a transverse magnetic field B₀. One plate is at T₁, the other at T₂. Derive the velocity profile u(y) and the Hartmann number Ha = B₀L√(σ/(μ)). Show that the Nusselt number approaches 1 as Ha → ∞.",
  "given": {
    "sigma": "fluid conductivity",
    "mu": "dynamic viscosity",
    "rho": "density",
    "B_0": "magnetic field",
    "L": "half-channel width"
  },
  "find": "Hartmann number definition and Nusselt number limit",
  "expected": {
    "kind": "symbolic",
    "value": "B_0*L*sqrt(sigma/mu)",
    "unit": null,
    "tolerance_rel": 0.01,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": null,
  "notes": "Multi-domain: Navier-Stokes + Lorentz force + heat equation."
}
```

Create `benchmark/problems/em/synchrotron-radiation.json`:

```json
{
  "id": "novel-em-003",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-novel-2026-06-05",
    "generated_at": "2026-06-05T00:00:00Z",
    "seed": null
  },
  "domain": "em",
  "subdomain": "relativistic-electrodynamics",
  "tags": ["synchrotron", "relativistic", "larmor", "radiation"],
  "difficulty": "hard",
  "statement": "A charged particle moves in a circular orbit of radius R at relativistic speed with Lorentz factor γ ≫ 1. Show that the total radiated power scales as γ⁴ (relativistic Larmor formula). Find the critical harmonic number n_c at which the spectrum peaks.",
  "given": {
    "gamma": "Lorentz factor",
    "R": "orbit radius",
    "q": "charge",
    "omega_0": "orbital frequency"
  },
  "find": "Critical harmonic scaling with gamma",
  "expected": {
    "kind": "symbolic",
    "value": "gamma**3",
    "unit": null,
    "tolerance_rel": 0.01,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": null,
  "notes": "n_c ~ γ³. Power ~ γ⁴. Requires retarded-time analysis."
}
```

Create `benchmark/problems/quantum/floquet-tunneling.json`:

```json
{
  "id": "novel-qm-002",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-novel-2026-06-05",
    "generated_at": "2026-06-05T00:00:00Z",
    "seed": null
  },
  "domain": "quantum",
  "subdomain": "time-dependent-qm",
  "tags": ["floquet", "tunneling", "oscillating-barrier", "sidebands"],
  "difficulty": "hard",
  "statement": "A particle of mass m and energy E encounters a rectangular barrier of average height V₀ and width d, whose height oscillates as V(t) = V₀ + V₁cos(ωt) with V₁ ≪ V₀. Using first-order Floquet perturbation theory, show that the transmitted wave contains sidebands at energies E ± ℏω. Derive the ratio of sideband transmission to central transmission.",
  "given": {
    "m": "particle mass",
    "E": "incident energy",
    "V_0": "average barrier height",
    "V_1": "oscillation amplitude",
    "d": "barrier width",
    "omega": "oscillation frequency"
  },
  "find": "Sideband transmission ratio",
  "expected": {
    "kind": "symbolic",
    "value": "(V_1/(2*hbar*omega))**2",
    "unit": null,
    "tolerance_rel": 0.05,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": null,
  "notes": "First-order perturbation in V₁/ℏω. Sidebands at E±ℏω with amplitude ~ V₁/(2ℏω)."
}
```

Create `benchmark/problems/em/electromagnetic-sail.json`:

```json
{
  "id": "novel-em-004",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-novel-2026-06-05",
    "generated_at": "2026-06-05T00:00:00Z",
    "seed": null
  },
  "domain": "em",
  "subdomain": "space-physics",
  "tags": ["magnetic-sail", "solar-wind", "magnetopause", "pressure-balance"],
  "difficulty": "hard",
  "statement": "A spacecraft deploys a superconducting magnetic loop (radius R_loop, current I) to deflect solar wind protons (density n, speed v, mass m_p). Find the magnetopause standoff distance r₀ where magnetic pressure B²/(2μ₀) equals solar wind dynamic pressure ½nm_pv². Express the effective cross-section and thrust force in terms of the magnetic moment μ = πR²_loop·I.",
  "given": {
    "R_loop": "loop radius",
    "I": "current",
    "n": "proton density",
    "v": "solar wind speed",
    "m_p": "proton mass"
  },
  "find": "Standoff distance scaling with magnetic moment",
  "expected": {
    "kind": "symbolic",
    "value": "(mu_0*mu**2/(8*pi**2*n*m_p*v**2))**(1/6)",
    "unit": "m",
    "tolerance_rel": 0.05,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": null,
  "notes": "Dipole field B ~ μ₀μ/(4πr³). Pressure balance gives r₀ ~ μ^(1/3)."
}
```

Create `benchmark/problems/quantum/thermal-casimir.json`:

```json
{
  "id": "novel-qm-003",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-novel-2026-06-05",
    "generated_at": "2026-06-05T00:00:00Z",
    "seed": null
  },
  "domain": "quantum",
  "subdomain": "quantum-field-theory",
  "tags": ["casimir", "thermal", "high-temperature", "plates"],
  "difficulty": "hard",
  "statement": "Two parallel perfectly conducting plates separated by distance d are at temperature T. In the high-temperature limit k_BT ≫ ℏc/d, show that the Casimir free energy per unit area becomes F = -ζ(3)k_BT/(8πd²) where ζ(3) ≈ 1.202 is the Riemann zeta function. Compare with the T=0 Casimir energy E = -π²ℏc/(720d³).",
  "given": {
    "d": "plate separation",
    "T": "temperature"
  },
  "find": "High-temperature Casimir free energy scaling",
  "expected": {
    "kind": "symbolic",
    "value": "k_B*T/d**2",
    "unit": "J/m^2",
    "tolerance_rel": 0.05,
    "verification": {
      "method": "sympy_equivalence",
      "script": null
    }
  },
  "contamination": null,
  "notes": "High-T: classical, scales as T/d². Low-T: quantum, scales as 1/d³. The crossover is at k_BT ~ ℏc/d."
}
```

- [ ] **Step 4: Verify all problems load correctly**

Run:
```bash
uv run python -c "
from solvay.benchmark.schema import load_problems_dir
problems = load_problems_dir('benchmark/problems')
print(f'Loaded {len(problems)} problems:')
for p in problems:
    print(f'  {p.id}: {p.domain}/{p.difficulty} - {p.statement[:60]}...')
"
```
Expected: 9 problems loaded (1 existing + 8 new)

- [ ] **Step 5: Commit**

```bash
git add benchmark/problems/
git commit -m "feat: add 8 hard novel physics benchmark problems

3 from stress test (quadrupole braking, acoustic Hawking, phonon
Casimir) + 5 new (Hartmann flow, synchrotron radiation, Floquet
tunneling, electromagnetic sail, thermal Casimir). All hard
difficulty, multi-domain, with verified expected answers."
```

---

## Task 5: Generate hard problems with the pipeline

This task uses the existing `solvay-bench generate` command to create 15 hard problems across 5 domains.

- [ ] **Step 1: Generate mechanics problems**

Run:
```bash
uv run solvay-bench generate \
    --domain mechanics \
    --compose "gyroscope,precession,angular-momentum" \
    --difficulty hard \
    --n 3 \
    --seed 100 \
    --out benchmark/problems/mechanics
```

- [ ] **Step 2: Generate em problems**

Run:
```bash
uv run solvay-bench generate \
    --domain em \
    --compose "waveguide,cutoff,electromagnetic-modes" \
    --difficulty hard \
    --n 3 \
    --seed 200 \
    --out benchmark/problems/em
```

- [ ] **Step 3: Generate thermo problems**

Run:
```bash
uv run solvay-bench generate \
    --domain thermo \
    --compose "van-der-waals,joule-thomson,thermodynamic-coefficient" \
    --difficulty hard \
    --n 3 \
    --seed 300 \
    --out benchmark/problems/thermo
```

- [ ] **Step 4: Generate quantum problems**

Run:
```bash
uv run solvay-bench generate \
    --domain quantum \
    --compose "tunneling,barrier,wkb-approximation" \
    --difficulty hard \
    --n 3 \
    --seed 400 \
    --out benchmark/problems/quantum
```

- [ ] **Step 5: Generate waves problems**

Run:
```bash
uv run solvay-bench generate \
    --domain waves \
    --compose "cherenkov,dispersion,radiation" \
    --difficulty hard \
    --n 3 \
    --seed 500 \
    --out benchmark/problems/waves
```

- [ ] **Step 6: Verify total problem count**

Run:
```bash
uv run python -c "
from solvay.benchmark.schema import load_problems_dir
problems = load_problems_dir('benchmark/problems')
print(f'Total: {len(problems)} problems')
for domain in sorted(set(p.domain for p in problems)):
    count = sum(1 for p in problems if p.domain == domain)
    print(f'  {domain}: {count}')
"
```
Expected: ~24 problems across 5 domains

- [ ] **Step 7: Commit**

```bash
git add benchmark/problems/
git commit -m "feat: generate 15 hard physics problems across 5 domains

Generated via solvay-bench generate with hard difficulty.
3 per domain: mechanics, em, thermo, quantum, waves."
```

---

## Final Verification

- [ ] **Run full test suite**: `uv run pytest -v` — all passing
- [ ] **Verify problem loading**: `uv run python -c "from solvay.benchmark.schema import load_problems_dir; print(len(load_problems_dir('benchmark/problems')))"` — 20+ problems
- [ ] **Verify compare command**: `uv run solvay-bench compare --help` — shows usage
- [ ] **Dry run**: `uv run solvay-bench run --profiles bare,prompted --models "azure_openai:gpt-5.5-codex" --problems benchmark/problems --domain mechanics --out /tmp/test-run.jsonl && uv run solvay-bench compare /tmp/test-run.jsonl`
