# Solvay Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-layer benchmark suite defined in `docs/superpowers/specs/2026-04-18-benchmark-design.md`: a synthetic problem generator, a matrix runner over (problems × profiles × models) with version fingerprinting, and a reporter computing metrics A (pure reasoning), B (system ladder), and D (robustness — error taxonomy, cost, self-calibration).

**Architecture:** New subpackage `src/solvay/benchmark/` with three internal modules — `generator/`, `profiles/`, `report/` — plus shared `schema.py`, `fingerprint.py`, `runner.py`, `config.py`, and a dedicated Typer CLI at `solvay.benchmark.cli:app` wired as a new `solvay-bench` entry point. File-based contracts: JSON problems in `benchmark/problems/<domain>/`, JSONL runs in `benchmark/runs/`, markdown reports in `benchmark/reports/`.

**Tech Stack:** Python 3.14, Typer (CLI), Pydantic (schema validation), sympy (symbolic verification), scipy (numeric verification), pytest (tests), ruff + mypy (strict). LLM access via the existing `langchain.chat_models.init_chat_model` entry point already used by Solvay.

---

## File structure

### New files

```
src/solvay/benchmark/
  __init__.py
  config.py              # BenchConfig dataclass
  schema.py              # Problem/ProfileResult pydantic models + JSON IO
  fingerprint.py         # config_fingerprint computation
  runner.py              # matrix executor (problems × profiles × models × repeats)
  jsonl.py               # JSONL writer with version header + reader
  cli.py                 # Typer app for solvay-bench
  profiles/
    __init__.py          # Profile dataclass + REGISTRY
    bare.py
    prompted.py
    tooled.py
    solvay_full.py
    solvay_noweb.py
  generator/
    __init__.py
    skeleton.py          # GenerationSkeleton dataclass
    composer.py          # LLM-driven problem authoring
    verifier.py          # sympy + numeric verification
    probe.py             # contamination probe (LLM)
    pipeline.py          # compose → verify → probe orchestrator
  report/
    __init__.py
    loader.py            # JSONL → in-memory RunSet
    metric_a.py
    metric_b.py
    metric_d.py          # D.1, D.2, D.3
    classifier.py        # post-hoc error taxonomy LLM classifier
    noise.py             # noise floor + baseline_stats.json I/O
    regression.py        # history + regression detection
    markdown.py          # markdown report writer

tests/benchmark/
  __init__.py
  test_schema.py
  test_fingerprint.py
  test_runner.py
  test_jsonl.py
  test_profiles_registry.py
  test_profile_bare.py
  test_profile_prompted.py
  test_profile_tooled.py
  test_profile_solvay_full.py
  test_profile_solvay_noweb.py
  test_generator_skeleton.py
  test_generator_composer.py
  test_generator_verifier.py
  test_generator_probe.py
  test_generator_pipeline.py
  test_report_loader.py
  test_metric_a.py
  test_metric_b.py
  test_metric_d.py
  test_noise.py
  test_regression.py
  test_markdown.py
  test_cli.py

benchmark/
  problems/
    mechanics/           # 20 generated problems (Phase 4)
  verifications/         # optional per-problem scripts (empty initially)
  runs/                  # created at runtime
  reports/               # created at runtime
  baseline_stats.json    # created in Phase 5
  README.md              # created in Phase 6
```

### Modified files

- `pyproject.toml` — new entry point `solvay-bench = "solvay.benchmark.cli:app"`
- `benchmark/problems/sample-mechanics-1.json` — migrated to new schema → moved under `benchmark/problems/mechanics/`
- `benchmark/run_bench.py` — deleted; replaced by new subpackage
- `src/solvay/cli.py` — `bench` command deleted (superseded by dedicated CLI)
- `tests/test_cli.py` — drop assertions on the old `bench` command
- `CLAUDE.md` — document new structure (Phase 6)
- `FUTURE.md` — confirm that "tracking longitudinal" is removed (now in v1)

---

## Phase 0 — Foundations

### Task 1: Benchmark subpackage scaffolding + CLI entry point

**Files:**
- Create: `src/solvay/benchmark/__init__.py`
- Create: `src/solvay/benchmark/cli.py`
- Create: `tests/benchmark/__init__.py`
- Create: `tests/benchmark/test_cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/benchmark/test_cli.py`:

```python
"""Smoke tests for the solvay-bench CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from solvay.benchmark.cli import app


def test_cli_app_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "solvay-bench" in result.stdout.lower() or "benchmark" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'solvay.benchmark'`.

- [ ] **Step 3: Create package init**

Create `src/solvay/benchmark/__init__.py`:

```python
"""Solvay benchmark suite: problem generation, execution, and reporting."""
```

Create `tests/benchmark/__init__.py` as an empty file.

- [ ] **Step 4: Create the CLI skeleton**

Create `src/solvay/benchmark/cli.py`:

```python
"""Typer CLI for the Solvay benchmark suite."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="solvay-bench",
    help="Generate, run, and report on the Solvay benchmark suite.",
    no_args_is_help=True,
)


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Wire the entry point**

Edit `pyproject.toml`, extend `[project.scripts]`:

```toml
[project.scripts]
solvay = "solvay.cli:app"
solvay-bench = "solvay.benchmark.cli:app"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv sync && uv run python -m pytest tests/benchmark/test_cli.py -v`
Expected: PASS.

- [ ] **Step 7: Confirm the binary is installed**

Run: `uv run solvay-bench --help`
Expected: help text mentioning "Solvay benchmark suite".

- [ ] **Step 8: Commit**

```bash
git add src/solvay/benchmark/__init__.py src/solvay/benchmark/cli.py \
        tests/benchmark/__init__.py tests/benchmark/test_cli.py pyproject.toml
git commit -m "feat(bench): scaffold benchmark subpackage and solvay-bench CLI"
```

---

### Task 2: Problem schema module

**Files:**
- Create: `src/solvay/benchmark/schema.py`
- Create: `tests/benchmark/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_schema.py`:

```python
"""Tests for the benchmark problem schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from solvay.benchmark.schema import Problem, load_problem, load_problems_dir


def _valid_payload() -> dict:
    return {
        "id": "synth-mech-0001",
        "version": 1,
        "source": {
            "kind": "synthetic",
            "origin": "solvay-generator-v0.1",
            "generated_at": "2026-04-18T14:00:00Z",
            "seed": 42,
        },
        "domain": "mechanics",
        "subdomain": "kinematics",
        "tags": ["incline", "newton"],
        "difficulty": "intermediate",
        "statement": "A block slides down a frictionless incline of angle 30 deg.",
        "given": {"m": "2 kg", "theta": "30 deg"},
        "find": "Acceleration of the block.",
        "expected": {
            "kind": "numeric",
            "value": "4.905",
            "unit": "m/s^2",
            "tolerance_rel": 0.01,
            "verification": {"method": "numeric_eval", "script": None},
        },
        "contamination": {
            "checked_at": "2026-04-18T14:05:00Z",
            "method": "model-recall-probe-v1",
            "score": 0.02,
            "verdict": "clean",
        },
        "notes": None,
    }


def test_problem_parses_valid_payload() -> None:
    problem = Problem.model_validate(_valid_payload())
    assert problem.id == "synth-mech-0001"
    assert problem.source.kind == "synthetic"
    assert problem.expected.kind == "numeric"
    assert problem.expected.tolerance_rel == 0.01


def test_problem_rejects_invalid_kind() -> None:
    payload = _valid_payload()
    payload["source"]["kind"] = "made-up"
    with pytest.raises(ValidationError):
        Problem.model_validate(payload)


def test_problem_rejects_invalid_expected_kind() -> None:
    payload = _valid_payload()
    payload["expected"]["kind"] = "essay"
    with pytest.raises(ValidationError):
        Problem.model_validate(payload)


def test_load_problem_roundtrip(tmp_path: Path) -> None:
    payload = _valid_payload()
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    problem = load_problem(path)
    assert problem.id == payload["id"]


def test_load_problems_dir_recursive(tmp_path: Path) -> None:
    mech = tmp_path / "mechanics"
    mech.mkdir()
    payload = _valid_payload()
    (mech / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    payload2 = {**payload, "id": "synth-mech-0002"}
    (mech / "b.json").write_text(json.dumps(payload2), encoding="utf-8")
    problems = load_problems_dir(tmp_path)
    assert sorted(p.id for p in problems) == ["synth-mech-0001", "synth-mech-0002"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_schema.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement the schema**

Create `src/solvay/benchmark/schema.py`:

```python
"""Pydantic models and JSON IO for benchmark problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["synthetic", "olympiad", "textbook"]
    origin: str
    generated_at: str
    seed: int | None = None


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["sympy_equivalence", "numeric_eval", "dimensional_only"]
    script: str | None = None


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["symbolic", "numeric"]
    value: str
    unit: str | None = None
    tolerance_rel: float = 0.01
    verification: Verification


class Contamination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checked_at: str
    method: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["clean", "suspect", "contaminated"]


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: int
    source: Source
    domain: str
    subdomain: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "intermediate", "hard"] = "intermediate"
    statement: str
    given: dict[str, str] = Field(default_factory=dict)
    find: str
    expected: Expected
    contamination: Contamination | None = None
    notes: str | None = None


def load_problem(path: Path) -> Problem:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Problem.model_validate(data)


def load_problems_dir(root: Path) -> list[Problem]:
    problems: list[Problem] = []
    for json_path in sorted(Path(root).rglob("*.json")):
        problems.append(load_problem(json_path))
    return problems


def dump_problem(problem: Problem, path: Path) -> None:
    path.write_text(problem.model_dump_json(indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_schema.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/schema.py tests/benchmark/test_schema.py
git commit -m "feat(bench): add Problem pydantic schema with JSON IO"
```

---

### Task 3: Migrate the sample problem to the new schema

**Files:**
- Create: `benchmark/problems/mechanics/sample-mechanics-0001.json`
- Delete: `benchmark/problems/sample-mechanics-1.json`

- [ ] **Step 1: Write the migrated problem**

Create `benchmark/problems/mechanics/sample-mechanics-0001.json`:

```json
{
  "id": "sample-mech-0001",
  "version": 1,
  "source": {
    "kind": "textbook",
    "origin": "tipler-6th-ed-4.1",
    "generated_at": "2026-04-18T00:00:00Z",
    "seed": null
  },
  "domain": "mechanics",
  "subdomain": "newtonian-dynamics",
  "tags": ["incline", "newton", "1d"],
  "difficulty": "easy",
  "statement": "A block of mass 2 kg slides down a frictionless incline of angle 30 degrees. Find the acceleration of the block.",
  "given": {
    "m": "2 kg",
    "theta": "30 deg"
  },
  "find": "Acceleration of the block along the incline.",
  "expected": {
    "kind": "numeric",
    "value": "4.905",
    "unit": "m/s^2",
    "tolerance_rel": 0.01,
    "verification": {
      "method": "numeric_eval",
      "script": null
    }
  },
  "contamination": {
    "checked_at": "2026-04-18T00:00:00Z",
    "method": "manual-seed",
    "score": 1.0,
    "verdict": "contaminated"
  },
  "notes": "Canonical Tipler problem; marked contaminated on purpose to exercise the pipeline."
}
```

- [ ] **Step 2: Verify the new problem loads**

Run: `uv run python -c "from solvay.benchmark.schema import load_problem; from pathlib import Path; print(load_problem(Path('benchmark/problems/mechanics/sample-mechanics-0001.json')).id)"`
Expected output: `sample-mech-0001`.

- [ ] **Step 3: Delete the old flat problem file**

Run: `git rm benchmark/problems/sample-mechanics-1.json`

- [ ] **Step 4: Commit**

```bash
git add benchmark/problems/mechanics/sample-mechanics-0001.json
git commit -m "chore(bench): migrate sample problem to new schema"
```

---

### Task 4: BenchConfig

**Files:**
- Create: `src/solvay/benchmark/config.py`
- Create: `tests/benchmark/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_config.py`:

```python
"""Tests for BenchConfig."""

from __future__ import annotations

from solvay.benchmark.config import BenchConfig


def test_bench_config_defaults() -> None:
    cfg = BenchConfig()
    assert cfg.composer_model
    assert cfg.probe_model
    assert cfg.error_classifier_model
    assert cfg.cost_guard_threshold == 100


def test_bench_config_overrides() -> None:
    cfg = BenchConfig(composer_model="openai:gpt-5", cost_guard_threshold=500)
    assert cfg.composer_model == "openai:gpt-5"
    assert cfg.cost_guard_threshold == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement BenchConfig**

Create `src/solvay/benchmark/config.py`:

```python
"""Benchmark-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass

from solvay.config import DEFAULT_MODEL


@dataclass(frozen=True)
class BenchConfig:
    """Configuration for the benchmark suite.

    All LLM-consuming components of the benchmark read their model strings
    from here. Kept separate from :class:`solvay.config.SolvayConfig` because
    these knobs are meta-uses of LLMs (generation, evaluation) rather than
    solver-time choices.
    """

    composer_model: str = DEFAULT_MODEL
    probe_model: str = DEFAULT_MODEL
    error_classifier_model: str = DEFAULT_MODEL
    composer_max_attempts: int = 3
    cost_guard_threshold: int = 100
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_config.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/config.py tests/benchmark/test_config.py
git commit -m "feat(bench): add BenchConfig for generator and reporter LLM knobs"
```

---

### Task 5: Fingerprint module

**Files:**
- Create: `src/solvay/benchmark/fingerprint.py`
- Create: `tests/benchmark/test_fingerprint.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_fingerprint.py`:

```python
"""Tests for the Solvay config fingerprint."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.fingerprint import compute_fingerprint
from solvay.config import SolvayConfig


def test_fingerprint_is_deterministic() -> None:
    cfg = SolvayConfig()
    a = compute_fingerprint(cfg)
    b = compute_fingerprint(cfg)
    assert a == b
    assert a.startswith("sha256:")


def test_fingerprint_changes_with_model_override() -> None:
    base = SolvayConfig()
    other = SolvayConfig(default_model="openai:gpt-4o")
    assert compute_fingerprint(base) != compute_fingerprint(other)


def test_fingerprint_changes_with_prompt_edit(tmp_path: Path, monkeypatch) -> None:
    # Copy the real prompts dir to tmp, tweak one file, point the resolver there.
    import shutil

    import solvay.subagents as subagents

    real_prompts = Path(subagents.__file__).parent.parent / "prompts"
    fake_prompts = tmp_path / "prompts"
    shutil.copytree(real_prompts, fake_prompts)

    cfg = SolvayConfig()
    monkeypatch.setattr("solvay.benchmark.fingerprint._PROMPTS_DIR", fake_prompts)
    before = compute_fingerprint(cfg)

    target = fake_prompts / "solver.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nMORE TEXT\n", encoding="utf-8")
    after = compute_fingerprint(cfg)

    assert before != after
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_fingerprint.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the fingerprint module**

Create `src/solvay/benchmark/fingerprint.py`:

```python
"""Deterministic behavior fingerprint for a Solvay configuration.

Two runs with the same fingerprint are considered "the same Solvay" for
regression-comparison purposes. The fingerprint changes if and only if
something that affects model output changes: prompt text, subagent roster,
model-by-role mapping, or tool module identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from solvay.config import SolvayConfig

_PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"
_SUBAGENT_ROSTER: tuple[str, ...] = (
    "orchestrator",
    "parser",
    "researcher",
    "solver",
    "verifier",
    "peer_reviewer",
    "consolidator",
)
_TOOL_MODULES: tuple[str, ...] = (
    "solvay.tools.dimensional",
    "solvay.tools.python_exec",
    "solvay.tools.url_fetch",
)


def _hash_prompts() -> str:
    h = hashlib.sha256()
    for md in sorted(_PROMPTS_DIR.glob("*.md")):
        h.update(md.name.encode("utf-8"))
        h.update(b"\0")
        h.update(md.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _hash_tools() -> str:
    h = hashlib.sha256()
    for mod in _TOOL_MODULES:
        pkg, _, name = mod.rpartition(".")
        path = Path(__file__).resolve().parent.parent / "tools" / f"{name}.py"
        if path.exists():
            h.update(mod.encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def compute_fingerprint(config: SolvayConfig) -> str:
    """Return a sha256 fingerprint of the runtime behavior surface."""
    model_map = {role: config.model_for(role) for role in _SUBAGENT_ROSTER}
    payload = {
        "prompts_hash": _hash_prompts(),
        "tools_hash": _hash_tools(),
        "subagents": list(_SUBAGENT_ROSTER),
        "model_map": model_map,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_fingerprint.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/fingerprint.py tests/benchmark/test_fingerprint.py
git commit -m "feat(bench): compute deterministic config fingerprint"
```

---

## Phase 1 — Runner infrastructure

### Task 6: Profile dataclass and registry

**Files:**
- Create: `src/solvay/benchmark/profiles/__init__.py`
- Create: `tests/benchmark/test_profiles_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_profiles_registry.py`:

```python
"""Tests for the profile registry."""

from __future__ import annotations

import pytest

from solvay.benchmark.profiles import (
    PROFILES,
    Profile,
    ProfileResult,
    get_profile,
    register,
)


def test_profile_result_fields() -> None:
    r = ProfileResult(
        answer_raw="42",
        answer_extracted="42",
        elapsed_seconds=0.1,
        tokens={"input": 1, "output": 1, "cache_read": 0},
        trace_id=None,
        error=None,
    )
    assert r.answer_raw == "42"


def test_register_adds_profile() -> None:
    def runner(problem, model, config):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    p = Profile(name="unit-test", description="desc", runner=runner, cost_estimate="low")
    register(p)
    try:
        assert get_profile("unit-test") is p
    finally:
        PROFILES.pop("unit-test", None)


def test_get_profile_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_profiles_registry.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the registry**

Create `src/solvay/benchmark/profiles/__init__.py`:

```python
"""Profile registry: encapsulate the different experimental conditions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Problem


@dataclass(frozen=True)
class ProfileResult:
    """Raw output of a single (problem, profile, model) invocation."""

    answer_raw: str
    answer_extracted: str | None
    elapsed_seconds: float
    tokens: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None
    error: str | None = None


ProfileRunner = Callable[[Problem, str, BenchConfig], ProfileResult]


@dataclass(frozen=True)
class Profile:
    """What apparatus runs against a problem."""

    name: str
    description: str
    runner: ProfileRunner
    cost_estimate: Literal["low", "medium", "high"]


PROFILES: dict[str, Profile] = {}


def register(profile: Profile) -> None:
    PROFILES[profile.name] = profile


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile: {name!r}. Known: {sorted(PROFILES)}")
    return PROFILES[name]


def all_profiles() -> list[Profile]:
    return list(PROFILES.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_profiles_registry.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/profiles/__init__.py tests/benchmark/test_profiles_registry.py
git commit -m "feat(bench): add Profile dataclass and registry"
```

---

### Task 7: JSONL run writer and reader

**Files:**
- Create: `src/solvay/benchmark/jsonl.py`
- Create: `tests/benchmark/test_jsonl.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_jsonl.py`:

```python
"""Tests for the JSONL run writer/reader."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter, read_run


def _header() -> RunHeader:
    return RunHeader(
        run_id="2026-04-18-test",
        solvay_version="0.1.0",
        git_commit="abc1234",
        git_dirty=False,
        config_fingerprint="sha256:deadbeef",
        started_at="2026-04-18T00:00:00Z",
    )


def _record(problem_id: str = "p1") -> RunRecord:
    return RunRecord(
        problem_id=problem_id,
        profile="bare",
        model="anthropic:claude-sonnet-4-6",
        repeat_idx=0,
        answer_raw="42",
        answer_extracted="42",
        correct=True,
        elapsed_seconds=0.5,
        tokens={"input": 100, "output": 50, "cache_read": 0},
        trace_id=None,
        error=None,
    )


def test_writer_emits_header_then_records(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = RunWriter(path, _header())
    writer.write(_record("p1"))
    writer.write(_record("p2"))
    writer.close()

    header, records = read_run(path)
    assert header.run_id == "2026-04-18-test"
    assert [r.problem_id for r in records] == ["p1", "p2"]


def test_writer_is_append_safe(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    w = RunWriter(path, _header())
    w.write(_record("p1"))
    w.close()

    # Re-open to append: keep the same header, do not rewrite it.
    w2 = RunWriter(path, _header(), append=True)
    w2.write(_record("p2"))
    w2.close()

    _, records = read_run(path)
    assert [r.problem_id for r in records] == ["p1", "p2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_jsonl.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the JSONL module**

Create `src/solvay/benchmark/jsonl.py`:

```python
"""JSONL writer/reader for benchmark runs.

Format: the first line of every run file is a header record tagged
``"kind": "header"``. Every subsequent line is a data record tagged
``"kind": "record"``. Both types are self-describing so the file is
forward-compatible with future fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class RunHeader:
    run_id: str
    solvay_version: str
    git_commit: str
    git_dirty: bool
    config_fingerprint: str
    started_at: str


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


class RunWriter:
    """Append-safe JSONL writer. Writes header on first open if file is new."""

    def __init__(self, path: Path, header: RunHeader, append: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        mode = "a" if append or not is_new else "w"
        self._fp: TextIO = self._path.open(mode, encoding="utf-8")
        if is_new:
            self._fp.write(json.dumps({"kind": "header", **asdict(header)}) + "\n")
            self._fp.flush()

    def write(self, record: RunRecord) -> None:
        self._fp.write(json.dumps({"kind": "record", **asdict(record)}) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_run(path: Path) -> tuple[RunHeader, list[RunRecord]]:
    header: RunHeader | None = None
    records: list[RunRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.pop("kind")
        if kind == "header":
            header = RunHeader(**obj)
        elif kind == "record":
            records.append(RunRecord(**obj))
    if header is None:
        raise ValueError(f"No header found in {path}")
    return header, records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_jsonl.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/jsonl.py tests/benchmark/test_jsonl.py
git commit -m "feat(bench): JSONL writer/reader with versioned header"
```

---

### Task 8: Matrix runner

**Files:**
- Create: `src/solvay/benchmark/runner.py`
- Create: `tests/benchmark/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_runner.py`:

```python
"""Tests for the matrix runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.jsonl import read_run
from solvay.benchmark.profiles import PROFILES, Profile, ProfileResult, register
from solvay.benchmark.runner import MatrixSpec, run_matrix
from solvay.benchmark.schema import Problem


def _fake_problem(pid: str = "p1") -> Problem:
    payload = {
        "id": pid,
        "version": 1,
        "source": {
            "kind": "synthetic",
            "origin": "test",
            "generated_at": "2026-04-18T00:00:00Z",
            "seed": 0,
        },
        "domain": "mechanics",
        "subdomain": None,
        "tags": [],
        "difficulty": "easy",
        "statement": "what is one plus one?",
        "given": {},
        "find": "sum",
        "expected": {
            "kind": "numeric",
            "value": "2",
            "unit": None,
            "tolerance_rel": 0.01,
            "verification": {"method": "numeric_eval", "script": None},
        },
        "contamination": None,
        "notes": None,
    }
    return Problem.model_validate(payload)


@pytest.fixture(autouse=True)
def _register_fake_profile():
    def runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
        return ProfileResult(
            answer_raw=f"answer={problem.id}:{model}",
            answer_extracted="2",
            elapsed_seconds=0.01,
            tokens={"input": 1, "output": 1, "cache_read": 0},
        )

    register(Profile(name="fake", description="fake", runner=runner, cost_estimate="low"))
    yield
    PROFILES.pop("fake", None)


def test_run_matrix_emits_row_per_cell(tmp_path: Path) -> None:
    problems = [_fake_problem("p1"), _fake_problem("p2")]
    spec = MatrixSpec(
        problems=problems,
        profile_names=["fake"],
        models=["anthropic:claude-sonnet-4-6", "openai:gpt-4o"],
        repeats=2,
    )
    out = tmp_path / "run.jsonl"
    run_matrix(spec, out_path=out, config=BenchConfig())

    _, records = read_run(out)
    assert len(records) == 2 * 1 * 2 * 2
    assert all(r.profile == "fake" for r in records)


def test_run_matrix_correctness_uses_expected_value(tmp_path: Path) -> None:
    spec = MatrixSpec(
        problems=[_fake_problem()],
        profile_names=["fake"],
        models=["anthropic:claude-sonnet-4-6"],
        repeats=1,
    )
    out = tmp_path / "run.jsonl"
    run_matrix(spec, out_path=out, config=BenchConfig())
    _, records = read_run(out)
    assert records[0].correct is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_runner.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the runner**

Create `src/solvay/benchmark/runner.py`:

```python
"""Matrix runner over (problems × profiles × models × repeats)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import re
import subprocess
import sympy
from dataclasses import dataclass
from pathlib import Path

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
    except Exception:
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
    except Exception:
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
                        correct = (
                            False
                            if res.error
                            else _evaluate_correct(res.answer_raw, problem.expected)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_runner.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/runner.py tests/benchmark/test_runner.py
git commit -m "feat(bench): matrix runner over (problems × profiles × models × repeats)"
```

---

### Task 9: `bench run` CLI command (deferred profiles import)

**Files:**
- Modify: `src/solvay/benchmark/cli.py`
- Modify: `tests/benchmark/test_cli.py`

- [ ] **Step 1: Extend the failing test**

Append to `tests/benchmark/test_cli.py`:

```python
from pathlib import Path


def test_cli_run_invokes_matrix(tmp_path: Path, monkeypatch) -> None:
    from solvay.benchmark.profiles import PROFILES, Profile, ProfileResult, register

    def runner(problem, model, config):  # type: ignore[no-untyped-def]
        return ProfileResult(
            answer_raw="4.905",
            answer_extracted="4.905",
            elapsed_seconds=0.0,
            tokens={"input": 0, "output": 0, "cache_read": 0},
        )

    register(Profile(name="cli-fake", description="", runner=runner, cost_estimate="low"))
    out = tmp_path / "run.jsonl"
    runner_cli = CliRunner()
    result = runner_cli.invoke(
        app,
        [
            "run",
            "--profiles",
            "cli-fake",
            "--models",
            "anthropic:claude-sonnet-4-6",
            "--problems",
            "benchmark/problems",
            "--out",
            str(out),
        ],
    )
    PROFILES.pop("cli-fake", None)
    assert result.exit_code == 0, result.stdout
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_cli.py -v`
Expected: FAIL (the `run` command does not exist yet).

- [ ] **Step 3: Implement the `run` command**

Replace `src/solvay/benchmark/cli.py`:

```python
"""Typer CLI for the Solvay benchmark suite."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="solvay-bench",
    help="Generate, run, and report on the Solvay benchmark suite.",
    no_args_is_help=True,
)


@app.command("run")
def run_cmd(
    problems: Annotated[
        Path, typer.Option("--problems", help="Root directory of problem JSONs.")
    ] = Path("benchmark/problems"),
    profiles: Annotated[
        str,
        typer.Option(
            "--profiles",
            help="Comma-separated profile names, or 'all'.",
        ),
    ] = "solvay-full",
    models: Annotated[
        str,
        typer.Option(
            "--models",
            help="Comma-separated model strings for langchain.chat_models.init_chat_model.",
        ),
    ] = "",
    domain: Annotated[
        str | None,
        typer.Option("--domain", help="Filter problems to this domain (subdirectory)."),
    ] = None,
    repeat: Annotated[int, typer.Option("--repeat", min=1)] = 1,
    out: Annotated[
        Path | None, typer.Option("--out", help="Destination JSONL (default: benchmark/runs/<id>.jsonl).")
    ] = None,
    confirm_cost: Annotated[
        bool, typer.Option("--confirm-cost", help="Confirm running a matrix above the cost threshold.")
    ] = False,
) -> None:
    """Execute the (problems × profiles × models × repeats) matrix."""
    # Import locally so the module tree cost isn't paid on --help.
    from solvay.benchmark.config import BenchConfig
    from solvay.benchmark.profiles import PROFILES
    from solvay.benchmark.runner import MatrixSpec, run_matrix
    from solvay.benchmark.schema import load_problems_dir
    from solvay.config import DEFAULT_MODEL

    # Force import of the built-in profiles so they self-register. The try/except
    # lets earlier phases run before Phase 2 is merged.
    try:
        import solvay.benchmark.profiles.bare  # noqa: F401
        import solvay.benchmark.profiles.prompted  # noqa: F401
        import solvay.benchmark.profiles.solvay_full  # noqa: F401
        import solvay.benchmark.profiles.solvay_noweb  # noqa: F401
        import solvay.benchmark.profiles.tooled  # noqa: F401
    except ImportError:
        pass

    root = problems / domain if domain else problems
    all_problems = load_problems_dir(root)
    if not all_problems:
        typer.echo(f"No problems found under {root}.", err=True)
        raise typer.Exit(1)

    profile_names = list(PROFILES) if profiles == "all" else [p.strip() for p in profiles.split(",")]
    for pn in profile_names:
        if pn not in PROFILES:
            typer.echo(f"Unknown profile: {pn}. Known: {sorted(PROFILES)}", err=True)
            raise typer.Exit(2)

    model_list = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
    cfg = BenchConfig()
    invocations = len(all_problems) * len(profile_names) * len(model_list) * repeat
    if invocations > cfg.cost_guard_threshold and not confirm_cost:
        typer.echo(
            f"Matrix would make {invocations} invocations (threshold {cfg.cost_guard_threshold}). "
            "Pass --confirm-cost to proceed.",
            err=True,
        )
        raise typer.Exit(3)

    out = out or Path("benchmark/runs") / (
        dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S") + ".jsonl"
    )
    spec = MatrixSpec(
        problems=all_problems,
        profile_names=profile_names,
        models=model_list,
        repeats=repeat,
    )
    path = run_matrix(spec, out_path=out, config=cfg)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_cli.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/cli.py tests/benchmark/test_cli.py
git commit -m "feat(bench): add 'solvay-bench run' CLI with cost guard"
```

---

## Phase 2 — The five profiles

### Task 10: `bare` profile

**Files:**
- Create: `src/solvay/benchmark/profiles/bare.py`
- Create: `tests/benchmark/test_profile_bare.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_profile_bare.py`:

```python
"""Tests for the bare profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.bare import bare_runner  # noqa: F401
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    return Problem.model_validate(
        {
            "id": "p1",
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "t",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 0,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "Compute 1+1.",
            "given": {},
            "find": "sum",
            "expected": {
                "kind": "numeric",
                "value": "2",
                "unit": None,
                "tolerance_rel": 0.01,
                "verification": {"method": "numeric_eval", "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_bare_profile_registered() -> None:
    p = get_profile("bare")
    assert p.cost_estimate == "low"


def test_bare_profile_uses_user_only_message() -> None:
    fake_chat = MagicMock()
    response = MagicMock()
    response.content = "2"
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 1}
    fake_chat.invoke.return_value = response
    with patch("solvay.benchmark.profiles.bare.init_chat_model", return_value=fake_chat):
        result = get_profile("bare").runner(_problem(), "anthropic:claude-sonnet-4-6", BenchConfig())
    assert result.answer_raw == "2"
    call_args = fake_chat.invoke.call_args[0][0]
    # bare profile: a single user message, no system prompt
    assert len(call_args) == 1
    assert call_args[0]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_profile_bare.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the bare profile**

Create `src/solvay/benchmark/profiles/bare.py`:

```python
"""'bare' profile: raw model, no system prompt, no tools."""

from __future__ import annotations

import time

from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.schema import Problem


def bare_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    chat = init_chat_model(model, temperature=0)
    t0 = time.monotonic()
    try:
        response = chat.invoke([{"role": "user", "content": problem.statement}])
    except Exception as exc:  # noqa: BLE001
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    text = getattr(response, "content", "")
    usage = getattr(response, "usage_metadata", {}) or {}
    tokens = {
        "input": int(usage.get("input_tokens", 0) or 0),
        "output": int(usage.get("output_tokens", 0) or 0),
        "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
    }
    return ProfileResult(
        answer_raw=str(text),
        answer_extracted=None,
        elapsed_seconds=time.monotonic() - t0,
        tokens=tokens,
    )


register(
    Profile(
        name="bare",
        description="Raw model, no system prompt, no tools.",
        runner=bare_runner,
        cost_estimate="low",
    )
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_profile_bare.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/profiles/bare.py tests/benchmark/test_profile_bare.py
git commit -m "feat(bench): add 'bare' profile"
```

---

### Task 11: `prompted` profile

**Files:**
- Create: `src/solvay/benchmark/profiles/prompted.py`
- Create: `src/solvay/benchmark/profiles/system_prompt.py`
- Create: `tests/benchmark/test_profile_prompted.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_profile_prompted.py`:

```python
"""Tests for the prompted profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.prompted import prompted_runner  # noqa: F401
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_prompted_profile_registered() -> None:
    assert get_profile("prompted").cost_estimate == "low"


def test_prompted_profile_sends_system_prompt() -> None:
    fake_chat = MagicMock()
    response = MagicMock()
    response.content = "2 (dimensionless)"
    response.usage_metadata = {"input_tokens": 20, "output_tokens": 4}
    fake_chat.invoke.return_value = response
    with patch("solvay.benchmark.profiles.prompted.init_chat_model", return_value=fake_chat):
        get_profile("prompted").runner(_problem(), "anthropic:claude-sonnet-4-6", BenchConfig())
    call_args = fake_chat.invoke.call_args[0][0]
    assert call_args[0]["role"] == "system"
    assert "physicist" in call_args[0]["content"].lower()
    assert call_args[1]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_profile_prompted.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the shared system prompt and the profile**

Create `src/solvay/benchmark/profiles/system_prompt.py`:

```python
"""System prompt shared by 'prompted' and 'tooled' baseline profiles."""

PHYSICIST_SYSTEM_PROMPT = """\
You are a physicist. Solve the problem step by step.

Guidelines:
- Identify the physical principles that apply.
- Set up equations with named symbols before substituting numbers.
- Track units throughout; state the final units explicitly.
- Give the final numerical or symbolic answer clearly at the end.
"""
```

Create `src/solvay/benchmark/profiles/prompted.py`:

```python
"""'prompted' profile: model + physicist system prompt, no tools."""

from __future__ import annotations

import time

from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.profiles.system_prompt import PHYSICIST_SYSTEM_PROMPT
from solvay.benchmark.schema import Problem


def prompted_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    chat = init_chat_model(model, temperature=0)
    t0 = time.monotonic()
    try:
        response = chat.invoke(
            [
                {"role": "system", "content": PHYSICIST_SYSTEM_PROMPT},
                {"role": "user", "content": problem.statement},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    text = getattr(response, "content", "")
    usage = getattr(response, "usage_metadata", {}) or {}
    tokens = {
        "input": int(usage.get("input_tokens", 0) or 0),
        "output": int(usage.get("output_tokens", 0) or 0),
        "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
    }
    return ProfileResult(
        answer_raw=str(text),
        answer_extracted=None,
        elapsed_seconds=time.monotonic() - t0,
        tokens=tokens,
    )


register(
    Profile(
        name="prompted",
        description="Raw model + physicist system prompt, no tools.",
        runner=prompted_runner,
        cost_estimate="low",
    )
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_profile_prompted.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/profiles/prompted.py \
        src/solvay/benchmark/profiles/system_prompt.py \
        tests/benchmark/test_profile_prompted.py
git commit -m "feat(bench): add 'prompted' profile with shared physicist prompt"
```

---

### Task 12: `tooled` profile

**Files:**
- Create: `src/solvay/benchmark/profiles/tooled.py`
- Create: `tests/benchmark/test_profile_tooled.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_profile_tooled.py`:

```python
"""Tests for the tooled profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.tooled import tooled_runner  # noqa: F401


def _problem():
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_tooled_profile_registered() -> None:
    assert get_profile("tooled").cost_estimate == "medium"


def test_tooled_profile_wires_local_tools() -> None:
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [MagicMock(content="final: 2")],
    }
    with patch(
        "solvay.benchmark.profiles.tooled.create_react_agent",
        return_value=fake_agent,
    ) as create_react:
        result = get_profile("tooled").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    tools_arg = create_react.call_args.kwargs.get("tools") or create_react.call_args.args[1]
    tool_names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools_arg}
    assert "python_exec" in tool_names
    assert "check_dimensions" in tool_names
    assert result.answer_raw == "final: 2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_profile_tooled.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the tooled profile**

Create `src/solvay/benchmark/profiles/tooled.py`:

```python
"""'tooled' profile: single agent with local tools, no web, no orchestration."""

from __future__ import annotations

import time

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.profiles.system_prompt import PHYSICIST_SYSTEM_PROMPT
from solvay.benchmark.schema import Problem
from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec


def tooled_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    chat = init_chat_model(model, temperature=0)
    agent = create_react_agent(
        chat,
        tools=[python_exec, check_dimensions],
        prompt=PHYSICIST_SYSTEM_PROMPT,
    )
    t0 = time.monotonic()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": problem.statement}]})
    except Exception as exc:  # noqa: BLE001
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    final = result["messages"][-1]
    text = str(getattr(final, "content", ""))
    return ProfileResult(
        answer_raw=text,
        answer_extracted=None,
        elapsed_seconds=time.monotonic() - t0,
        tokens={"input": 0, "output": 0, "cache_read": 0},
    )


register(
    Profile(
        name="tooled",
        description="Single agent with sympy/python_exec/dimensional tools, no web.",
        runner=tooled_runner,
        cost_estimate="medium",
    )
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_profile_tooled.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/profiles/tooled.py tests/benchmark/test_profile_tooled.py
git commit -m "feat(bench): add 'tooled' profile (single agent with local tools)"
```

---

### Task 13: `solvay-full` profile

**Files:**
- Create: `src/solvay/benchmark/profiles/solvay_full.py`
- Create: `tests/benchmark/test_profile_solvay_full.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_profile_solvay_full.py`:

```python
"""Tests for the solvay-full profile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.solvay_full import solvay_full_runner  # noqa: F401


def _problem():
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_solvay_full_profile_registered() -> None:
    assert get_profile("solvay-full").cost_estimate == "high"


def test_solvay_full_profile_uses_full_agent() -> None:
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [MagicMock(content="2 m/s^2")]}
    with patch(
        "solvay.benchmark.profiles.solvay_full.create_solvay_agent",
        return_value=fake_agent,
    ) as create_full:
        result = get_profile("solvay-full").runner(
            _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
        )
    # The injected model must propagate to SolvayConfig.default_model
    cfg = create_full.call_args.args[0]
    assert cfg.default_model == "anthropic:claude-sonnet-4-6"
    assert cfg.persistence.enabled is False
    assert result.answer_raw == "2 m/s^2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_profile_solvay_full.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the solvay-full profile**

Create `src/solvay/benchmark/profiles/solvay_full.py`:

```python
"""'solvay-full' profile: complete Solvay system with web access."""

from __future__ import annotations

import time

from solvay.agent import create_solvay_agent
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.schema import Problem
from solvay.config import PersistenceConfig, SolvayConfig


def solvay_full_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    scfg = SolvayConfig(
        default_model=model,
        persistence=PersistenceConfig(enabled=False),
    )
    agent = create_solvay_agent(scfg)
    t0 = time.monotonic()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": problem.statement}]})
    except Exception as exc:  # noqa: BLE001
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    final = result["messages"][-1]
    return ProfileResult(
        answer_raw=str(getattr(final, "content", "")),
        answer_extracted=None,
        elapsed_seconds=time.monotonic() - t0,
        tokens={"input": 0, "output": 0, "cache_read": 0},
    )


register(
    Profile(
        name="solvay-full",
        description="Full Solvay system with web search enabled.",
        runner=solvay_full_runner,
        cost_estimate="high",
    )
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_profile_solvay_full.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/profiles/solvay_full.py tests/benchmark/test_profile_solvay_full.py
git commit -m "feat(bench): add 'solvay-full' profile"
```

---

### Task 14: `solvay-noweb` profile

**Files:**
- Create: `src/solvay/benchmark/profiles/solvay_noweb.py`
- Create: `tests/benchmark/test_profile_solvay_noweb.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_profile_solvay_noweb.py`:

```python
"""Tests for the solvay-noweb profile."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import get_profile
from solvay.benchmark.profiles.solvay_noweb import solvay_noweb_runner  # noqa: F401


def _problem():
    from tests.benchmark.test_profile_bare import _problem as bare_problem

    return bare_problem()


def test_solvay_noweb_registered() -> None:
    assert get_profile("solvay-noweb").cost_estimate == "high"


def test_solvay_noweb_unsets_tavily_during_call(monkeypatch) -> None:
    seen: dict[str, str | None] = {}

    def fake_create(scfg):  # type: ignore[no-untyped-def]
        seen["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [MagicMock(content="0")]}
        return agent

    monkeypatch.setenv("TAVILY_API_KEY", "PRESENT")
    with patch("solvay.benchmark.profiles.solvay_noweb.create_solvay_agent", side_effect=fake_create):
        get_profile("solvay-noweb").runner(_problem(), "anthropic:claude-sonnet-4-6", BenchConfig())
    assert seen["TAVILY_API_KEY"] is None
    # Restored after the call:
    assert os.environ.get("TAVILY_API_KEY") == "PRESENT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_profile_solvay_noweb.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the solvay-noweb profile**

Create `src/solvay/benchmark/profiles/solvay_noweb.py`:

```python
"""'solvay-noweb' profile: full Solvay with Tavily web search disabled."""

from __future__ import annotations

import os
import time

from solvay.agent import create_solvay_agent
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.schema import Problem
from solvay.config import PersistenceConfig, SolvayConfig


def solvay_noweb_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    saved = os.environ.pop("TAVILY_API_KEY", None)
    try:
        scfg = SolvayConfig(
            default_model=model,
            persistence=PersistenceConfig(enabled=False),
        )
        agent = create_solvay_agent(scfg)
        t0 = time.monotonic()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": problem.statement}]}
            )
        except Exception as exc:  # noqa: BLE001
            return ProfileResult(
                answer_raw="",
                answer_extracted=None,
                elapsed_seconds=time.monotonic() - t0,
                error=str(exc),
            )
        final = result["messages"][-1]
        return ProfileResult(
            answer_raw=str(getattr(final, "content", "")),
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            tokens={"input": 0, "output": 0, "cache_read": 0},
        )
    finally:
        if saved is not None:
            os.environ["TAVILY_API_KEY"] = saved


register(
    Profile(
        name="solvay-noweb",
        description="Full Solvay system with web search disabled (feeds metric A).",
        runner=solvay_noweb_runner,
        cost_estimate="high",
    )
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_profile_solvay_noweb.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/profiles/solvay_noweb.py tests/benchmark/test_profile_solvay_noweb.py
git commit -m "feat(bench): add 'solvay-noweb' profile (drives metric A)"
```

---

## Phase 3 — Generation pipeline

### Task 15: Generation skeleton + structured output schema

**Files:**
- Create: `src/solvay/benchmark/generator/__init__.py`
- Create: `src/solvay/benchmark/generator/skeleton.py`
- Create: `tests/benchmark/test_generator_skeleton.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_generator_skeleton.py`:

```python
"""Tests for the generation skeleton."""

from __future__ import annotations

from solvay.benchmark.generator.skeleton import GenerationSkeleton


def test_skeleton_defaults() -> None:
    sk = GenerationSkeleton(domain="mechanics", compose=["pendulum"])
    assert sk.approach == "backward"
    assert sk.difficulty == "intermediate"


def test_skeleton_rejects_empty_compose() -> None:
    import pytest

    with pytest.raises(ValueError):
        GenerationSkeleton(domain="mechanics", compose=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_generator_skeleton.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the skeleton**

Create `src/solvay/benchmark/generator/__init__.py`:

```python
"""Synthetic benchmark problem generation pipeline."""
```

Create `src/solvay/benchmark/generator/skeleton.py`:

```python
"""Input specification for a single generation attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GenerationSkeleton:
    domain: str
    compose: list[str]
    subdomain: str | None = None
    difficulty: Literal["easy", "intermediate", "hard"] = "intermediate"
    approach: Literal["backward", "forward"] = "backward"
    seed: int | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.compose:
            raise ValueError("compose must contain at least one concept")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_generator_skeleton.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/generator/__init__.py \
        src/solvay/benchmark/generator/skeleton.py \
        tests/benchmark/test_generator_skeleton.py
git commit -m "feat(bench): add GenerationSkeleton input spec"
```

---

### Task 16: Composer (LLM-driven problem authoring)

**Files:**
- Create: `src/solvay/benchmark/generator/composer.py`
- Create: `tests/benchmark/test_generator_composer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_generator_composer.py`:

```python
"""Tests for the problem composer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.composer import compose_problem
from solvay.benchmark.generator.skeleton import GenerationSkeleton


_VALID_JSON = """
{
  "id": "synth-mech-TEST",
  "version": 1,
  "source": {
    "kind": "synthetic",
    "origin": "solvay-generator-v0.1",
    "generated_at": "2026-04-18T00:00:00Z",
    "seed": 42
  },
  "domain": "mechanics",
  "subdomain": "kinematics",
  "tags": ["pendulum"],
  "difficulty": "intermediate",
  "statement": "A pendulum...",
  "given": {"L": "1 m", "g": "9.81 m/s^2"},
  "find": "Angular frequency.",
  "expected": {
    "kind": "symbolic",
    "value": "sqrt(g/L)",
    "unit": "rad/s",
    "tolerance_rel": 0.01,
    "verification": {"method": "sympy_equivalence", "script": null}
  },
  "contamination": null,
  "notes": null
}
"""


def test_compose_returns_problem() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.return_value = MagicMock(content=_VALID_JSON)
    with patch("solvay.benchmark.generator.composer.init_chat_model", return_value=fake_chat):
        problem = compose_problem(
            GenerationSkeleton(domain="mechanics", compose=["pendulum"]),
            BenchConfig(),
        )
    assert problem.id == "synth-mech-TEST"
    assert problem.expected.value == "sqrt(g/L)"


def test_compose_retries_on_bad_json() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.side_effect = [
        MagicMock(content="not json"),
        MagicMock(content=_VALID_JSON),
    ]
    with patch("solvay.benchmark.generator.composer.init_chat_model", return_value=fake_chat):
        problem = compose_problem(
            GenerationSkeleton(domain="mechanics", compose=["pendulum"]),
            BenchConfig(composer_max_attempts=3),
        )
    assert problem.id == "synth-mech-TEST"
    assert fake_chat.invoke.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_generator_composer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the composer**

Create `src/solvay/benchmark/generator/composer.py`:

```python
"""LLM-driven authoring of benchmark problems."""

from __future__ import annotations

import json
import re

from langchain.chat_models import init_chat_model
from pydantic import ValidationError

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.skeleton import GenerationSkeleton
from solvay.benchmark.schema import Problem


_COMPOSER_SYSTEM = """\
You are a physics problem-writer for a benchmark that measures reasoning capacity.
Your job: produce ONE problem that composes the listed concepts in a way that does
not appear in any standard textbook. The problem MUST be answerable in closed form
or with a numerical integration.

Requirements:
- Emit ONLY a single JSON object matching the target schema. No prose, no markdown,
  no code fences.
- When approach="backward", first pick the closed-form answer, then construct a
  statement that leads to it. This is the default — use it whenever feasible.
- When approach="forward", compose the statement freely, then solve it yourself and
  include the verified answer.
- `expected.value` must be a sympy-parseable expression (for `kind=symbolic`) or a
  plain number (for `kind=numeric`).
- `expected.verification.method` is one of `sympy_equivalence`, `numeric_eval`, or
  `dimensional_only`.
- Do NOT set a `contamination` field — that is filled in later by the probe.
- Use `id` of the form `synth-<domain-short>-<4-digit-hex>` using the provided seed.
"""


_SCHEMA_EXAMPLE = """\
{
  "id": "synth-mech-0001",
  "version": 1,
  "source": {"kind": "synthetic", "origin": "solvay-generator-v0.1",
             "generated_at": "2026-04-18T00:00:00Z", "seed": 42},
  "domain": "mechanics",
  "subdomain": "rigid-body-dynamics",
  "tags": ["pendulum", "lorentz"],
  "difficulty": "intermediate",
  "statement": "...",
  "given": {"L": "1 m"},
  "find": "...",
  "expected": {
    "kind": "symbolic",
    "value": "sqrt(g/L)",
    "unit": "rad/s",
    "tolerance_rel": 0.01,
    "verification": {"method": "sympy_equivalence", "script": null}
  },
  "contamination": null,
  "notes": null
}
"""


def _extract_json(text: str) -> str:
    # Strip code fences if the model used them anyway.
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    return stripped


def compose_problem(skeleton: GenerationSkeleton, config: BenchConfig) -> Problem:
    """Author one problem. Retries up to ``composer_max_attempts`` on parse/validation failures."""
    chat = init_chat_model(config.composer_model, temperature=0.7)
    user = (
        f"Skeleton:\n{json.dumps(skeleton.__dict__, default=list, indent=2)}\n\n"
        f"Target schema example:\n{_SCHEMA_EXAMPLE}\n"
    )

    last_error: str | None = None
    for _ in range(config.composer_max_attempts):
        messages = [
            {"role": "system", "content": _COMPOSER_SYSTEM},
            {"role": "user", "content": user + (f"\n\nPrevious attempt failed: {last_error}\nFix it." if last_error else "")},
        ]
        response = chat.invoke(messages)
        raw = _extract_json(str(getattr(response, "content", "")))
        try:
            payload = json.loads(raw)
            return Problem.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:400]
            continue
    raise RuntimeError(f"Composer failed after {config.composer_max_attempts} attempts: {last_error}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_generator_composer.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/generator/composer.py tests/benchmark/test_generator_composer.py
git commit -m "feat(bench): add LLM-driven problem composer"
```

---

### Task 17: Verifier (sympy + numeric)

**Files:**
- Create: `src/solvay/benchmark/generator/verifier.py`
- Create: `tests/benchmark/test_generator_verifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_generator_verifier.py`:

```python
"""Tests for the symbolic/numeric verifier."""

from __future__ import annotations

from solvay.benchmark.generator.verifier import VerifyOutcome, verify_problem
from solvay.benchmark.schema import Problem


def _make(value: str, method: str, kind: str = "symbolic", unit: str | None = None) -> Problem:
    return Problem.model_validate(
        {
            "id": "t",
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "t",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 0,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "a",
            "given": {"g": "9.81", "L": "1"},
            "find": "omega",
            "expected": {
                "kind": kind,
                "value": value,
                "unit": unit,
                "tolerance_rel": 0.01,
                "verification": {"method": method, "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_sympy_equivalence_accepts_parseable_answer() -> None:
    p = _make("sqrt(g/L)", "sympy_equivalence", kind="symbolic")
    outcome = verify_problem(p)
    assert outcome.ok
    assert outcome.method == "sympy_equivalence"


def test_sympy_equivalence_rejects_unparseable_answer() -> None:
    p = _make("this is not sympy", "sympy_equivalence", kind="symbolic")
    outcome = verify_problem(p)
    assert not outcome.ok


def test_numeric_eval_passes_when_value_is_number() -> None:
    p = _make("3.13", "numeric_eval", kind="numeric")
    outcome = verify_problem(p)
    assert outcome.ok


def test_dimensional_only_passes_when_unit_is_known() -> None:
    p = _make("g / L", "dimensional_only", kind="symbolic", unit="1/second**2")
    outcome = verify_problem(p)
    assert outcome.ok


def test_dimensional_only_rejects_unknown_unit() -> None:
    p = _make("g / L", "dimensional_only", kind="symbolic", unit="made_up_unit")
    outcome = verify_problem(p)
    assert not outcome.ok
    assert isinstance(outcome, VerifyOutcome)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_generator_verifier.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the verifier**

Create `src/solvay/benchmark/generator/verifier.py`:

```python
"""Deterministic verification of generator output.

This verifier does NOT re-solve the problem from the statement (that would
require NLP-level understanding). Instead it verifies the *coherence* of the
generator's own output: that the answer is parseable in the declared
mode, that the declared units exist, and that numeric answers are
within tolerance of themselves when re-evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.physics.units import convert_to  # noqa: F401  (kept for future method)
from sympy.physics.units.systems import SI

from solvay.benchmark.schema import Problem


@dataclass(frozen=True)
class VerifyOutcome:
    ok: bool
    method: str
    reason: str | None = None


def _sympify_with_given(expr: str, given: dict[str, str]) -> sympy.Expr:
    namespace = {k: sympy.Symbol(k) for k in given}
    namespace["sqrt"] = sympy.sqrt
    return sympy.sympify(expr, locals=namespace)


def _unit_is_known(unit_str: str) -> bool:
    ns: dict[str, object] = {}
    for u in SI.get_units_non_prefixed():
        ns[str(u)] = u
    ns["sqrt"] = sympy.sqrt
    try:
        sympy.sympify(unit_str, locals=ns, evaluate=False)
    except Exception:
        return False
    return True


def verify_problem(problem: Problem) -> VerifyOutcome:
    method = problem.expected.verification.method
    try:
        if method == "sympy_equivalence":
            _sympify_with_given(problem.expected.value, problem.given)
            return VerifyOutcome(ok=True, method=method)
        if method == "numeric_eval":
            float(sympy.sympify(problem.expected.value).evalf())
            return VerifyOutcome(ok=True, method=method)
        if method == "dimensional_only":
            if problem.expected.unit is None:
                return VerifyOutcome(ok=False, method=method, reason="unit missing")
            if not _unit_is_known(problem.expected.unit):
                return VerifyOutcome(ok=False, method=method, reason=f"unknown unit: {problem.expected.unit}")
            return VerifyOutcome(ok=True, method=method)
    except Exception as exc:  # noqa: BLE001
        return VerifyOutcome(ok=False, method=method, reason=str(exc)[:200])
    return VerifyOutcome(ok=False, method=method, reason=f"unknown method: {method}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_generator_verifier.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/generator/verifier.py tests/benchmark/test_generator_verifier.py
git commit -m "feat(bench): add sympy+numeric+dimensional verifier"
```

---

### Task 18: Contamination probe

**Files:**
- Create: `src/solvay/benchmark/generator/probe.py`
- Create: `tests/benchmark/test_generator_probe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_generator_probe.py`:

```python
"""Tests for the contamination probe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.probe import probe_contamination
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    return Problem.model_validate(
        {
            "id": "t",
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "t",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 0,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "Strange unique problem X.",
            "given": {},
            "find": "x",
            "expected": {
                "kind": "numeric",
                "value": "1",
                "unit": None,
                "tolerance_rel": 0.01,
                "verification": {"method": "numeric_eval", "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_probe_returns_clean_when_model_denies_recognition() -> None:
    fake_chat = MagicMock()
    # Three responses: recognition denial, mismatched continuation, wrong cold answer
    fake_chat.invoke.side_effect = [
        MagicMock(content='{"recognized": false, "source": null}'),
        MagicMock(content='{"continuation": "completely different text"}'),
        MagicMock(content='{"answer": "0"}'),
    ]
    with patch("solvay.benchmark.generator.probe.init_chat_model", return_value=fake_chat):
        result = probe_contamination(_problem(), BenchConfig())
    assert result.verdict == "clean"
    assert 0.0 <= result.score < 0.34


def test_probe_returns_contaminated_when_model_cites_source() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.side_effect = [
        MagicMock(content='{"recognized": true, "source": "Tipler 4.1"}'),
        MagicMock(content='{"continuation": "Strange unique problem X."}'),
        MagicMock(content='{"answer": "1"}'),
    ]
    with patch("solvay.benchmark.generator.probe.init_chat_model", return_value=fake_chat):
        result = probe_contamination(_problem(), BenchConfig())
    assert result.verdict == "contaminated"
    assert result.score > 0.66
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_generator_probe.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the probe**

Create `src/solvay/benchmark/generator/probe.py`:

```python
"""Contamination probe: estimate how likely the target model has seen the problem."""

from __future__ import annotations

import datetime as dt
import difflib
import json
import re

from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Contamination, Problem

PROBE_METHOD_ID = "model-recall-probe-v1"


def _parse_json(text: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _recognition_probe(chat, problem: Problem) -> float:
    msg = (
        "Do you recognize the following physics problem? If yes, name the source "
        "(textbook, competition, year). Answer in JSON with keys 'recognized' "
        "(bool) and 'source' (string or null).\n\n"
        f"Problem: {problem.statement}"
    )
    resp = chat.invoke([{"role": "user", "content": msg}])
    obj = _parse_json(str(getattr(resp, "content", "")))
    return 1.0 if obj.get("recognized") and obj.get("source") else 0.0


def _continuation_probe(chat, problem: Problem) -> float:
    cut = max(20, int(len(problem.statement) * 0.3))
    head = problem.statement[:cut]
    tail = problem.statement[cut:]
    msg = (
        "Continue the following physics problem statement as you believe it is "
        "originally written. Return JSON with a single key 'continuation'.\n\n"
        f"Begin: {head}"
    )
    resp = chat.invoke([{"role": "user", "content": msg}])
    obj = _parse_json(str(getattr(resp, "content", "")))
    guess = str(obj.get("continuation", ""))
    if not tail or not guess:
        return 0.0
    return difflib.SequenceMatcher(None, tail.lower(), guess.lower()).ratio()


def _cold_answer_probe(chat, problem: Problem) -> float:
    msg = (
        "Give ONLY the final numerical or symbolic answer to this physics problem, "
        "no work shown. Return JSON with a single key 'answer'.\n\n"
        f"Problem: {problem.statement}"
    )
    resp = chat.invoke([{"role": "user", "content": msg}])
    obj = _parse_json(str(getattr(resp, "content", "")))
    guess = str(obj.get("answer", "")).strip()
    truth = problem.expected.value.strip()
    if not guess:
        return 0.0
    return 1.0 if guess == truth else 0.0


def probe_contamination(problem: Problem, config: BenchConfig) -> Contamination:
    chat = init_chat_model(config.probe_model, temperature=0)
    scores = [
        _recognition_probe(chat, problem),
        _continuation_probe(chat, problem),
        _cold_answer_probe(chat, problem),
    ]
    score = sum(scores) / len(scores)
    if score < 0.34:
        verdict: str = "clean"
    elif score < 0.67:
        verdict = "suspect"
    else:
        verdict = "contaminated"
    return Contamination(
        checked_at=dt.datetime.now(dt.UTC).isoformat(),
        method=PROBE_METHOD_ID,
        score=round(score, 3),
        verdict=verdict,  # type: ignore[arg-type]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_generator_probe.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/generator/probe.py tests/benchmark/test_generator_probe.py
git commit -m "feat(bench): add contamination probe (recognition + continuation + cold answer)"
```

---

### Task 19: Generation pipeline orchestrator

**Files:**
- Create: `src/solvay/benchmark/generator/pipeline.py`
- Create: `tests/benchmark/test_generator_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_generator_pipeline.py`:

```python
"""Tests for the generator pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.pipeline import GenerationReport, generate_batch
from solvay.benchmark.generator.skeleton import GenerationSkeleton
from solvay.benchmark.generator.verifier import VerifyOutcome
from solvay.benchmark.schema import Contamination, Problem


def _good_problem(pid: str = "synth-mech-T1") -> Problem:
    return Problem.model_validate(
        {
            "id": pid,
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "solvay-generator-v0.1",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 1,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": ["pendulum"],
            "difficulty": "intermediate",
            "statement": "good problem",
            "given": {"g": "9.81", "L": "1"},
            "find": "omega",
            "expected": {
                "kind": "symbolic",
                "value": "sqrt(g/L)",
                "unit": "rad/s",
                "tolerance_rel": 0.01,
                "verification": {"method": "sympy_equivalence", "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_generate_batch_writes_verified_problems(tmp_path: Path) -> None:
    contamination = Contamination(
        checked_at="2026-04-18T00:00:00Z",
        method="model-recall-probe-v1",
        score=0.1,
        verdict="clean",
    )
    with (
        patch(
            "solvay.benchmark.generator.pipeline.compose_problem",
            side_effect=[_good_problem("synth-mech-T1"), _good_problem("synth-mech-T2")],
        ),
        patch(
            "solvay.benchmark.generator.pipeline.verify_problem",
            return_value=VerifyOutcome(ok=True, method="sympy_equivalence"),
        ),
        patch(
            "solvay.benchmark.generator.pipeline.probe_contamination",
            return_value=contamination,
        ),
    ):
        report = generate_batch(
            skeletons=[
                GenerationSkeleton(domain="mechanics", compose=["pendulum"]),
                GenerationSkeleton(domain="mechanics", compose=["lorentz"]),
            ],
            out_dir=tmp_path,
            config=BenchConfig(),
        )
    assert isinstance(report, GenerationReport)
    assert report.written == 2
    assert report.rejected_verification == 0
    assert (tmp_path / "synth-mech-T1.json").exists()
    assert (tmp_path / "synth-mech-T2.json").exists()


def test_generate_batch_rejects_verification_failures(tmp_path: Path) -> None:
    with (
        patch("solvay.benchmark.generator.pipeline.compose_problem", return_value=_good_problem()),
        patch(
            "solvay.benchmark.generator.pipeline.verify_problem",
            return_value=VerifyOutcome(ok=False, method="sympy_equivalence", reason="bad"),
        ),
    ):
        report = generate_batch(
            skeletons=[GenerationSkeleton(domain="mechanics", compose=["pendulum"])],
            out_dir=tmp_path,
            config=BenchConfig(),
        )
    assert report.written == 0
    assert report.rejected_verification == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_generator_pipeline.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the pipeline**

Create `src/solvay/benchmark/generator/pipeline.py`:

```python
"""Generator orchestrator: compose → verify → probe → write."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.composer import compose_problem
from solvay.benchmark.generator.probe import probe_contamination
from solvay.benchmark.generator.skeleton import GenerationSkeleton
from solvay.benchmark.generator.verifier import verify_problem
from solvay.benchmark.schema import dump_problem


@dataclass
class GenerationReport:
    attempted: int = 0
    written: int = 0
    rejected_verification: int = 0
    rejected_compose: int = 0
    contamination_breakdown: dict[str, int] = field(default_factory=dict)
    written_paths: list[Path] = field(default_factory=list)


def generate_batch(
    skeletons: list[GenerationSkeleton],
    out_dir: Path,
    config: BenchConfig,
) -> GenerationReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = GenerationReport()
    for sk in skeletons:
        report.attempted += 1
        try:
            problem = compose_problem(sk, config)
        except Exception:
            report.rejected_compose += 1
            continue
        outcome = verify_problem(problem)
        if not outcome.ok:
            report.rejected_verification += 1
            continue
        contamination = probe_contamination(problem, config)
        problem = problem.model_copy(update={"contamination": contamination})
        report.contamination_breakdown[contamination.verdict] = (
            report.contamination_breakdown.get(contamination.verdict, 0) + 1
        )
        path = out_dir / f"{problem.id}.json"
        dump_problem(problem, path)
        report.written += 1
        report.written_paths.append(path)
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_generator_pipeline.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/generator/pipeline.py tests/benchmark/test_generator_pipeline.py
git commit -m "feat(bench): add generator pipeline (compose → verify → probe → write)"
```

---

### Task 20: `bench generate` CLI command

**Files:**
- Modify: `src/solvay/benchmark/cli.py`
- Modify: `tests/benchmark/test_cli.py`

- [ ] **Step 1: Extend the failing test**

Append to `tests/benchmark/test_cli.py`:

```python
def test_cli_generate_invokes_pipeline(tmp_path: Path, monkeypatch) -> None:
    from solvay.benchmark.generator.pipeline import GenerationReport

    called: dict[str, object] = {}

    def fake_generate_batch(skeletons, out_dir, config):  # type: ignore[no-untyped-def]
        called["skeleton_count"] = len(skeletons)
        called["out_dir"] = out_dir
        return GenerationReport(attempted=len(skeletons), written=len(skeletons))

    monkeypatch.setattr("solvay.benchmark.cli.generate_batch", fake_generate_batch)

    runner_cli = CliRunner()
    result = runner_cli.invoke(
        app,
        [
            "generate",
            "--domain",
            "mechanics",
            "--compose",
            "pendulum,lorentz",
            "--n",
            "3",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert called["skeleton_count"] == 3
    assert called["out_dir"] == tmp_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_cli.py::test_cli_generate_invokes_pipeline -v`
Expected: FAIL.

- [ ] **Step 3: Add the `generate` command**

Append to `src/solvay/benchmark/cli.py`, just before `if __name__ == "__main__":`:

```python
from solvay.benchmark.generator.pipeline import generate_batch  # noqa: E402
from solvay.benchmark.generator.skeleton import GenerationSkeleton  # noqa: E402


@app.command("generate")
def generate_cmd(
    domain: Annotated[str, typer.Option("--domain", help="Physics domain.")] = "mechanics",
    compose: Annotated[
        str,
        typer.Option(
            "--compose",
            help="Comma-separated concepts to compose in each generated problem.",
        ),
    ] = "",
    n: Annotated[int, typer.Option("--n", min=1)] = 5,
    approach: Annotated[
        str, typer.Option("--approach", help="'backward' or 'forward'.")
    ] = "backward",
    difficulty: Annotated[
        str, typer.Option("--difficulty", help="'easy' | 'intermediate' | 'hard'.")
    ] = "intermediate",
    seed: Annotated[int, typer.Option("--seed", help="Starting seed.")] = 1,
    out: Annotated[
        Path, typer.Option("--out", help="Output directory for generated problem JSONs.")
    ] = Path("benchmark/problems/mechanics"),
) -> None:
    """Generate a batch of synthetic problems."""
    from solvay.benchmark.config import BenchConfig

    concepts = [c.strip() for c in compose.split(",") if c.strip()]
    if not concepts:
        typer.echo("--compose must list at least one concept.", err=True)
        raise typer.Exit(2)

    skeletons = [
        GenerationSkeleton(
            domain=domain,
            compose=concepts,
            difficulty=difficulty,  # type: ignore[arg-type]
            approach=approach,  # type: ignore[arg-type]
            seed=seed + i,
        )
        for i in range(n)
    ]
    report = generate_batch(skeletons=skeletons, out_dir=out, config=BenchConfig())
    typer.echo(
        f"Attempted: {report.attempted}  "
        f"Written: {report.written}  "
        f"Rejected (compose): {report.rejected_compose}  "
        f"Rejected (verification): {report.rejected_verification}  "
        f"Contamination: {report.contamination_breakdown}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/cli.py tests/benchmark/test_cli.py
git commit -m "feat(bench): add 'solvay-bench generate' CLI command"
```

---

## Phase 4 — Generate the v1 mechanics dataset

### Task 21: Author the real 20-problem mechanics batch

This task is not pure-code; it runs the pipeline against a real LLM and commits the surviving problems.

**Files:**
- Create (via pipeline): `benchmark/problems/mechanics/synth-mech-*.json` (up to 20)

- [ ] **Step 1: Confirm API keys are available**

Run: `test -n "$ANTHROPIC_API_KEY" && echo OK || echo MISSING`
Expected: `OK`. If missing, set it in `.env`.

- [ ] **Step 2: Dry-run the composer on a single skeleton**

Run:
```bash
uv run solvay-bench generate \
  --domain mechanics --compose pendulum,damping --n 1 \
  --out /tmp/solvay-gen-test --approach backward
```
Expected: at least one JSON in `/tmp/solvay-gen-test/`. Inspect it; if it looks off, iterate the composer prompt (Task 16) and re-test.

- [ ] **Step 3: Run the full batch**

Run (this WILL consume tokens — expect ~20-60 minutes and budget accordingly):
```bash
uv run solvay-bench generate \
  --domain mechanics \
  --compose "pendulum,small-oscillation;incline,variable-friction;two-body,spring-coupling;rotational,moment-of-inertia;projectile,drag;orbital,perturbation;collision,elastic;harmonic,driven;rigid-body,precession;central-force,potential" \
  --n 2 \
  --out benchmark/problems/mechanics
```
(Or run 10 separate invocations with different `--compose` strings if the generator prefers short compositions.)

Goal: end up with ≥ 15 surviving JSONs under `benchmark/problems/mechanics/`.

- [ ] **Step 4: Sanity-check every generated problem**

Run: `uv run python -m pytest tests/benchmark/test_schema.py -v` and then:
```bash
uv run python -c "
from pathlib import Path
from solvay.benchmark.schema import load_problems_dir
problems = load_problems_dir(Path('benchmark/problems/mechanics'))
for p in problems:
    print(p.id, p.expected.kind, p.contamination.verdict if p.contamination else '-')
"
```
Expected: all load, all have a `contamination` verdict filled in.

- [ ] **Step 5: Commit the dataset**

```bash
git add benchmark/problems/mechanics/synth-mech-*.json
git commit -m "feat(bench): add v1 mechanics synthetic dataset"
```

---

## Phase 5 — Metrics and reporting

### Task 22: Run loader (JSONL → in-memory RunSet)

**Files:**
- Create: `src/solvay/benchmark/report/__init__.py`
- Create: `src/solvay/benchmark/report/loader.py`
- Create: `tests/benchmark/test_report_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_report_loader.py`:

```python
"""Tests for the run loader."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.loader import load_runset


def _write(path: Path, records: list[RunRecord]) -> None:
    header = RunHeader(
        run_id="rid",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:x",
        started_at="2026-04-18T00:00:00Z",
    )
    with RunWriter(path, header) as w:
        for r in records:
            w.write(r)


def test_loader_groups_by_profile_model_problem(tmp_path: Path) -> None:
    records = [
        RunRecord(
            problem_id="p1",
            profile="bare",
            model="m1",
            repeat_idx=0,
            answer_raw="",
            answer_extracted=None,
            correct=True,
            elapsed_seconds=0.0,
            tokens={"input": 1, "output": 1, "cache_read": 0},
        ),
        RunRecord(
            problem_id="p1",
            profile="bare",
            model="m1",
            repeat_idx=1,
            answer_raw="",
            answer_extracted=None,
            correct=False,
            elapsed_seconds=0.0,
            tokens={"input": 1, "output": 1, "cache_read": 0},
        ),
    ]
    path = tmp_path / "run.jsonl"
    _write(path, records)
    rs = load_runset(path)
    cell = rs.cell("p1", "bare", "m1")
    assert len(cell) == 2
    assert rs.profiles() == ["bare"]
    assert rs.models() == ["m1"]
    assert rs.problem_ids() == ["p1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_report_loader.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the loader**

Create `src/solvay/benchmark/report/__init__.py`:

```python
"""Report/analysis layer for benchmark runs."""
```

Create `src/solvay/benchmark/report/loader.py`:

```python
"""Load one or more run JSONLs into an in-memory RunSet."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, read_run


@dataclass
class RunSet:
    headers: list[RunHeader]
    records: list[RunRecord]
    _index: dict[tuple[str, str, str], list[RunRecord]]

    def cell(self, problem_id: str, profile: str, model: str) -> list[RunRecord]:
        return self._index.get((problem_id, profile, model), [])

    def profiles(self) -> list[str]:
        return sorted({r.profile for r in self.records})

    def models(self) -> list[str]:
        return sorted({r.model for r in self.records})

    def problem_ids(self) -> list[str]:
        return sorted({r.problem_id for r in self.records})


def load_runset(*paths: Path) -> RunSet:
    headers: list[RunHeader] = []
    records: list[RunRecord] = []
    for p in paths:
        h, rs = read_run(Path(p))
        headers.append(h)
        records.extend(rs)
    index: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        index[(r.problem_id, r.profile, r.model)].append(r)
    return RunSet(headers=headers, records=records, _index=dict(index))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_report_loader.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/report/__init__.py \
        src/solvay/benchmark/report/loader.py \
        tests/benchmark/test_report_loader.py
git commit -m "feat(bench): add RunSet loader"
```

---

### Task 23: Metric A calculator

**Files:**
- Create: `src/solvay/benchmark/report/metric_a.py`
- Create: `tests/benchmark/test_metric_a.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_metric_a.py`:

```python
"""Tests for metric A (pure reasoning on clean problems)."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.loader import load_runset
from solvay.benchmark.report.metric_a import compute_metric_a
from solvay.benchmark.schema import Contamination, Problem


def _make_problem(pid: str, verdict: str) -> Problem:
    return Problem.model_validate(
        {
            "id": pid,
            "version": 1,
            "source": {"kind": "synthetic", "origin": "t", "generated_at": "2026-04-18T00:00:00Z", "seed": 0},
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "s",
            "given": {},
            "find": "f",
            "expected": {
                "kind": "numeric",
                "value": "1",
                "unit": None,
                "tolerance_rel": 0.01,
                "verification": {"method": "numeric_eval", "script": None},
            },
            "contamination": {
                "checked_at": "2026-04-18T00:00:00Z",
                "method": "model-recall-probe-v1",
                "score": 0.0 if verdict == "clean" else 0.9,
                "verdict": verdict,
            },
            "notes": None,
        }
    )


def _records(tmp_path: Path) -> Path:
    header = RunHeader(
        run_id="r",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:x",
        started_at="2026-04-18T00:00:00Z",
    )
    path = tmp_path / "run.jsonl"
    with RunWriter(path, header) as w:
        # clean problem, solvay-noweb got it right
        w.write(RunRecord("p-clean", "solvay-noweb", "m1", 0, "", None, True, 0.0, {}))
        # contaminated problem, solvay-noweb also right, should be EXCLUDED
        w.write(RunRecord("p-contam", "solvay-noweb", "m1", 0, "", None, True, 0.0, {}))
        # another clean problem, solvay-noweb wrong
        w.write(RunRecord("p-clean-2", "solvay-noweb", "m1", 0, "", None, False, 0.0, {}))
        # different profile should be ignored
        w.write(RunRecord("p-clean", "solvay-full", "m1", 0, "", None, True, 0.0, {}))
    return path


def test_metric_a_only_clean_solvay_noweb(tmp_path: Path) -> None:
    run_path = _records(tmp_path)
    rs = load_runset(run_path)
    problems = {
        "p-clean": _make_problem("p-clean", "clean"),
        "p-contam": _make_problem("p-contam", "contaminated"),
        "p-clean-2": _make_problem("p-clean-2", "clean"),
    }
    a = compute_metric_a(rs, problems)
    # m1: 1 correct out of 2 clean attempts → 0.5
    assert a["m1"].clean_total == 2
    assert a["m1"].clean_correct == 1
    assert a["m1"].accuracy == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_metric_a.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement metric A**

Create `src/solvay/benchmark/report/metric_a.py`:

```python
"""Metric A: pure-reasoning accuracy on clean problems via solvay-noweb."""

from __future__ import annotations

from dataclasses import dataclass

from solvay.benchmark.report.loader import RunSet
from solvay.benchmark.schema import Problem


@dataclass(frozen=True)
class MetricAByModel:
    clean_total: int
    clean_correct: int
    accuracy: float


def compute_metric_a(
    runset: RunSet, problems: dict[str, Problem], profile: str = "solvay-noweb"
) -> dict[str, MetricAByModel]:
    clean_ids = {
        pid for pid, pr in problems.items() if pr.contamination and pr.contamination.verdict == "clean"
    }
    by_model: dict[str, list[bool]] = {}
    for record in runset.records:
        if record.profile != profile:
            continue
        if record.problem_id not in clean_ids:
            continue
        by_model.setdefault(record.model, []).append(record.correct)
    result: dict[str, MetricAByModel] = {}
    for model, outcomes in by_model.items():
        total = len(outcomes)
        correct = sum(1 for x in outcomes if x)
        result[model] = MetricAByModel(
            clean_total=total,
            clean_correct=correct,
            accuracy=(correct / total) if total else 0.0,
        )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_metric_a.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/report/metric_a.py tests/benchmark/test_metric_a.py
git commit -m "feat(bench): metric A (clean-problem accuracy for solvay-noweb)"
```

---

### Task 24: Metric B calculator (ladder)

**Files:**
- Create: `src/solvay/benchmark/report/metric_b.py`
- Create: `tests/benchmark/test_metric_b.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_metric_b.py`:

```python
"""Tests for metric B (ladder)."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.loader import load_runset
from solvay.benchmark.report.metric_b import LADDER_ORDER, compute_metric_b


def test_metric_b_ladder_shape(tmp_path: Path) -> None:
    header = RunHeader(
        run_id="r",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:x",
        started_at="2026-04-18T00:00:00Z",
    )
    path = tmp_path / "run.jsonl"
    with RunWriter(path, header) as w:
        # Two problems, Sonnet improving along the ladder.
        for pid in ["p1", "p2"]:
            w.write(RunRecord(pid, "bare", "sonnet", 0, "", None, False, 0.0, {}))
            w.write(RunRecord(pid, "prompted", "sonnet", 0, "", None, False, 0.0, {}))
            w.write(RunRecord(pid, "tooled", "sonnet", 0, "", None, True, 0.0, {}))
            w.write(RunRecord(pid, "solvay-noweb", "sonnet", 0, "", None, True, 0.0, {}))
            w.write(RunRecord(pid, "solvay-full", "sonnet", 0, "", None, True, 0.0, {}))

    rs = load_runset(path)
    ladder = compute_metric_b(rs)
    assert list(ladder["sonnet"].keys()) == list(LADDER_ORDER)
    assert ladder["sonnet"]["bare"] == 0.0
    assert ladder["sonnet"]["tooled"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_metric_b.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement metric B**

Create `src/solvay/benchmark/report/metric_b.py`:

```python
"""Metric B: accuracy ladder across profiles for each model."""

from __future__ import annotations

from solvay.benchmark.report.loader import RunSet

LADDER_ORDER: tuple[str, ...] = (
    "bare",
    "prompted",
    "tooled",
    "solvay-noweb",
    "solvay-full",
)


def compute_metric_b(runset: RunSet) -> dict[str, dict[str, float]]:
    """Return ``{model: {profile: accuracy}}`` preserving LADDER_ORDER."""
    by_cell: dict[tuple[str, str], list[bool]] = {}
    for record in runset.records:
        if record.profile not in LADDER_ORDER:
            continue
        by_cell.setdefault((record.model, record.profile), []).append(record.correct)
    models = sorted({m for (m, _) in by_cell})
    out: dict[str, dict[str, float]] = {}
    for model in models:
        rung_map: dict[str, float] = {}
        for profile in LADDER_ORDER:
            outcomes = by_cell.get((model, profile), [])
            rung_map[profile] = (sum(outcomes) / len(outcomes)) if outcomes else 0.0
        out[model] = rung_map
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_metric_b.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/report/metric_b.py tests/benchmark/test_metric_b.py
git commit -m "feat(bench): metric B (ladder across profiles)"
```

---

### Task 25: Metric D.1 — error taxonomy classifier

**Files:**
- Create: `src/solvay/benchmark/report/classifier.py`
- Create: `src/solvay/benchmark/report/metric_d.py`
- Create: `tests/benchmark/test_metric_d.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_metric_d.py`:

```python
"""Tests for metric D (error taxonomy, cost, calibration)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.loader import load_runset
from solvay.benchmark.report.metric_d import (
    compute_cost_per_correct,
    compute_error_histogram,
    compute_self_calibration,
)
from solvay.benchmark.schema import Problem


def _problem(pid: str = "p1") -> Problem:
    return Problem.model_validate(
        {
            "id": pid,
            "version": 1,
            "source": {"kind": "synthetic", "origin": "t", "generated_at": "2026-04-18T00:00:00Z", "seed": 0},
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "s",
            "given": {},
            "find": "f",
            "expected": {
                "kind": "numeric",
                "value": "1",
                "unit": "m",
                "tolerance_rel": 0.01,
                "verification": {"method": "numeric_eval", "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def _write(tmp_path: Path, records: list[RunRecord]) -> Path:
    header = RunHeader(
        run_id="r",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:x",
        started_at="2026-04-18T00:00:00Z",
    )
    path = tmp_path / "run.jsonl"
    with RunWriter(path, header) as w:
        for r in records:
            w.write(r)
    return path


def test_cost_per_correct_divides_tokens_by_correct(tmp_path: Path) -> None:
    records = [
        RunRecord("p1", "bare", "m1", 0, "", None, True, 0.0, {"input": 50, "output": 50}),
        RunRecord("p1", "bare", "m1", 1, "", None, False, 0.0, {"input": 50, "output": 50}),
    ]
    rs = load_runset(_write(tmp_path, records))
    table = compute_cost_per_correct(rs)
    assert table[("bare", "m1")].tokens_total == 200
    assert table[("bare", "m1")].correct_count == 1
    assert table[("bare", "m1")].cost_per_correct == 200


def test_self_calibration_flags_detection(tmp_path: Path) -> None:
    # answer_raw contains "[uncertain]" → treated as flagged
    records = [
        RunRecord("p1", "solvay-full", "m1", 0, "answer: 0.5 [uncertain]", None, False, 0.0, {}),
        RunRecord("p2", "solvay-full", "m1", 0, "answer: 1.0", None, True, 0.0, {}),
        RunRecord("p3", "solvay-full", "m1", 0, "answer: 2.0 [uncertain]", None, True, 0.0, {}),
        RunRecord("p4", "solvay-full", "m1", 0, "answer: 3.0", None, False, 0.0, {}),
    ]
    rs = load_runset(_write(tmp_path, records))
    cal = compute_self_calibration(rs)
    entry = cal[("solvay-full", "m1")]
    assert entry.flagged_failures == 1  # p1
    assert entry.total_failures == 2  # p1 + p4
    assert entry.flagged_successes == 1  # p3
    assert entry.total_successes == 2
    assert entry.true_flag_rate == 0.5
    assert entry.false_flag_rate == 0.5


def test_error_histogram_uses_classifier(tmp_path: Path) -> None:
    records = [
        RunRecord("p1", "bare", "m1", 0, "final answer 42 kg (wrong)", None, False, 0.0, {}),
        RunRecord("p2", "bare", "m1", 0, "oops", None, False, 0.0, {}),
    ]
    rs = load_runset(_write(tmp_path, records))
    fake_chat = MagicMock()
    fake_chat.invoke.side_effect = [
        MagicMock(content='{"category": "units_error"}'),
        MagicMock(content='{"category": "incomplete"}'),
    ]
    with patch(
        "solvay.benchmark.report.classifier.init_chat_model",
        return_value=fake_chat,
    ):
        hist = compute_error_histogram(rs, {"p1": _problem("p1"), "p2": _problem("p2")}, BenchConfig())
    assert hist[("bare", "m1")]["units_error"] == 1
    assert hist[("bare", "m1")]["incomplete"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_metric_d.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the classifier**

Create `src/solvay/benchmark/report/classifier.py`:

```python
"""Post-hoc LLM classifier for error taxonomy."""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Problem

ErrorCategory = Literal[
    "wrong_concept",
    "algebra_error",
    "units_error",
    "dimensional_inconsistency",
    "incomplete",
    "hallucinated_data",
]

ALL_CATEGORIES: tuple[ErrorCategory, ...] = (
    "wrong_concept",
    "algebra_error",
    "units_error",
    "dimensional_inconsistency",
    "incomplete",
    "hallucinated_data",
)

_SYSTEM = """\
You are a physics error classifier. Given a problem and a WRONG answer from a
model, assign exactly ONE category from this closed list:
- wrong_concept: applied the wrong law or framework
- algebra_error: set up correctly, mistake in the algebra or arithmetic
- units_error: magnitude correct, unit wrong or missing
- dimensional_inconsistency: answer has wrong physical dimensions
- incomplete: no final answer produced
- hallucinated_data: invented quantities not given in the problem

Return JSON with a single key "category".
"""


def classify_error(
    problem: Problem,
    answer_raw: str,
    config: BenchConfig,
) -> ErrorCategory:
    chat = init_chat_model(config.error_classifier_model, temperature=0)
    user = (
        f"Problem statement:\n{problem.statement}\n\n"
        f"Given: {problem.given}\n"
        f"Expected answer: {problem.expected.value} {problem.expected.unit or ''}\n\n"
        f"Wrong answer (verbatim):\n{answer_raw}\n"
    )
    resp = chat.invoke([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}])
    raw = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        str(getattr(resp, "content", "")).strip(),
        flags=re.MULTILINE,
    )
    try:
        obj = json.loads(raw)
        category = obj.get("category", "incomplete")
    except json.JSONDecodeError:
        category = "incomplete"
    if category not in ALL_CATEGORIES:
        category = "incomplete"
    return category  # type: ignore[return-value]
```

- [ ] **Step 4: Implement metric D (three sub-metrics)**

Create `src/solvay/benchmark/report/metric_d.py`:

```python
"""Metric D: error taxonomy (D.1), cost per correct (D.2), self-calibration (D.3)."""

from __future__ import annotations

from dataclasses import dataclass

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.report.classifier import ALL_CATEGORIES, classify_error
from solvay.benchmark.report.loader import RunSet
from solvay.benchmark.schema import Problem

UNCERTAINTY_MARKER = "[uncertain]"


@dataclass(frozen=True)
class CostEntry:
    tokens_total: int
    correct_count: int
    cost_per_correct: float


@dataclass(frozen=True)
class CalibrationEntry:
    total_successes: int
    total_failures: int
    flagged_successes: int
    flagged_failures: int
    true_flag_rate: float
    false_flag_rate: float


def compute_error_histogram(
    runset: RunSet, problems: dict[str, Problem], config: BenchConfig
) -> dict[tuple[str, str], dict[str, int]]:
    hist: dict[tuple[str, str], dict[str, int]] = {}
    for r in runset.records:
        if r.correct:
            continue
        key = (r.profile, r.model)
        if r.problem_id not in problems:
            continue
        bucket = hist.setdefault(key, {cat: 0 for cat in ALL_CATEGORIES})
        category = classify_error(problems[r.problem_id], r.answer_raw, config)
        bucket[category] += 1
    return hist


def compute_cost_per_correct(runset: RunSet) -> dict[tuple[str, str], CostEntry]:
    bucket: dict[tuple[str, str], list[int]] = {}
    correct: dict[tuple[str, str], int] = {}
    for r in runset.records:
        key = (r.profile, r.model)
        total = sum(int(v) for v in r.tokens.values())
        bucket.setdefault(key, []).append(total)
        correct[key] = correct.get(key, 0) + (1 if r.correct else 0)
    out: dict[tuple[str, str], CostEntry] = {}
    for key, totals in bucket.items():
        tokens_total = sum(totals)
        correct_count = correct.get(key, 0)
        cpc = tokens_total / correct_count if correct_count else float("inf")
        out[key] = CostEntry(tokens_total, correct_count, cpc)
    return out


def compute_self_calibration(runset: RunSet) -> dict[tuple[str, str], CalibrationEntry]:
    out: dict[tuple[str, str], CalibrationEntry] = {}
    by_cell: dict[tuple[str, str], list] = {}
    for r in runset.records:
        if not r.profile.startswith("solvay"):
            continue
        by_cell.setdefault((r.profile, r.model), []).append(r)
    for key, records in by_cell.items():
        total_s = sum(1 for r in records if r.correct)
        total_f = sum(1 for r in records if not r.correct)
        flagged_s = sum(1 for r in records if r.correct and UNCERTAINTY_MARKER in r.answer_raw)
        flagged_f = sum(1 for r in records if not r.correct and UNCERTAINTY_MARKER in r.answer_raw)
        true_rate = (flagged_f / total_f) if total_f else 0.0
        false_rate = (flagged_s / total_s) if total_s else 0.0
        out[key] = CalibrationEntry(
            total_successes=total_s,
            total_failures=total_f,
            flagged_successes=flagged_s,
            flagged_failures=flagged_f,
            true_flag_rate=true_rate,
            false_flag_rate=false_rate,
        )
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_metric_d.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/solvay/benchmark/report/classifier.py \
        src/solvay/benchmark/report/metric_d.py \
        tests/benchmark/test_metric_d.py
git commit -m "feat(bench): metric D (error histogram, cost/correct, self-calibration)"
```

---

### Task 26: Noise floor computation

**Files:**
- Create: `src/solvay/benchmark/report/noise.py`
- Create: `tests/benchmark/test_noise.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_noise.py`:

```python
"""Tests for noise floor calibration."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.loader import load_runset
from solvay.benchmark.report.noise import (
    compute_noise_floor,
    load_baseline_stats,
    save_baseline_stats,
)


def _write_two_repeats(tmp_path: Path) -> Path:
    header = RunHeader(
        run_id="r",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:x",
        started_at="2026-04-18T00:00:00Z",
    )
    path = tmp_path / "run.jsonl"
    with RunWriter(path, header) as w:
        # Three problems, two repeats, 4/6 correct → accuracy ~= 0.667, σ = 0 since same set
        w.write(RunRecord("p1", "solvay-full", "m1", 0, "", None, True, 0, {}))
        w.write(RunRecord("p1", "solvay-full", "m1", 1, "", None, False, 0, {}))
        w.write(RunRecord("p2", "solvay-full", "m1", 0, "", None, True, 0, {}))
        w.write(RunRecord("p2", "solvay-full", "m1", 1, "", None, True, 0, {}))
        w.write(RunRecord("p3", "solvay-full", "m1", 0, "", None, True, 0, {}))
        w.write(RunRecord("p3", "solvay-full", "m1", 1, "", None, False, 0, {}))
    return path


def test_noise_floor_computes_std_across_repeats(tmp_path: Path) -> None:
    rs = load_runset(_write_two_repeats(tmp_path))
    floor = compute_noise_floor(rs)
    # repeat 0: 3/3 = 1.0; repeat 1: 1/3 ≈ 0.333 → mean 0.667, std ≈ 0.333
    entry = floor[("solvay-full", "m1")]
    assert abs(entry.mean - 0.667) < 0.01
    assert entry.std > 0.3


def test_baseline_stats_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "baseline_stats.json"
    payload = {("solvay-full", "m1"): {"mean": 0.67, "std": 0.03, "n": 10}}
    save_baseline_stats(path, payload)
    loaded = load_baseline_stats(path)
    assert loaded[("solvay-full", "m1")]["std"] == 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_noise.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement noise**

Create `src/solvay/benchmark/report/noise.py`:

```python
"""Noise floor calibration across repeated runs of the same matrix cell."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from solvay.benchmark.report.loader import RunSet


@dataclass(frozen=True)
class NoiseEntry:
    mean: float
    std: float
    n: int


def compute_noise_floor(runset: RunSet) -> dict[tuple[str, str], NoiseEntry]:
    # group by (profile, model, repeat_idx) → accuracy per repetition
    per_repeat: dict[tuple[str, str, int], list[bool]] = {}
    for r in runset.records:
        per_repeat.setdefault((r.profile, r.model, r.repeat_idx), []).append(r.correct)
    accuracies: dict[tuple[str, str], list[float]] = {}
    for (profile, model, _), outcomes in per_repeat.items():
        if not outcomes:
            continue
        acc = sum(outcomes) / len(outcomes)
        accuracies.setdefault((profile, model), []).append(acc)
    out: dict[tuple[str, str], NoiseEntry] = {}
    for key, vals in accuracies.items():
        mean = statistics.fmean(vals) if vals else 0.0
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[key] = NoiseEntry(mean=mean, std=std, n=len(vals))
    return out


def save_baseline_stats(path: Path, stats: dict[tuple[str, str], dict]) -> None:
    serializable = {f"{p}|{m}": payload for (p, m), payload in stats.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def load_baseline_stats(path: Path) -> dict[tuple[str, str], dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict] = {}
    for key, payload in raw.items():
        profile, model = key.split("|", 1)
        out[(profile, model)] = payload
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_noise.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/report/noise.py tests/benchmark/test_noise.py
git commit -m "feat(bench): noise floor calibration with baseline_stats.json"
```

---

### Task 27: Regression/history detector

**Files:**
- Create: `src/solvay/benchmark/report/regression.py`
- Create: `tests/benchmark/test_regression.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_regression.py`:

```python
"""Tests for regression detection and history walking."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.regression import HistoryEntry, load_history, verdict_for_delta


def _write(tmp_path: Path, fingerprint: str, correct_count: int) -> Path:
    header = RunHeader(
        run_id=f"r-{fingerprint[:4]}",
        solvay_version="0.3.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint=fingerprint,
        started_at="2026-04-18T00:00:00Z",
    )
    path = tmp_path / f"{header.run_id}.jsonl"
    with RunWriter(path, header) as w:
        for i in range(3):
            w.write(
                RunRecord(
                    f"p{i}",
                    "solvay-full",
                    "m1",
                    0,
                    "",
                    None,
                    i < correct_count,
                    0.0,
                    {},
                )
            )
    return path


def test_verdict_for_delta_uses_two_sigma() -> None:
    assert verdict_for_delta(delta=-0.03, sigma=0.05) == "within_noise"
    assert verdict_for_delta(delta=-0.15, sigma=0.05) == "regression"
    assert verdict_for_delta(delta=+0.15, sigma=0.05) == "improvement"


def test_load_history_groups_by_fingerprint(tmp_path: Path) -> None:
    _write(tmp_path, "sha256:a3f", correct_count=2)
    _write(tmp_path, "sha256:7f3", correct_count=3)
    entries = load_history(tmp_path, profile="solvay-full", model="m1")
    assert len(entries) == 2
    assert all(isinstance(e, HistoryEntry) for e in entries)
    acc_by_fp = {e.config_fingerprint: e.accuracy for e in entries}
    assert acc_by_fp["sha256:a3f"] == 2 / 3
    assert acc_by_fp["sha256:7f3"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_regression.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement regression/history**

Create `src/solvay/benchmark/report/regression.py`:

```python
"""Longitudinal history walking and regression detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from solvay.benchmark.jsonl import read_run


@dataclass(frozen=True)
class HistoryEntry:
    run_id: str
    solvay_version: str
    config_fingerprint: str
    started_at: str
    accuracy: float
    n_correct: int
    n_total: int


Verdict = Literal["improvement", "within_noise", "regression"]


def verdict_for_delta(delta: float, sigma: float) -> Verdict:
    threshold = 2 * sigma
    if abs(delta) <= threshold:
        return "within_noise"
    return "improvement" if delta > 0 else "regression"


def load_history(
    runs_dir: Path, profile: str, model: str
) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    for path in sorted(Path(runs_dir).glob("*.jsonl")):
        header, records = read_run(path)
        filtered = [
            r for r in records if r.profile == profile and r.model == model
        ]
        if not filtered:
            continue
        n_total = len(filtered)
        n_correct = sum(1 for r in filtered if r.correct)
        acc = n_correct / n_total
        entries.append(
            HistoryEntry(
                run_id=header.run_id,
                solvay_version=header.solvay_version,
                config_fingerprint=header.config_fingerprint,
                started_at=header.started_at,
                accuracy=acc,
                n_correct=n_correct,
                n_total=n_total,
            )
        )
    return sorted(entries, key=lambda e: e.started_at)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_regression.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/report/regression.py tests/benchmark/test_regression.py
git commit -m "feat(bench): longitudinal history + 2σ regression verdict"
```

---

### Task 28: Markdown report writer

**Files:**
- Create: `src/solvay/benchmark/report/markdown.py`
- Create: `tests/benchmark/test_markdown.py`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_markdown.py`:

```python
"""Tests for the markdown report writer."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter
from solvay.benchmark.report.loader import load_runset
from solvay.benchmark.report.markdown import render_report
from solvay.benchmark.schema import Problem


def _problem(pid: str, verdict: str = "clean") -> Problem:
    return Problem.model_validate(
        {
            "id": pid,
            "version": 1,
            "source": {"kind": "synthetic", "origin": "t", "generated_at": "2026-04-18T00:00:00Z", "seed": 0},
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "s",
            "given": {},
            "find": "f",
            "expected": {
                "kind": "numeric",
                "value": "1",
                "unit": None,
                "tolerance_rel": 0.01,
                "verification": {"method": "numeric_eval", "script": None},
            },
            "contamination": {
                "checked_at": "2026-04-18T00:00:00Z",
                "method": "model-recall-probe-v1",
                "score": 0.0,
                "verdict": verdict,
            },
            "notes": None,
        }
    )


def test_render_report_has_expected_sections(tmp_path: Path) -> None:
    header = RunHeader(
        run_id="r",
        solvay_version="0.1.0",
        git_commit="abc",
        git_dirty=False,
        config_fingerprint="sha256:x",
        started_at="2026-04-18T00:00:00Z",
    )
    path = tmp_path / "run.jsonl"
    with RunWriter(path, header) as w:
        w.write(RunRecord("p1", "bare", "m1", 0, "", None, False, 0.1, {"input": 1, "output": 1}))
        w.write(RunRecord("p1", "solvay-noweb", "m1", 0, "", None, True, 0.1, {"input": 1, "output": 1}))
        w.write(RunRecord("p1", "solvay-full", "m1", 0, "", None, True, 0.1, {"input": 1, "output": 1}))
    rs = load_runset(path)
    md = render_report(rs, problems={"p1": _problem("p1")}, include_error_histogram=False)
    assert "# Benchmark Report" in md
    assert "Metric A" in md
    assert "Metric B" in md
    assert "bare" in md and "solvay-full" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_markdown.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the writer**

Create `src/solvay/benchmark/report/markdown.py`:

```python
"""Render a benchmark RunSet to markdown."""

from __future__ import annotations

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.report.loader import RunSet
from solvay.benchmark.report.metric_a import compute_metric_a
from solvay.benchmark.report.metric_b import LADDER_ORDER, compute_metric_b
from solvay.benchmark.report.metric_d import (
    compute_cost_per_correct,
    compute_error_histogram,
    compute_self_calibration,
)
from solvay.benchmark.report.noise import compute_noise_floor
from solvay.benchmark.schema import Problem


def _header_block(runset: RunSet) -> str:
    lines = ["# Benchmark Report", ""]
    for h in runset.headers:
        dirty = "  ⚠ DIRTY" if h.git_dirty else ""
        lines.append(
            f"- Run `{h.run_id}` — solvay v{h.solvay_version} — "
            f"commit {h.git_commit}{dirty} — fp `{h.config_fingerprint[:20]}…`"
        )
    lines.append("")
    return "\n".join(lines)


def _metric_a_block(runset: RunSet, problems: dict[str, Problem]) -> str:
    lines = ["## Metric A — pure reasoning (solvay-noweb on clean problems)", ""]
    a = compute_metric_a(runset, problems)
    if not a:
        lines.append("_No solvay-noweb runs or no clean problems._")
    else:
        lines.append("| Model | Clean correct | Clean total | Accuracy |")
        lines.append("|---|---|---|---|")
        for model in sorted(a):
            row = a[model]
            lines.append(f"| {model} | {row.clean_correct} | {row.clean_total} | {row.accuracy:.3f} |")
    lines.append("")
    return "\n".join(lines)


def _metric_b_block(runset: RunSet) -> str:
    lines = ["## Metric B — ladder (profile × model)", ""]
    ladder = compute_metric_b(runset)
    if not ladder:
        lines.append("_No runs._")
    else:
        header = "| Model | " + " | ".join(LADDER_ORDER) + " |"
        sep = "|---|" + "---|" * len(LADDER_ORDER)
        lines.append(header)
        lines.append(sep)
        for model in sorted(ladder):
            row = ladder[model]
            cells = " | ".join(f"{row[p]:.3f}" for p in LADDER_ORDER)
            lines.append(f"| {model} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def _metric_d_blocks(
    runset: RunSet,
    problems: dict[str, Problem],
    include_error_histogram: bool,
    bench_config: BenchConfig,
) -> str:
    lines: list[str] = []

    cost = compute_cost_per_correct(runset)
    lines.append("## Metric D.2 — cost per correct answer")
    lines.append("")
    lines.append("| Profile | Model | Tokens | Correct | Tokens/correct |")
    lines.append("|---|---|---|---|---|")
    for (profile, model), entry in sorted(cost.items()):
        lines.append(
            f"| {profile} | {model} | {entry.tokens_total} | "
            f"{entry.correct_count} | {entry.cost_per_correct:.1f} |"
        )
    lines.append("")

    cal = compute_self_calibration(runset)
    lines.append("## Metric D.3 — self-calibration (Solvay profiles)")
    lines.append("")
    lines.append("| Profile | Model | True-flag rate | False-flag rate |")
    lines.append("|---|---|---|---|")
    for (profile, model), entry in sorted(cal.items()):
        lines.append(
            f"| {profile} | {model} | {entry.true_flag_rate:.3f} | {entry.false_flag_rate:.3f} |"
        )
    lines.append("")

    if include_error_histogram:
        hist = compute_error_histogram(runset, problems, bench_config)
        lines.append("## Metric D.1 — error taxonomy")
        lines.append("")
        for (profile, model), bucket in sorted(hist.items()):
            lines.append(f"### {profile} × {model}")
            lines.append("")
            for cat, count in sorted(bucket.items()):
                lines.append(f"- {cat}: {count}")
            lines.append("")
    return "\n".join(lines)


def _noise_block(runset: RunSet) -> str:
    lines = ["## Noise floor (from repeats in this run)", ""]
    floor = compute_noise_floor(runset)
    if not floor or all(e.n == 1 for e in floor.values()):
        lines.append("_No --repeat data in this run; noise floor not computable._")
    else:
        lines.append("| Profile | Model | Mean | Std | n |")
        lines.append("|---|---|---|---|---|")
        for (profile, model), entry in sorted(floor.items()):
            lines.append(
                f"| {profile} | {model} | {entry.mean:.3f} | {entry.std:.3f} | {entry.n} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_report(
    runset: RunSet,
    problems: dict[str, Problem],
    include_error_histogram: bool = True,
    bench_config: BenchConfig | None = None,
) -> str:
    cfg = bench_config or BenchConfig()
    sections = [
        _header_block(runset),
        _metric_a_block(runset, problems),
        _metric_b_block(runset),
        _metric_d_blocks(runset, problems, include_error_histogram, cfg),
        _noise_block(runset),
    ]
    return "\n".join(sections).strip() + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_markdown.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/report/markdown.py tests/benchmark/test_markdown.py
git commit -m "feat(bench): markdown report renderer"
```

---

### Task 29: `bench report`, `bench history`, `bench calibrate` CLI commands

**Files:**
- Modify: `src/solvay/benchmark/cli.py`
- Modify: `tests/benchmark/test_cli.py`

- [ ] **Step 1: Extend the failing test**

Append to `tests/benchmark/test_cli.py`:

```python
def test_cli_report_writes_markdown(tmp_path: Path) -> None:
    from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter

    run_path = tmp_path / "run.jsonl"
    with RunWriter(
        run_path,
        RunHeader(
            run_id="r",
            solvay_version="0.1.0",
            git_commit="abc",
            git_dirty=False,
            config_fingerprint="sha256:x",
            started_at="2026-04-18T00:00:00Z",
        ),
    ) as w:
        w.write(
            RunRecord(
                "sample-mech-0001",
                "bare",
                "m1",
                0,
                "",
                None,
                False,
                0.0,
                {"input": 1, "output": 1},
            )
        )

    out = tmp_path / "report.md"
    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "report",
            str(run_path),
            "--problems",
            "benchmark/problems",
            "--out",
            str(out),
            "--no-error-histogram",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Benchmark Report" in text


def test_cli_history_walks_runs_dir(tmp_path: Path) -> None:
    from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter

    def _emit(fp: str, run_id: str) -> None:
        header = RunHeader(
            run_id=run_id,
            solvay_version="0.3.0",
            git_commit="abc",
            git_dirty=False,
            config_fingerprint=fp,
            started_at=f"2026-04-18T00:00:{run_id[-2:]}Z",
        )
        with RunWriter(tmp_path / f"{run_id}.jsonl", header) as w:
            w.write(RunRecord("p1", "solvay-full", "m1", 0, "", None, True, 0, {}))

    _emit("sha256:aa", "r01")
    _emit("sha256:bb", "r02")

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "history",
            "--runs-dir",
            str(tmp_path),
            "--profile",
            "solvay-full",
            "--model",
            "m1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "sha256:aa" in result.stdout or "sha256:bb" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/benchmark/test_cli.py::test_cli_report_writes_markdown tests/benchmark/test_cli.py::test_cli_history_walks_runs_dir -v`
Expected: FAIL.

- [ ] **Step 3: Add the `report` and `history` commands**

Append to `src/solvay/benchmark/cli.py`, just before `if __name__ == "__main__":`:

```python
from solvay.benchmark.report.loader import load_runset  # noqa: E402
from solvay.benchmark.report.markdown import render_report  # noqa: E402
from solvay.benchmark.report.regression import load_history  # noqa: E402
from solvay.benchmark.schema import load_problems_dir  # noqa: E402


@app.command("report")
def report_cmd(
    run_paths: Annotated[list[Path], typer.Argument(help="One or more run JSONL files.")],
    problems: Annotated[
        Path, typer.Option("--problems", help="Root directory of problem JSONs.")
    ] = Path("benchmark/problems"),
    out: Annotated[
        Path | None, typer.Option("--out", help="Destination markdown (default: stdout).")
    ] = None,
    include_error_histogram: Annotated[
        bool,
        typer.Option(
            "--error-histogram/--no-error-histogram",
            help="Whether to invoke the post-hoc LLM classifier.",
        ),
    ] = True,
) -> None:
    """Render a markdown report for one or more runs."""
    rs = load_runset(*run_paths)
    problem_list = load_problems_dir(problems)
    problem_index = {p.id: p for p in problem_list}
    md = render_report(rs, problem_index, include_error_histogram=include_error_histogram)
    if out is None:
        typer.echo(md)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {out}")


@app.command("history")
def history_cmd(
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Directory of run JSONL files.")
    ] = Path("benchmark/runs"),
    profile: Annotated[str, typer.Option("--profile")] = "solvay-full",
    model: Annotated[str, typer.Option("--model")] = "",
) -> None:
    """Walk runs_dir and print accuracy history for (profile, model)."""
    entries = load_history(runs_dir, profile=profile, model=model)
    if not entries:
        typer.echo("No matching runs.", err=True)
        raise typer.Exit(1)
    typer.echo(f"{'version':<10} {'fingerprint':<24} {'started_at':<26} {'acc':<6}")
    for e in entries:
        typer.echo(
            f"{e.solvay_version:<10} {e.config_fingerprint:<24} {e.started_at:<26} {e.accuracy:.3f}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/benchmark/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/benchmark/cli.py tests/benchmark/test_cli.py
git commit -m "feat(bench): add 'report' and 'history' CLI commands"
```

---

### Task 30: Retire the old benchmark runner and `solvay bench` command

**Files:**
- Delete: `benchmark/run_bench.py`
- Modify: `src/solvay/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Remove the old `bench` command from the main CLI**

Edit `src/solvay/cli.py`: delete the entire `@app.command() def bench(...)` block (lines 117-143 approximately) and any now-unused imports (`sys`, `from benchmark.run_bench import run_benchmark` stays gone).

- [ ] **Step 2: Drop tests that reference the deleted command**

Edit `tests/test_cli.py`: remove any test case that invokes `["bench", ...]` on the main `solvay` app. Keep the `solve`-related tests.

- [ ] **Step 3: Delete the legacy runner**

Run: `git rm benchmark/run_bench.py`

- [ ] **Step 4: Run the full test suite**

Run: `uv run python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/solvay/cli.py tests/test_cli.py
git commit -m "chore(bench): retire legacy solvay bench command and run_bench.py"
```

---

## Phase 6 — Documentation and demo run

### Task 31: `benchmark/README.md`

**Files:**
- Create: `benchmark/README.md`

- [ ] **Step 1: Write the README**

Create `benchmark/README.md`:

```markdown
# Solvay Benchmark Suite

Three-layer suite for evaluating Solvay: synthetic problem generation,
matrix execution across profiles and models, and metric reporting.

See `docs/superpowers/specs/2026-04-18-benchmark-design.md` for the full design.

## Quick start

    # Generate 5 synthetic mechanics problems
    uv run solvay-bench generate \
        --domain mechanics --compose pendulum,damping --n 5

    # Run the full matrix
    uv run solvay-bench run \
        --profiles all \
        --models anthropic:claude-sonnet-4-6 \
        --repeat 1 \
        --confirm-cost

    # Report
    uv run solvay-bench report benchmark/runs/<run-id>.jsonl \
        --out benchmark/reports/<run-id>.md

    # Longitudinal history
    uv run solvay-bench history --profile solvay-full --model anthropic:claude-sonnet-4-6

## Profiles

| Profile | Feeds |
|---|---|
| `bare` | Ladder B, bottom rung |
| `prompted` | Ladder B |
| `tooled` | Ladder B |
| `solvay-noweb` | Metric A + Ladder B |
| `solvay-full` | Ladder B top |

## Directory layout

- `problems/<domain>/` — versioned problem JSONs
- `verifications/` — optional per-problem verification scripts
- `runs/` — JSONL output of `solvay-bench run`
- `reports/` — markdown output of `solvay-bench report`
- `baseline_stats.json` — noise floor, refreshed with `--repeat 10`
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/README.md
git commit -m "docs(bench): add benchmark README"
```

---

### Task 32: Update CLAUDE.md and FUTURE.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `FUTURE.md`

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`, below the existing "How to run" block, add a "Benchmark" section:

```markdown
## Benchmark

    uv run solvay-bench generate --domain mechanics --n 5
    uv run solvay-bench run --profiles all --confirm-cost
    uv run solvay-bench report benchmark/runs/<run-id>.jsonl

Full design: `docs/superpowers/specs/2026-04-18-benchmark-design.md`.
Three layers: `src/solvay/benchmark/{generator,profiles,report}`.
```

- [ ] **Step 2: Update FUTURE.md**

In `FUTURE.md`, remove any line referring to longitudinal tracking (now in v1). Add, under a new "Benchmark suite — deferred" section:

```markdown
## Benchmark suite — deferred

- **Track C** — real-world utility benchmark (separate spec).
- **Olympiad scraper skill** — scrape recent olympiad problem archives.
- **Textbook ingester** — offline script over `/Users/enriquebook/Personal/Education/UCM`.
- **Other domains** — E&M, thermo, quantum, etc. (same pipeline, new batches).
- **`derivation` expected kind** — problems without a single final answer.
- **On-the-fly generation mode** — as alternative to the static dataset.
- **Per-role model mixing** within a Solvay profile (currently uniform per invocation).
- **HTML / interactive reports.**
- **CI gate** that blocks merges on regression above 2σ.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md FUTURE.md
git commit -m "docs: document benchmark suite and update deferred items"
```

---

### Task 33: Demo run and final commit

**Files:**
- Create (via runner): `benchmark/runs/demo.jsonl`
- Create (via reporter): `benchmark/reports/demo.md`

- [ ] **Step 1: Execute a small matrix against the committed dataset**

Run (token cost: small, one profile, one model, one repeat):
```bash
uv run solvay-bench run \
  --profiles solvay-full \
  --models anthropic:claude-sonnet-4-6 \
  --problems benchmark/problems \
  --out benchmark/runs/demo.jsonl
```

- [ ] **Step 2: Render the report**

Run:
```bash
uv run solvay-bench report \
  benchmark/runs/demo.jsonl \
  --problems benchmark/problems \
  --out benchmark/reports/demo.md \
  --no-error-histogram
```

- [ ] **Step 3: Inspect output**

Run: `uv run python -c "from pathlib import Path; print(Path('benchmark/reports/demo.md').read_text()[:600])"`
Expected: markdown with "Benchmark Report", metric A / B / D sections, and the `config_fingerprint` of the current HEAD.

- [ ] **Step 4: Commit**

```bash
git add benchmark/runs/demo.jsonl benchmark/reports/demo.md
git commit -m "docs(bench): capture demo run and report"
```

- [ ] **Step 5: Full test suite**

Run: `uv run python -m pytest -q`
Expected: all tests pass.

---

## Self-review checklist

Before handing off, spot-check:

1. **Spec coverage:**
   - Three-layer architecture (Tasks 1-9 runner, 15-20 generator, 22-29 report) ✓
   - Problem schema with Source / Expected / Contamination (Task 2) ✓
   - Synthetic generation with composer + verifier + probe (Tasks 16-19) ✓
   - Five profiles (Tasks 10-14) ✓
   - (problems × profiles × models × repeats) matrix runner (Task 8) ✓
   - `solvay_version` / `git_commit` / `git_dirty` / `config_fingerprint` in header (Tasks 5, 7, 8) ✓
   - Metrics A / B / D.1 / D.2 / D.3 (Tasks 23, 24, 25) ✓
   - Noise floor (Task 26) ✓
   - Regression verdict via 2σ (Task 27) ✓
   - Markdown report (Task 28) ✓
   - CLIs: generate / run / report / history (Tasks 9, 20, 29) ✓
   - Cost guard (Task 9) ✓
   - v1 done criteria demo (Task 33) ✓

2. **Placeholder scan:** no "TBD" / "implement later" / placeholder asserts anywhere.

3. **Type consistency:** `Profile.runner` signature matches implementations in Tasks 10-14; `ProfileResult` used uniformly; `RunRecord` fields match between writer (Task 7) and reader consumers (Tasks 22-27); `BenchConfig` fields referenced in Tasks 4, 16, 18, 25 all exist.
