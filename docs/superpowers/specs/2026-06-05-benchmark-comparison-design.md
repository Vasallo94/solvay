# Solvay Benchmark Comparison: Pipeline vs. Baseline

**Date**: 2026-06-05
**Status**: Draft
**Scope**: Generate hard physics problems, improve grading, add comparison tooling

---

## Goal

Measure whether Solvay's multi-agent pipeline (parser → researcher → solver → peer_reviewer → consolidator) actually improves physics problem-solving quality compared to a raw LLM call. The existing benchmark system has profiles (bare, prompted, tooled, solvay-full, solvay-noweb) and a matrix runner, but only 1 sample problem and regex-based grading.

## What's Missing

1. **Hard problems** — The existing sample problem is trivial (block on incline). We need 15-20 hard problems across domains, plus 6-8 hand-crafted novel problems.
2. **Better grading** — Current grading uses regex to extract numbers and compare with tolerance. This misses symbolic answers, conceptual answers, and expressions with different forms.
3. **Comparison tooling** — No way to compare results across profiles/runs in a structured way.

## Component 1: Problem Generation

### Generated Problems (15 hard problems, 3 per domain)

Use the existing generator pipeline (`solvay-bench generate`) with `difficulty="hard"` skeletons:

**Mechanics (3):**
- Gyroscope precession rate derivation
- Lagrangian with holonomic constraints (bead on rotating hoop)
- Coupled oscillators with damping (normal modes)

**Electromagnetism (3):**
- Magnetic mirror force on charged particle
- Rectangular waveguide cutoff frequency (TE/TM modes)
- Skin depth in conductor with alternating field

**Thermodynamics (3):**
- Joule-Thomson coefficient from van der Waals equation of state
- Finite-time Carnot engine efficiency (endoreversible)
- Wien displacement law derivation from Planck distribution

**Quantum Mechanics (3):**
- Tunneling through double rectangular barrier (resonances)
- WKB approximation for quartic potential energy levels
- First-order perturbation theory for anharmonic oscillator (x⁴ term)

**Waves (3):**
- Cherenkov radiation angle and threshold velocity
- Rayleigh scattering cross-section (wavelength dependence)
- Group velocity dispersion in a plasma

### Hand-Crafted Novel Problems (8, difficulty=hard)

Problems unlikely to be in training data, requiring multi-domain reasoning:

**Already tested (3):**
1. Quadrupole magnetic braking — em, expected: N = 32π/315 σB₀²ωR⁵
2. Acoustic Hawking radiation — waves+relativity, expected: T_H = ℏc_s/(k_BQ)
3. Phonon Casimir force — quantum+waves, expected: ΔE ~ d⁻³, F ~ d⁻⁴

**New (5):**
4. MHD Hartmann flow with thermal coupling — em+thermo, find: velocity and temperature profiles, Nusselt number vs Hartmann number
5. Synchrotron radiation angular distribution — em+relativity, find: critical harmonic number, power scaling with γ
6. Quantum tunneling through oscillating barrier (Floquet) — quantum, find: transmission coefficient with sidebands
7. Electromagnetic sail thrust — em+mechanics, find: effective cross-section, thrust force, specific impulse
8. Non-equilibrium Casimir force between plates at different temperatures — quantum+thermo, find: force and heat transfer scaling

Each is written as a JSON file following the `Problem` schema with `source.kind="synthetic"`, `difficulty="hard"`, and manually verified expected answers.

### Output

All problems written to `benchmark/problems/{domain}/` as JSON files.
Total: 23 problems (15 generated + 8 hand-crafted).

## Component 2: Improved Grading

Replace `_evaluate_correct()` in `runner.py` with a two-stage grader.

### Stage 1: SymPy Equivalence (deterministic, free)

```python
def _sympy_grade(answer_raw: str, expected: Expected) -> bool | None:
```

- Extract candidate expressions from `answer_raw` using regex (numbers, fractions, symbolic expressions)
- Parse both candidate and expected with `sympy.sympify()`
- For numeric: check `abs(candidate - expected) / abs(expected) <= tolerance`
- For symbolic: check `sympy.simplify(candidate - expected) == 0`
- Returns `True`/`False` if parseable, `None` if parsing fails

### Stage 2: LLM-as-Judge Fallback

```python
def _llm_grade(
    answer_raw: str, expected: Expected, problem: Problem, config: BenchConfig
) -> bool:
```

- Prompt a cheap model (configurable via `config.grader_model`):
  ```
  Compare this physics answer to the expected answer.
  Problem: {problem.statement}
  Student answer: {answer_raw}
  Expected: {expected.value} {expected.unit}
  Are they mathematically equivalent? Return JSON: {"correct": bool, "reason": "..."}
  ```
- Default grader model: `azure_openai:gpt-5.5-codex` (or configurable)
- Returns parsed `correct` field

### Combined Flow

```python
def _evaluate_correct(
    answer_raw: str, expected: Expected, problem: Problem, config: BenchConfig
) -> tuple[bool, str]:
    """Returns (correct, grading_method)."""
    sympy_result = _sympy_grade(answer_raw, expected)
    if sympy_result is not None:
        return sympy_result, "sympy"
    return _llm_grade(answer_raw, expected, problem, config), "llm-judge"
```

The `grading_method` field is added to `RunRecord` for analysis.

### Changes to BenchConfig

Add `grader_model: str` field (default: same as `DEFAULT_MODEL`).

### Changes to RunRecord

Add `grading_method: str` field ("regex", "sympy", or "llm-judge").

## Component 3: Compare Command

### CLI

```bash
solvay bench compare results.jsonl [--output report.md]
```

Reads a single JSONL file (which already contains multiple profiles from the matrix run) and produces a comparison table.

### Output Format

Markdown table to stdout:

```
# Solvay Benchmark Comparison
Model: azure_openai:gpt-5.5-codex | Problems: 23 | Date: 2026-06-05

| Problem                    | Domain  | Diff | bare | prompted | tooled | solvay-full |
|---------------------------|---------|------|------|----------|--------|-------------|
| gyroscope-precession       | mech    | hard | ✗    | ✗        | ~      | ✓           |
| quadrupole-braking         | em      | hard | ✗    | ✗        | ✗      | ✓           |
| acoustic-hawking           | waves   | hard | ✗    | ~        | ~      | ✓           |
| ...                        |         |      |      |          |        |             |

## Summary

| Metric              | bare | prompted | tooled | solvay-full |
|---------------------|------|----------|--------|-------------|
| Accuracy            | 15%  | 35%      | 50%    | 80%         |
| Avg time (s)        | 8    | 12       | 45     | 180         |
| Avg output tokens   | 1.5k | 2.5k     | 8k     | 20k         |
| Cost per problem    | $0.01| $0.02    | $0.05  | $0.15       |
```

If `--output` is provided, also writes to file.

### Implementation

New file: `src/solvay/benchmark/compare.py`
- `compare_run(path: Path) -> ComparisonReport`
- `format_markdown(report: ComparisonReport) -> str`

New CLI command in `src/solvay/benchmark/cli.py`:
- `@app.command("compare")`

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `benchmark/problems/{domain}/*.json` | Create | 15 generated + 8 hand-crafted problems |
| `src/solvay/benchmark/runner.py` | Modify | Improved grading (sympy + llm-judge) |
| `src/solvay/benchmark/schema.py` | Modify | Add `grading_method` to RunRecord |
| `src/solvay/benchmark/config.py` | Modify | Add `grader_model` field |
| `src/solvay/benchmark/compare.py` | Create | Comparison report generator |
| `src/solvay/benchmark/cli.py` | Modify | Add `compare` command |
| `tests/benchmark/test_compare.py` | Create | Tests for comparison tooling |
| `tests/benchmark/test_runner.py` | Modify | Tests for improved grading |

## Testing Strategy

- All existing benchmark tests must continue passing
- New tests for SymPy grading: numeric equivalence, symbolic equivalence, unparseable fallback
- New tests for LLM grading: mocked model, correct/incorrect verdicts
- New tests for compare command: JSONL parsing, table generation
- Integration test: run `bare` and `solvay-full` on 2-3 problems and verify compare output

## Non-Goals

- No UI/dashboard (markdown tables are sufficient)
- No statistical significance testing (not enough repeats yet)
- No automated CI integration (manual runs for now)
- No changes to the generator pipeline itself
