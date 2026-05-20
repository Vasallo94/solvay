# Solvay Benchmark Suite — Design Spec

**Date:** 2026-04-18
**Status:** Draft for review
**Supersedes:** Benchmark scaffolding from `2026-04-15-solvay-design.md` (the current `benchmark/run_bench.py` and `sample-mechanics-1.json` are placeholders to be replaced).

---

## 1. Context and motivation

The current benchmark (`benchmark/run_bench.py` + one Tipler problem) is a placeholder. Two structural problems make it inadequate for evaluating Solvay:

1. **Contamination.** Modern LLMs have memorized the standard physics curriculum — Tipler, Griffiths, Marion, Jackson — including answers. Running them against canonical textbook problems measures recall, not reasoning.
2. **Tool leakage.** When `web_search` is enabled, the model can retrieve the published solution, contaminating the measurement of internal reasoning.

Beyond contamination, the current setup answers only one question — "did Solvay get the right number?" — and conflates several distinct things we actually want to know:

- Does the model **reason** about physics, or does it **recall** problems it has seen?
- Does the multi-agent **orchestration** add value over a bare model, a prompted model, or a single agent with tools?
- Is Solvay **robust** — what kinds of errors does it make, can it detect its own failures, what does correctness cost in tokens?

This spec defines a benchmark suite that addresses each question explicitly.

---

## 2. Goals and non-goals

### Goals

1. Measure Solvay's reasoning capacity on problems the underlying model has not seen (metric **A**).
2. Measure the marginal value of each layer of the Solvay system over baselines (metric **B**, "the ladder").
3. Characterize Solvay's failure modes, cost-per-correct-answer, and self-calibration (metric **D**).
4. Provide a reproducible, versioned dataset and a runner that supports comparing Solvay across versions and across base models.
5. Build the runner and reporting infrastructure such that a future "real-world utility" benchmark (track C) can plug in without architectural changes.

### Non-goals (v1)

- Evaluating real-world / research-grade physics utility (track C). Deferred to a separate spec.
- Covering all physics domains. v1 ships mechanics only; the pipeline is designed to scale to other domains by re-running the generator.
- Beating published physics benchmarks (e.g., JEEBench, PhysicsQA).
- A web UI or interactive reports.
- CI gates that block merges on regression. The data to enable this is produced; the gate is future work.

---

## 3. Architecture

The benchmark splits into three layers communicating through file-based contracts. Each layer is independently developable, testable, and replaceable.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — PROBLEM PRODUCTION                               │
│  Three sources, all emit the same JSON schema               │
│  ├── Synthetic generator   (LLM + sympy, this spec)         │
│  ├── Olympiad scraper      (skill, future)                  │
│  └── Textbook ingester     (offline script, future)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
                  benchmark/problems/<domain>/*.json
                  (versioned static dataset)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — EXECUTION                                        │
│  Runner takes (problems × profiles × models) matrix         │
│  Profiles encapsulate "what apparatus runs":                │
│  ├── bare           (raw model, no system prompt, no tools) │
│  ├── prompted       (+ physicist system prompt)             │
│  ├── tooled         (+ local tools, no orchestration)       │
│  ├── solvay-noweb   (full Solvay, web disabled — feeds A)   │
│  └── solvay-full    (full Solvay including web search)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
                  benchmark/runs/<run-id>.jsonl
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — ANALYSIS AND REPORTING                           │
│  Pure post-processing over JSONL. No model calls.           │
│  ├── Accuracy per profile/model      (feeds A and B)        │
│  ├── Error taxonomy classifier       (feeds D.1)            │
│  ├── Cost-per-correct                (feeds D.2)            │
│  ├── Self-calibration analysis       (feeds D.3)            │
│  └── Longitudinal version tracking   (regression detection) │
└─────────────────────────────────────────────────────────────┘
```

**Why this separation:**

- **Layer 1 decoupled** — the future olympiad scraper and textbook ingester only need to emit JSON in the shared schema. They never touch the runner or reporter.
- **Layer 2 with profiles** — adding a new experimental condition (a new model, a Solvay variant without the verifier subagent, etc.) is a new entry in a profile registry, not a fork of the runner.
- **Layer 3 separated from execution** — old runs can be re-analyzed with new metrics without paying token cost again. Track C will reuse Layers 1 and 2; it will only need its own qualitative reporter in Layer 3.

---

## 4. Problem schema

A single JSON schema serves all three sources and both tracks. It extends the current minimal schema.

```json
{
  "id": "synth-mech-0042",
  "version": 1,

  "source": {
    "kind": "synthetic | olympiad | textbook",
    "origin": "solvay-generator-v0.3 | spain-olympiad-2024 | tipler-6th-ed | ucm-mechanics-2020-exam-3",
    "generated_at": "2026-04-18T14:22:00Z",
    "seed": 8473
  },

  "domain": "mechanics",
  "subdomain": "rigid-body-dynamics",
  "tags": ["pendulum", "magnetic-field", "non-linear-drag"],
  "difficulty": "intermediate",

  "statement": "A pendulum of length L and mass m carrying charge q swings...",
  "given": {
    "L": "1.2 m",
    "m": "0.5 kg",
    "q": "+2e-6 C",
    "B0": "0.4 T"
  },
  "find": "Equation of motion in the small-angle approximation.",

  "expected": {
    "kind": "symbolic | numeric",
    "value": "theta_ddot + (g/L)*theta + (q*B0/m)*theta_dot == 0",
    "unit": null,
    "tolerance_rel": 0.01,
    "verification": {
      "method": "sympy_equivalence | numeric_eval | dimensional_only",
      "script": "verifications/synth-mech-0042.py"
    }
  },

  "contamination": {
    "checked_at": "2026-04-18T14:25:00Z",
    "method": "model-recall-probe-v1",
    "score": 0.04,
    "verdict": "clean | suspect | contaminated"
  },

  "notes": "Composes pendulum + Lorentz force + linear drag. Backward-constructed from EOM."
}
```

### Field rationale

- **`source.kind`** is one of three values. UCM coursework, Tipler, and other published material are all `textbook` distinguished by `origin`. `kind` exists to let the reporter slice metrics by source type.
- **`given` / `find`** are separated from `statement` so that downstream code (verifier, peer reviewer, error classifier) can read structured data without re-parsing natural language. The `statement` is what the solver sees as a human-style prompt.
- **`expected.kind`** is `symbolic` (closed-form expression compared via sympy), `numeric` (value with relative tolerance), or `derivation` (process matters, not a single answer). v1 implements only `symbolic` and `numeric`. `derivation` is reserved in the schema for future use.
- **`expected.verification.script`** is optional. When present, it is a `.py` file under `benchmark/verifications/` that exposes a `verify(model_output: str, problem: dict) -> bool` function. Used for problems where verification is more nuanced than `value` + tolerance comparison.
- **`contamination`** is populated by the contamination probe (Section 5.3). Problems with `verdict != "clean"` remain in the dataset but are sliced separately by the reporter.

### Disk layout

```
benchmark/
  problems/
    mechanics/
      synth-mech-0001.json
      synth-mech-0002.json
      ...
    em/                          # future
    quantum/                     # future
  verifications/                 # optional per-problem scripts
    synth-mech-0042.py
  runs/                          # runner output (one JSONL per run)
    2026-04-18-solvay-full.jsonl
  reports/                       # reporter output
    2026-04-18-comparison.md
  baseline_stats.json            # noise floor calibration
```

---

## 5. Layer 1 — Synthetic generation pipeline

Lives in `src/solvay/benchmark/generator/`. Independent of the solver agent. Three stages, each with a single responsibility.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  COMPOSER    │ →   │  VERIFIER    │ →   │  CONTAMINATION   │
│  (LLM)       │     │  (sympy +    │     │  PROBE (LLM)     │
│              │     │   numeric)   │     │                  │
└──────────────┘     └──────────────┘     └──────────────────┘
       ↓                    ↓                       ↓
   problem.json        verified=True           clean=True
   (draft)             OR reject               OR flag
```

### 5.1 Composer

- **Input:** a "skeleton" specification — domain, list of physics concepts to compose, target difficulty.
  Example: `{"domain": "mechanics", "compose": ["pendulum", "lorentz-force", "linear-drag"], "approach": "backward"}`.
- **Output:** a complete problem in the JSON schema, including the closed-form solution in `expected.value`.

**Composition modes:**

- **Backward (preferred):** the composer first picks the answer (a closed-form expression or specific numeric value) and constructs the statement that leads to it. This is dramatically more reliable than "invent a problem and solve it." Used for the majority of v1 problems.
- **Forward:** compose a statement freely, then solve it. Marked with the appropriate `verification.method`. Used when backward construction is not natural for the concept.

The composer is an LLM call with a strong model (default Sonnet, configurable via `BenchConfig.composer_model`). It has access to a thinking scratchpad and can iterate internally before emitting the final JSON.

### 5.2 Verifier

Deterministic, no LLM. Validates that the proposed solution is consistent with the proposed statement.

Three modes selected by `verification.method`:

- **`sympy_equivalence`** — substitutes the values from `given` into the symbolic answer, simplifies, compares to a re-derivation if the statement permits.
- **`numeric_eval`** — numerically integrates the ODE/system implied by the statement (using scipy) and compares to `expected.value` within `tolerance_rel`.
- **`dimensional_only`** — last resort. Verifies only that the units of the answer are consistent with the units of `given` and `find`. Used for problems where no closed-form check is feasible.

If verification fails, the problem is rejected. The composer can regenerate with the failure message as feedback (loop with hard cap of N attempts; default N=3).

### 5.3 Contamination probe

LLM-based, uses a **different model from the one being benchmarked** to avoid measuring contamination with the same instrument that will be evaluated. Default: a cheap model from a different provider when available, otherwise a smaller variant of the same family.

Three probe questions:

1. **Recognition:** "Do you recognize this problem? What is the source?" — a confident, correct citation flags the problem.
2. **Continuation:** "Continue this passage: '<first 30% of statement>'" — if the continuation matches the rest of the statement above a threshold, flag.
3. **Cold answer:** "What is the answer to this problem?" without intermediate reasoning — if the model produces the correct answer with no work shown, suspicious.

Returns `score` in [0, 1] and `verdict` in `{clean, suspect, contaminated}`. Problems are kept in the dataset regardless of verdict; the reporter slices on it.

### 5.4 CLI

```bash
uv run solvay-bench generate \
  --domain mechanics \
  --compose pendulum,lorentz \
  --n 5 \
  --out benchmark/problems/mechanics/
```

Generates N problems, runs each through verifier and probe, writes survivors. Reports counts of generated, verified, contaminated, and surviving.

---

## 6. Layer 2 — Runner and profiles

Lives in `src/solvay/benchmark/`. Replaces `benchmark/run_bench.py`.

### 6.1 Profile concept

A profile is "what apparatus runs against the problem." Profiles do not bake in a model; the model is injected per call.

```python
@dataclass
class Profile:
    name: str
    description: str
    runner: Callable[[Problem, str, BenchConfig], ProfileResult]
    #                          ^^^ model_id
    cost_estimate: Literal["low", "medium", "high"]
```

Profiles register themselves in `src/solvay/benchmark/profiles/__init__.py`.

### 6.2 The five v1 profiles

| Profile | What it runs | Measures |
|---|---|---|
| `bare` | `client.messages.create(model, [user: statement])` — no system prompt, no tools | Raw model capacity |
| `prompted` | Same plus a "you are a physicist, reason step by step, give units" system prompt | Prompt engineering contribution |
| `tooled` | Single agent with `sympy`, `python_exec`, `dimensional` tools, **no web** | Local tools contribution |
| `solvay-noweb` | Full Solvay system with `TAVILY_API_KEY` disabled | **Metric A** (pure reasoning) and a rung of B |
| `solvay-full` | Full Solvay system with web search enabled | Top of ladder B |

The ladder `bare → prompted → tooled → solvay-noweb → solvay-full` falls out of running all five. Each rung is a column in the report.

`solvay-noweb` does double duty — it feeds metric A (no Internet) and is also a rung of ladder B.

### 6.3 Model dimension

The runner takes `--models` as an explicit axis. The full experiment matrix is `problems × profiles × models`. All profiles use the same model in a given row, so within-row comparisons isolate the apparatus (B), and across-row comparisons holding profile fixed isolate the model.

For multi-subagent profiles (`solvay-noweb`, `solvay-full`), the injected model is applied uniformly across all subagent roles. Per-role model mixing remains in `FUTURE.md`.

Defaults: `--models` defaults to the model configured in `SolvayConfig`. Running the full matrix requires explicit `--profiles all` and `--confirm-cost` if the estimated invocations exceed a threshold.

### 6.4 Repeats and noise floor

Single runs are noisy. The runner supports `--repeat N` to execute the same `(problem × profile × model)` cell N times with the same `config_fingerprint`. Each repetition is a distinct row with `repeat_idx`.

The reporter (Section 7.4) uses repeat data to compute the per-cell standard deviation, which serves as the regression-detection threshold.

### 6.5 Run output

JSONL, one line per `(problem × profile × model × repeat)`:

```json
{
  "run_id": "2026-04-18-14-30",
  "solvay_version": "0.3.1",
  "git_commit": "a1b2c3d",
  "git_dirty": false,
  "config_fingerprint": "sha256:7f3a...",
  "problem_id": "synth-mech-0042",
  "profile": "solvay-full",
  "model": "claude-sonnet-4-6",
  "repeat_idx": 0,
  "answer_raw": "...",
  "answer_extracted": "9.81 m/s^2",
  "correct": true,
  "elapsed_seconds": 23.4,
  "tokens": {"input": 4521, "output": 1832, "cache_read": 0},
  "error": null,
  "trace_id": "langsmith://..."
}
```

### 6.6 CLI

```bash
# Full matrix: 5 profiles × 3 models × 15 problems = 225 invocations
uv run solvay-bench run \
  --profiles all \
  --models claude-sonnet-4-6,claude-opus-4-7,claude-haiku-4-5 \
  --confirm-cost

# Subset
uv run solvay-bench run --profiles solvay-full --models claude-sonnet-4-6
uv run solvay-bench run --domain mechanics --profiles bare,solvay-full

# Noise calibration
uv run solvay-bench run --profiles all --repeat 10
```

---

## 7. Layer 3 — Metrics and reporting

Lives in `src/solvay/benchmark/report/`. Pure post-processing — no model calls, no token spend (except the error classifier in D.1, which is a one-shot pass over failures using a cheap model).

### 7.1 Metric A — pure reasoning

```
A(model) = accuracy of profile "solvay-noweb" with that model,
           restricted to problems where contamination.verdict == "clean"
```

Two filters: no Internet, no problems the model is likely to have seen.

### 7.2 Metric B — system delta (the ladder)

For each model, accuracy at every rung:

```
       bare   prompted   tooled   solvay-noweb   solvay-full
Sonnet  0.20   0.35       0.55      0.65            0.75
Opus    0.40   0.50       0.65      0.72            0.78
Haiku   0.10   0.20       0.40      0.50            0.60
```

The headline number is `solvay-full - bare` per model. The full table reveals where each layer contributes — if `solvay-full - tooled ≈ 0`, the multi-agent orchestration is adding nothing and we should know.

B is **model-dependent**: the apparatus may help Haiku substantially and contribute little to Opus. That asymmetry is itself a finding.

### 7.3 Metric D — robustness

Three sub-metrics, all computed from the JSONL with no re-execution.

**D.1 — Error taxonomy.** A post-hoc LLM classifier (cheap model) reads `answer_raw` of every failure and assigns one of:

- `wrong_concept` — applied the wrong law or framework
- `algebra_error` — set up correctly, computational mistake
- `units_error` — magnitude correct, unit wrong
- `dimensional_inconsistency` — answer dimensions inconsistent with expected
- `incomplete` — never produced a final answer
- `hallucinated_data` — invented quantities not given

Histogram per profile and per model.

**D.2 — Cost per correct answer.**

```
cost_per_correct(profile, model) = sum(tokens_total) / count(correct)
```

The economic question: does Solvay-full justify its 4× cost over `tooled`?

**D.3 — Self-calibration** (Solvay profiles only). When Solvay's internal verifier or peer reviewer flags uncertainty, does the answer turn out to be wrong? Reported as:

- **True flag rate**: of failures, what fraction did Solvay flag as uncertain
- **False flag rate**: of correct answers, what fraction did Solvay incorrectly flag

A well-calibrated system has high true-flag and low false-flag rates.

### 7.4 Versioning and regression detection

Three levels of versioning are recorded in every run:

| Level | Source | Purpose |
|---|---|---|
| `solvay_version` | `pyproject.toml` semver | Human-readable label |
| `git_commit` | `git rev-parse --short HEAD` | Exact reproducibility |
| `config_fingerprint` | sha256 of: `prompts/*.md` content + active subagents list + model-by-role mapping + tools list | Behavior fingerprint |

The fingerprint is the load-bearing field. Two runs with the same fingerprint are comparable as "the same Solvay" even if commits differ (typo fixes, README, refactors do not change the fingerprint). Two runs with different fingerprints differ in something that can affect output.

`git_dirty: true` flags runs with uncommitted changes. Such runs are not reproducible and the reporter marks them.

Implementation: `src/solvay/benchmark/fingerprint.py` enumerates the inputs deterministically and hashes them.

**Noise floor calibration.** A periodic `--repeat 10` run on the full dataset with the latest stable fingerprint produces `benchmark/baseline_stats.json` containing `σ(profile, model)` for accuracy. Frequency: once per release or whenever the model/provider changes.

**Regression detection.** A reported delta is significant only if `|Δ| > 2σ_baseline`. The reporter prints this annotation:

```
Solvay v0.3.1  fingerprint=7f3...  accuracy=0.68 ± 0.04 (2σ baseline)

Δ vs v0.3.0 = -0.03  →  within noise, NOT a regression
Δ vs v0.2.0 = -0.07  →  exceeds 2σ, real regression ⚠
```

### 7.5 Reports

Markdown to `benchmark/reports/<run-id>.md`:

- Summary table (models × profiles, accuracy)
- A by model (filtered to clean)
- B ladder by model
- D.1 error histogram
- D.2 cost-per-correct table
- D.3 calibration table (Solvay profiles)
- Regression annotations vs prior fingerprints
- Appendix: failed problems with `trace_id` links to LangSmith

### 7.6 CLI

```bash
# Single run
uv run solvay-bench report benchmark/runs/2026-04-18-14-30.jsonl

# Compare two runs (regression analysis)
uv run solvay-bench report --compare run-v1.jsonl run-v2.jsonl

# Single metric
uv run solvay-bench report --metric A benchmark/runs/...

# Longitudinal evolution by fingerprint
uv run solvay-bench history --metric accuracy --profile solvay-full --model claude-sonnet-4-6
```

---

## 8. v1 deliverables

### In scope

**Layer 1 — generation**
- `src/solvay/benchmark/generator/` with composer, verifier, contamination probe
- `solvay-bench generate` CLI
- 15-20 validated mechanics problems in `benchmark/problems/mechanics/`
- Internal regeneration loop with feedback (cap N=3 attempts)

**Layer 2 — execution**
- `src/solvay/benchmark/profiles/` with all five profiles
- Runner with `problems × profiles × models` matrix and `--repeat`
- JSONL output with full version provenance
- `src/solvay/benchmark/fingerprint.py`
- `solvay-bench run` CLI with `--confirm-cost` guard

**Layer 3 — reporting**
- `src/solvay/benchmark/report/` with A, B (full ladder), D.1/D.2/D.3 calculators
- Post-hoc error classifier (cheap LLM, one pass over failures)
- Noise floor measurement and storage in `benchmark/baseline_stats.json`
- Regression detection based on `2σ`
- `solvay-bench report`, `--compare`, `history`, `--metric`
- Markdown output in `benchmark/reports/`

**Schema and migration**
- New JSON schema documented
- The existing `sample-mechanics-1.json` migrated to the new schema
- Tests for schema validation, fingerprint determinism, profile execution, A/B/D evaluation, regression detection

**Documentation**
- `benchmark/README.md` covering generate / run / report
- `CLAUDE.md` updated with the new structure
- One full demo run committed and linked from the README

### Out of scope (deferred to `FUTURE.md`)

- Track C — real-world utility benchmark (separate spec)
- Olympiad scraper skill
- Textbook ingester for `/Users/enriquebook/Personal/Education/UCM`
- Domains beyond mechanics (E&M, thermo, quantum, etc. — same pipeline, just re-run generation)
- `derivation` mode in the schema (problems without a single answer)
- On-the-fly problem generation (alternative to static dataset)
- Per-role model mixing in profiles (uniform model per profile in v1)
- HTML / interactive reports
- CI gate that blocks merges on regression

### v1 done criteria

1. `solvay-bench generate --domain mechanics --n 20` produces 20 valid problems (verified, with contamination probe).
2. `solvay-bench run --profiles all --models <default>` executes the full matrix without errors.
3. `solvay-bench report <run.jsonl>` produces a readable markdown with A, B (ladder), D.1/D.2/D.3.
4. `solvay-bench history` shows evolution by `config_fingerprint`.
5. Tests cover: schema validation, fingerprint determinism, each profile executes, A/B/D evaluation, regression detection.
6. One demo run is documented in the `benchmark/README.md`.

---

## 9. Open questions for implementation

- **Composer prompt design** — the backward-construction approach is the riskiest part of the pipeline. Worth iterating on the prompt with a small held-out set before committing to N=20 problems.
- **Contamination probe model** — choice of a "different" model is non-trivial. If only Claude is available, a smaller variant must suffice; otherwise pick one open-weight option (e.g., a local model via the deferred Ollama path).
- **Cost guard threshold** — what invocation count triggers `--confirm-cost`? Initial proposal: 100.
- **Token accounting for multi-agent profiles** — Solvay subagents make many internal calls; we want totals across the whole tree per problem. Verify that deepagents exposes per-invocation token counts at the orchestrator level.

These do not block the design but should be addressed in the implementation plan.
