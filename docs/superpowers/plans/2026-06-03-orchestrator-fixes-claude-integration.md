# Orchestrator Fixes + Claude Code Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three orchestrator routing bugs (solver never called, peer reviewer misses analytical errors, parser called twice) and integrate Solvay with Claude Code via a skill and MCP server.

**Architecture:** Two independent work streams. Stream A fixes orchestrator bugs by disabling the auto-injected general-purpose subagent, hardening prompts, and adding a physics checklist tool. Stream B adds Claude Code integration via a CLI skill (phase 1) and a FastMCP stdio server (phase 2).

**Tech Stack:** deepagents v0.6+, LangGraph, FastMCP (Python `fastmcp` package), Typer CLI, Pydantic schemas.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/solvay/agent.py` | Modify | Add GP-as-solver clone to block auto-injection |
| `src/solvay/prompts/orchestrator.md` | Modify | Harden anti-loop, anti-GP, step-tracking rules |
| `src/solvay/prompts/peer_reviewer.md` | Modify | Add domain-specific analytical verification checklist |
| `src/solvay/tools/physics_checklist.py` | Create | Generate verification checks per physics domain |
| `src/solvay/subagents/peer_reviewer.py` | Modify | Register physics_checklist in tools list |
| `tests/test_physics_checklist.py` | Create | Unit tests for checklist tool |
| `.claude/skills/solve-physics/SKILL.md` | Create | Claude Code slash command for Solvay |
| `src/solvay/mcp_server.py` | Create | FastMCP stdio server with 3 tools |
| `tests/test_mcp_server.py` | Create | Unit tests for MCP tool handlers |
| `pyproject.toml` | Modify | Add `fastmcp` dependency |
| `.mcp.json` | Create | MCP server configuration for Claude Code |

---

## Stream A: Fix Orchestrator Bugs

### Task 1: Disable auto-injected general-purpose subagent

**Files:**
- Modify: `src/solvay/agent.py:69-109`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_no_auto_general_purpose_subagent() -> None:
    """The subagents list must include a 'general-purpose' entry that maps to
    the solver, preventing deepagents from auto-injecting its own."""
    from unittest.mock import patch

    captured_subagents: list = []

    original_create = None

    def spy_create_deep_agent(*args: object, **kwargs: object) -> object:
        captured_subagents.extend(kwargs.get("subagents", []))
        return original_create(*args, **kwargs)

    import deepagents
    original_create = deepagents.create_deep_agent

    with patch("solvay.agent.create_deep_agent", side_effect=spy_create_deep_agent):
        from solvay.agent import create_solvay_agent
        from solvay.config import SolvayConfig
        config = SolvayConfig(default_model="fake-model")
        try:
            create_solvay_agent(config)
        except Exception:
            pass  # Model resolution may fail; we only care about the subagents list

    names = [s.get("name") or s["name"] for s in captured_subagents]
    assert "general-purpose" in names, (
        f"Expected 'general-purpose' in subagents to block auto-injection, got: {names}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_no_auto_general_purpose_subagent -v`
Expected: FAIL — `"general-purpose"` not in subagent names

- [ ] **Step 3: Implement the fix**

Edit `src/solvay/agent.py`. After `solver = create_solver_subagent(...)` (line 87), add the GP clone. Then include it in the subagents list:

```python
def create_solvay_agent(
    config: SolvayConfig | None = None,
) -> Any:
    """Create the fully wired Solvay orchestrator agent."""
    if config is None:
        config = SolvayConfig()

    web_search = _create_web_search_tool()

    parser = create_parser_subagent(config)
    researcher = create_researcher_subagent(config, web_search, url_fetch)
    solver = create_solver_subagent(config, web_search, url_fetch)
    verifier = create_verifier_subagent(config)
    peer_reviewer = create_peer_reviewer_subagent(config, web_search)
    consolidator = create_consolidator_subagent(config)

    # Provide an explicit "general-purpose" subagent that routes to the solver
    # graph. This prevents deepagents from auto-injecting its own GP subagent
    # (see deepagents/graph.py:618-619), which biases the orchestrator LLM
    # away from calling "solver".
    gp_as_solver = CompiledSubAgent(
        name="general-purpose",
        description=solver["description"],
        runnable=solver["runnable"],
    )

    agent = create_deep_agent(
        model=config.model_for("orchestrator"),
        tools=[web_search],
        system_prompt=load_prompt("orchestrator"),
        memory=memory_paths(config.harness),
        backend=build_backend(),
        permissions=build_permissions(),
        subagents=[
            cast(SubAgent, parser),
            cast(SubAgent, researcher),
            solver,
            cast(SubAgent, verifier),
            cast(SubAgent, peer_reviewer),
            cast(SubAgent, consolidator),
            gp_as_solver,
        ],
    )

    return agent
```

Also add the import at the top of the file:

```python
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
```

(Replace the existing `from deepagents import SubAgent, create_deep_agent` line.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py::test_no_auto_general_purpose_subagent -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: 132+ passed

- [ ] **Step 6: Commit**

```bash
git add src/solvay/agent.py tests/test_agent.py
git commit -m "fix: disable auto-injected general-purpose subagent

Clone solver as 'general-purpose' to prevent deepagents from
auto-injecting its own GP subagent, which biased the orchestrator
LLM away from calling the solver subagent."
```

---

### Task 2: Harden orchestrator prompt

**Files:**
- Modify: `src/solvay/prompts/orchestrator.md`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Update the orchestrator prompt**

Replace the full content of `src/solvay/prompts/orchestrator.md` with:

```markdown
# Solvay Orchestrator

You are the orchestrator of a physics problem-solving research team.
You NEVER solve physics yourself. Your ONLY job is to call subagents
in the exact order below using the `task` tool, then present their output.

## CRITICAL: ALLOWED SUBAGENT TYPES

**You MUST only use these exact subagent_type values:**
- `"parser"` — parses the problem into a structured spec
- `"researcher"` — finds relevant physics principles and equations
- `"solver"` — solves the problem through an iterative loop
- `"peer_reviewer"` — reviews the solution draft
- `"consolidator"` — produces the final polished answer

**Calling `"general-purpose"` or any other subagent_type is FORBIDDEN and will
produce degraded results. Only the 5 types listed above are valid.**

## MANDATORY WORKFLOW — exactly 6 subagent calls, in order

**BUDGET: You have a maximum of 8 subagent calls total across the entire run.**
Plan accordingly. Do NOT waste calls on retries or re-parsing.

**ANTI-LOOP RULES (hard limits):**
- `parser`: EXACTLY 1 call. Once you receive a ProblemSpec, NEVER call parser again.
- `researcher`: at most 2 calls (Step 2, and optionally Step 4a).
- `solver`: at most 2 calls (Step 3, and optionally Step 4b).
- `peer_reviewer`: EXACTLY 1 call (Step 5 only).
- `consolidator`: EXACTLY 1 call (Step 6 only).

**Progress tracker — check off each step as you complete it:**

- [ ] Step 1: PARSE
- [ ] Step 2: RESEARCH
- [ ] Step 3: SOLVE
- [ ] Step 4: HANDLE SOLVER RESULT
- [ ] Step 5: PEER REVIEW
- [ ] Step 6: CONSOLIDATE

**Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Present answer**

### Step 1: PARSE

```
task(subagent_type="parser", description="<problem statement>")
```

You will receive a `ProblemSpec` JSON. Save it — you need it for steps 2 and 3.
After receiving the ProblemSpec, IMMEDIATELY proceed to Step 2.

**CRITICAL: Call `parser` EXACTLY ONCE. Never call it again after Step 1.**
Do NOT retry, re-parse, or validate the result — accept whatever the parser returns and move on.

### Step 2: RESEARCH

```
task(subagent_type="researcher", description="<ProblemSpec JSON as a string>")
```

You will receive a `ResearchBrief` JSON. Save it — you need it for step 3.
After receiving the ResearchBrief, IMMEDIATELY proceed to Step 3.

### Step 3: SOLVE

```
task(subagent_type="solver", description="Problem spec:\n<ProblemSpec JSON>\n\nResearch:\n<ResearchBrief JSON>")
```

You will receive solver results including `final_draft` and `termination_reason`.
After receiving the result, IMMEDIATELY proceed to Step 4.

### Step 4: HANDLE SOLVER RESULT

- If `termination_reason == "consensus"` or `"budget_exhausted"`: go to Step 5.
- If `termination_reason == "judge_forced"`:
  a. Call `task(subagent_type="researcher", ...)` again focused on `blocked_topic`.
  b. Call `task(subagent_type="solver", ...)` one more time.
  c. If still `judge_forced`, go to Step 5 with best-effort draft.

### Step 5: PEER REVIEW

```
task(subagent_type="peer_reviewer", description="<SolutionDraft JSON as a string>")
```

You will receive a verdict. After receiving it, IMMEDIATELY proceed to Step 6.

### Step 6: CONSOLIDATE

```
task(subagent_type="consolidator", description="<all prior outputs as plain text>")
```

You will receive the final polished answer. Present it to the user.

## Output format

After Step 6, present the consolidator's answer verbatim.

## Rules

- NEVER solve the problem yourself. NEVER produce a final answer before Step 6.
- After each step, your next message MUST be a `task` tool call for the next step.
- The `subagent_type` MUST be one of the 5 values above. DO NOT use "general-purpose".
- The `description` must contain all context the subagent needs (no shared memory).
- Pass data between steps as JSON strings, not Python dicts.
- Maximum ONE re-research attempt per run.
- Write a brief entry to `/workspace/lab_notebook.md` at the end summarizing the run.
- If a tool fails, append a short note to `/memories/solvay/harness_notes.md`:
  - Symptom: ...
  - Context: ...
  - Likely cause: ...
  - Suggested fix: ...
- All output in English.
```

- [ ] **Step 2: Run existing prompt tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS (tests verify prompt files exist and are non-empty)

- [ ] **Step 3: Commit**

```bash
git add src/solvay/prompts/orchestrator.md
git commit -m "fix: harden orchestrator prompt against GP calls and parser loops

Add explicit budget cap (8 calls), progress tracker checkboxes,
and stronger anti-GP language to prevent the LLM from calling
general-purpose instead of solver."
```

---

### Task 3: Enhance peer reviewer prompt

**Files:**
- Modify: `src/solvay/prompts/peer_reviewer.md`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Update the peer reviewer prompt**

Replace the full content of `src/solvay/prompts/peer_reviewer.md` with:

```markdown
# Solvay Peer Reviewer

You are the peer reviewer agent. Your job is semantic critique: assessing whether
the approach is correct, complete, and well-justified.

## Input

- A `SolutionDraft` to review
- The `ProblemSpec` for reference
- `/workspace/lab_notebook.md` for context

## Output (Verdict)

Return a JSON object with:
- `approved`: true if the approach is sound
- `issues`: list of specific concerns (empty if approved)
- `severity`: "blocker" | "minor" | "none"

## Tools

- `python_exec(code)` -- verify specific claims computationally
- `web_search(query)` -- rarely; only to cross-check known reference values
- `physics_checklist(domain, knowns, unknowns)` -- generate domain-specific verification checks

## Review criteria

1. **Approach correctness** -- is the method appropriate for this problem?
2. **Missing terms** -- are there forces, fields, or effects being ignored?
3. **Assumptions** -- are they justified? Are there hidden assumptions?
4. **Better methods** -- would a different approach be more suitable?
5. **Logical consistency** -- do the steps follow logically?

## MANDATORY: Analytical Verification

Before approving ANY solution, you MUST run these checks using `python_exec`.
Start by calling `physics_checklist` with the problem's domain to get the
domain-specific checks, then execute each one.

### Electromagnetism problems (domain="em")

- **Boundary conditions**: Compute J dot n_hat at all conductor surfaces.
  It MUST be zero for finite conductors. If it is not zero, the solution
  is MISSING an electric field correction (Laplace equation for the potential).
- **Current conservation**: Verify div(J) = 0 everywhere inside the conductor.
- **Energy conservation**: Confirm P_Joule = N * omega (Joule dissipation
  equals mechanical power lost to braking torque).

### Asymptotic scaling problems

- **Dominant term extraction**: If the answer claims F ~ d^(-n), take the
  derivative of the FULL expression (not just the envelope) and verify which
  term actually dominates at large d. Oscillatory terms like sin(kd)/d have
  derivative ~ cos(kd)/d which is O(1/d), NOT O(1/d^2).
- **Limiting cases**: Check d -> 0, d -> infinity, and all parameters -> 0
  or -> infinity. The answer must reduce to known limiting cases.

### Quantum mechanics problems (domain="quantum")

- **Normalization**: Verify that wavefunctions are normalized.
- **Hermiticity**: Check that operators are Hermitian.
- **Correspondence principle**: In the classical limit (hbar -> 0 or
  large quantum numbers), the result must reduce to the classical answer.

### All domains

- **Dimensional analysis**: Run check_dimensions on every intermediate
  and final result. Every equation must be dimensionally consistent.
- **Known limits**: Verify that the answer reduces to known results in
  special cases (non-relativistic limit, weak-field limit, etc.).
- **Conservation laws**: Check energy, momentum, and angular momentum
  conservation where applicable.

## Rules

- You are a physics expert reviewing methodology AND checking analytical details.
- Mark severity="blocker" for wrong approaches, missing boundary conditions,
  incorrect dominant-term extraction, or critical missing terms.
- Before acting: read `/workspace/lab_notebook.md`.
- After acting: append a brief entry (at most 4 lines) to
  `/workspace/lab_notebook.md`.
- If a tool fails, a path is confusing, dependencies are missing, structured
  output repeatedly fails, or the harness behavior blocks the task, append a
  short note to `/memories/solvay/harness_notes.md` using:
  - Symptom: ...
  - Context: ...
  - Likely cause: ...
  - Suggested fix: ...
  - Artifacts: ...
- Never include secrets, raw API keys, passwords, or private credentials in
  harness notes.
- All output in English.
```

- [ ] **Step 2: Run existing prompt tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/solvay/prompts/peer_reviewer.md
git commit -m "fix: enhance peer reviewer with domain-specific analytical checks

Add mandatory verification checklist for E&M boundary conditions,
asymptotic dominant terms, QM normalization, and dimensional analysis.
Addresses stress test failures where peer reviewer missed J·n≠0 and
incorrect force scaling."
```

---

### Task 4: Create physics_checklist tool

**Files:**
- Create: `src/solvay/tools/physics_checklist.py`
- Test: `tests/test_physics_checklist.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_physics_checklist.py`:

```python
"""Tests for the physics_checklist tool."""

import ast

import pytest

from solvay.tools.physics_checklist import physics_checklist


class TestPhysicsChecklist:
    def test_em_domain_returns_checks(self) -> None:
        result = physics_checklist(domain="em", knowns=["R", "sigma", "B0"], unknowns=["J", "N"])
        assert "checks" in result
        assert "domain_notes" in result
        assert len(result["checks"]) > 0

    def test_em_has_boundary_condition_check(self) -> None:
        result = physics_checklist(domain="em", knowns=["R"], unknowns=["J"])
        names = [c["name"] for c in result["checks"]]
        assert any("boundary" in n.lower() for n in names)

    def test_mechanics_domain_returns_checks(self) -> None:
        result = physics_checklist(domain="mechanics", knowns=["m", "g"], unknowns=["v"])
        assert len(result["checks"]) > 0

    def test_quantum_domain_returns_checks(self) -> None:
        result = physics_checklist(domain="quantum", knowns=["hbar"], unknowns=["psi"])
        assert len(result["checks"]) > 0

    def test_thermo_domain_returns_checks(self) -> None:
        result = physics_checklist(domain="thermo", knowns=["T"], unknowns=["S"])
        assert len(result["checks"]) > 0

    def test_unknown_domain_returns_generic_fallback(self) -> None:
        result = physics_checklist(domain="unknown_domain", knowns=[], unknowns=[])
        assert len(result["checks"]) > 0
        names = [c["name"] for c in result["checks"]]
        assert any("dimension" in n.lower() for n in names)

    def test_check_fields_are_complete(self) -> None:
        result = physics_checklist(domain="em", knowns=["R"], unknowns=["J"])
        for check in result["checks"]:
            assert "name" in check, f"Check missing 'name': {check}"
            assert "description" in check, f"Check missing 'description': {check}"
            assert "python_code" in check, f"Check missing 'python_code': {check}"

    def test_python_code_is_syntactically_valid(self) -> None:
        result = physics_checklist(domain="em", knowns=["R"], unknowns=["J"])
        for check in result["checks"]:
            code = check["python_code"]
            try:
                ast.parse(code)
            except SyntaxError as e:
                pytest.fail(f"Invalid Python in check '{check['name']}': {e}")

    def test_all_domains_produce_dimensional_check(self) -> None:
        for domain in ["em", "mechanics", "quantum", "thermo", "waves", "relativity", "other"]:
            result = physics_checklist(domain=domain, knowns=["x"], unknowns=["y"])
            names = [c["name"] for c in result["checks"]]
            assert any("dimension" in n.lower() for n in names), (
                f"Domain '{domain}' missing dimensional analysis check"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physics_checklist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'solvay.tools.physics_checklist'`

- [ ] **Step 3: Implement the tool**

Create `src/solvay/tools/physics_checklist.py`:

```python
"""Domain-specific physics verification checklist generator."""

from __future__ import annotations

_CHECK = dict[str, str]


def _generic_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    """Checks applicable to every physics domain."""
    unknown_str = ", ".join(unknowns) if unknowns else "result"
    return [
        {
            "name": "dimensional_analysis",
            "description": (
                f"Verify that {unknown_str} has the correct physical dimensions. "
                "Use sympy.physics.units to check every intermediate and final expression."
            ),
            "python_code": (
                "from sympy.physics.units import meter, second, kilogram, newton\n"
                "# Replace with actual expressions from the solution:\n"
                "# convert_to(expr, target_unit, unit_system=SI)\n"
                "print('Dimensional analysis: manually verify each step')"
            ),
        },
        {
            "name": "energy_conservation",
            "description": (
                "Verify that total energy is conserved (or that the energy "
                "budget closes: input energy = output energy + dissipation)."
            ),
            "python_code": (
                "# Compare total input energy with output + losses.\n"
                "# E_in = E_out + E_dissipated\n"
                "print('Energy conservation: manually verify')"
            ),
        },
        {
            "name": "limiting_cases",
            "description": (
                "Check the answer in known limiting cases: parameters -> 0, "
                "parameters -> infinity, and any known special-case results."
            ),
            "python_code": (
                "from sympy import symbols, limit, oo\n"
                "# x = symbols('x')\n"
                "# print(limit(answer_expr, x, 0))\n"
                "# print(limit(answer_expr, x, oo))\n"
                "print('Limiting cases: manually verify')"
            ),
        },
    ]


def _em_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    return [
        {
            "name": "boundary_condition_j_dot_n",
            "description": (
                "For a finite conductor, compute J·n_hat at every boundary surface. "
                "The normal component of current density MUST be zero at the surface. "
                "If J has a nonzero normal component, the solution is missing an "
                "electric field correction (solve Laplace's equation for the potential)."
            ),
            "python_code": (
                "from sympy import symbols, sin, cos, simplify\n"
                "r, theta, phi = symbols('r theta phi')\n"
                "# Substitute the solution's J expression and compute J·r_hat at r=R.\n"
                "# J_r_at_surface = J_z * cos(theta)  # for J in z-direction\n"
                "# If this is nonzero, flag as BLOCKER.\n"
                "print('Boundary check: compute J·n_hat at r=R')"
            ),
        },
        {
            "name": "current_conservation_div_j",
            "description": (
                "Verify that div(J) = 0 everywhere inside the conductor. "
                "In Cartesian: dJx/dx + dJy/dy + dJz/dz = 0."
            ),
            "python_code": (
                "from sympy import symbols, diff\n"
                "x, y, z = symbols('x y z')\n"
                "# Substitute J components and compute divergence.\n"
                "# div_J = diff(Jx, x) + diff(Jy, y) + diff(Jz, z)\n"
                "# assert simplify(div_J) == 0\n"
                "print('Current conservation: compute div(J)')"
            ),
        },
        {
            "name": "poynting_energy_balance",
            "description": (
                "Verify that P_Joule = P_mechanical. The Joule dissipation rate "
                "integral(J^2 / sigma, dV) must equal the mechanical power lost "
                "to braking (torque * angular velocity)."
            ),
            "python_code": (
                "# P_joule = integral(J**2 / sigma, volume)\n"
                "# P_mech = N * omega\n"
                "# assert simplify(P_joule - P_mech) == 0\n"
                "print('Energy balance: compare Joule and mechanical power')"
            ),
        },
    ]


def _mechanics_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    return [
        {
            "name": "momentum_conservation",
            "description": (
                "Verify momentum conservation: if no external forces act on the "
                "system, total momentum must be constant."
            ),
            "python_code": (
                "# p_initial = m * v_initial\n"
                "# p_final = m * v_final\n"
                "# assert p_initial == p_final  (if no external forces)\n"
                "print('Momentum conservation: manually verify')"
            ),
        },
        {
            "name": "limiting_mass_cases",
            "description": (
                "Check the answer in the limits m -> 0 and m -> infinity. "
                "The result should reduce to known physics in these extremes."
            ),
            "python_code": (
                "from sympy import symbols, limit, oo\n"
                "m = symbols('m', positive=True)\n"
                "# print(limit(answer_expr, m, 0))\n"
                "# print(limit(answer_expr, m, oo))\n"
                "print('Mass limits: manually verify')"
            ),
        },
    ]


def _quantum_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    return [
        {
            "name": "normalization",
            "description": (
                "Verify that wavefunctions are normalized: "
                "integral(|psi|^2, dx) = 1 over the appropriate domain."
            ),
            "python_code": (
                "from sympy import symbols, integrate, conjugate, oo\n"
                "x = symbols('x')\n"
                "# norm = integrate(conjugate(psi) * psi, (x, -oo, oo))\n"
                "# assert simplify(norm - 1) == 0\n"
                "print('Normalization: manually verify')"
            ),
        },
        {
            "name": "hermiticity",
            "description": (
                "Verify that all operators used are Hermitian: "
                "<f|O|g> = <g|O|f>* for any states f, g."
            ),
            "python_code": (
                "# Check that the operator equals its adjoint.\n"
                "print('Hermiticity: manually verify operators')"
            ),
        },
        {
            "name": "correspondence_principle",
            "description": (
                "In the classical limit (hbar -> 0 or large quantum numbers), "
                "the result must reduce to the classical answer."
            ),
            "python_code": (
                "from sympy import symbols, limit\n"
                "hbar = symbols('hbar', positive=True)\n"
                "# classical_limit = limit(quantum_result, hbar, 0)\n"
                "print('Correspondence principle: manually verify')"
            ),
        },
    ]


def _thermo_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    return [
        {
            "name": "second_law",
            "description": (
                "Verify that total entropy change dS >= 0 for irreversible processes, "
                "or dS = 0 for reversible processes."
            ),
            "python_code": (
                "# dS_total = dS_system + dS_surroundings\n"
                "# assert dS_total >= 0\n"
                "print('Second law: manually verify entropy')"
            ),
        },
        {
            "name": "temperature_limits",
            "description": (
                "Check the answer in the limits T -> 0 (third law behavior) "
                "and T -> infinity (classical equipartition)."
            ),
            "python_code": (
                "from sympy import symbols, limit, oo\n"
                "T = symbols('T', positive=True)\n"
                "# print(limit(answer_expr, T, 0))\n"
                "# print(limit(answer_expr, T, oo))\n"
                "print('Temperature limits: manually verify')"
            ),
        },
    ]


def _waves_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    return [
        {
            "name": "dispersion_consistency",
            "description": (
                "Verify that the dispersion relation omega(k) is consistent: "
                "group velocity v_g = d(omega)/dk and phase velocity v_p = omega/k "
                "must be physically reasonable."
            ),
            "python_code": (
                "from sympy import symbols, diff\n"
                "k = symbols('k')\n"
                "# v_group = diff(omega_expr, k)\n"
                "# v_phase = omega_expr / k\n"
                "print('Dispersion: manually verify')"
            ),
        },
        {
            "name": "energy_flux_conservation",
            "description": (
                "Verify that the energy flux (intensity) is conserved or properly "
                "accounts for absorption/reflection."
            ),
            "python_code": (
                "# I_incident = I_reflected + I_transmitted + I_absorbed\n"
                "print('Energy flux: manually verify')"
            ),
        },
    ]


def _relativity_checks(knowns: list[str], unknowns: list[str]) -> list[_CHECK]:
    return [
        {
            "name": "lorentz_invariance",
            "description": (
                "Verify that scalar quantities are Lorentz invariant and "
                "four-vectors transform correctly."
            ),
            "python_code": (
                "# Check that s^2 = -(ct)^2 + x^2 + y^2 + z^2 is invariant.\n"
                "print('Lorentz invariance: manually verify')"
            ),
        },
        {
            "name": "non_relativistic_limit",
            "description": (
                "In the limit v/c -> 0 (or c -> infinity), the result must "
                "reduce to the Newtonian/classical answer."
            ),
            "python_code": (
                "from sympy import symbols, limit, oo\n"
                "c = symbols('c', positive=True)\n"
                "# classical = limit(relativistic_result, c, oo)\n"
                "print('Non-relativistic limit: manually verify')"
            ),
        },
    ]


_DOMAIN_REGISTRY: dict[str, type[None] | None] = None  # type: ignore[assignment]


def _get_domain_checks(
    domain: str, knowns: list[str], unknowns: list[str]
) -> list[_CHECK]:
    """Return domain-specific checks for the given domain."""
    registry: dict[str, object] = {
        "em": _em_checks,
        "mechanics": _mechanics_checks,
        "quantum": _quantum_checks,
        "thermo": _thermo_checks,
        "waves": _waves_checks,
        "relativity": _relativity_checks,
    }
    factory = registry.get(domain)
    if factory and callable(factory):
        return factory(knowns, unknowns)
    return []


_DOMAIN_NOTES: dict[str, str] = {
    "em": (
        "Common E&M errors: (1) Forgetting the electric field needed to satisfy "
        "J·n=0 at conductor boundaries. (2) Using J=sigma(v×B) without checking "
        "if div(J)=0 and boundary conditions are satisfied. (3) Ignoring the "
        "magnetic Reynolds number regime."
    ),
    "mechanics": (
        "Common mechanics errors: (1) Forgetting constraint forces. "
        "(2) Wrong sign conventions for torque/angular momentum. "
        "(3) Not checking limiting cases."
    ),
    "quantum": (
        "Common QM errors: (1) Unnormalized wavefunctions. "
        "(2) Non-Hermitian operators. (3) Missing boundary conditions "
        "for bound states. (4) Not checking correspondence principle."
    ),
    "thermo": (
        "Common thermo errors: (1) Sign errors in work/heat. "
        "(2) Confusing reversible and irreversible processes. "
        "(3) Not checking third-law behavior at T→0."
    ),
    "waves": (
        "Common wave errors: (1) Confusing group and phase velocity. "
        "(2) Wrong dispersion relation. (3) Not checking energy conservation."
    ),
    "relativity": (
        "Common relativity errors: (1) Mixing proper and coordinate quantities. "
        "(2) Not checking non-relativistic limit. (3) Incorrect Lorentz boosts."
    ),
}


def physics_checklist(
    domain: str, knowns: list[str], unknowns: list[str]
) -> dict[str, object]:
    """Generate domain-specific verification checks for a physics problem.

    Args:
        domain: Physics domain from ProblemSpec (em, mechanics, quantum,
            thermo, waves, relativity, other).
        knowns: List of known quantity names.
        unknowns: List of unknown quantity names to solve for.

    Returns:
        dict with keys:
        - checks: list of {name, description, python_code} verification items.
        - domain_notes: domain-specific gotchas and common errors.
    """
    domain_checks = _get_domain_checks(domain, knowns, unknowns)
    generic = _generic_checks(knowns, unknowns)
    all_checks = domain_checks + generic

    notes = _DOMAIN_NOTES.get(domain, "No domain-specific notes. Apply generic checks.")

    return {
        "checks": all_checks,
        "domain_notes": notes,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_physics_checklist.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: 141+ passed (132 existing + 9 new)

- [ ] **Step 6: Commit**

```bash
git add src/solvay/tools/physics_checklist.py tests/test_physics_checklist.py
git commit -m "feat: add physics_checklist tool for domain-specific verification

Generates verification checks per physics domain (em, mechanics, qm,
thermo, waves, relativity) with executable Python code for each check.
Used by the peer reviewer to catch analytical errors like missing
boundary conditions."
```

---

### Task 5: Register physics_checklist in peer reviewer

**Files:**
- Modify: `src/solvay/subagents/peer_reviewer.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_peer_reviewer_has_physics_checklist_tool() -> None:
    """The peer reviewer subagent must include physics_checklist in its tools."""
    from solvay.config import SolvayConfig
    from solvay.subagents.peer_reviewer import create_peer_reviewer_subagent

    stub = lambda **kw: {"results": [], "error": "stub"}  # noqa: E731
    config = SolvayConfig(default_model="fake-model")
    spec = create_peer_reviewer_subagent(config, web_search_tool=stub)

    tool_names = []
    for t in spec["tools"]:
        name = getattr(t, "__name__", None) or getattr(t, "name", str(t))
        tool_names.append(name)

    assert "physics_checklist" in tool_names, (
        f"Expected 'physics_checklist' in peer_reviewer tools, got: {tool_names}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_peer_reviewer_has_physics_checklist_tool -v`
Expected: FAIL — `physics_checklist` not in tools

- [ ] **Step 3: Implement the change**

Edit `src/solvay/subagents/peer_reviewer.py`:

```python
"""Peer reviewer subagent: semantic critique of solution drafts."""

from __future__ import annotations

from typing import Any

from solvay.config import SolvayConfig
from solvay.schemas import Verdict
from solvay.subagents import load_prompt
from solvay.tools.physics_checklist import physics_checklist
from solvay.tools.python_exec import python_exec


def create_peer_reviewer_subagent(
    config: SolvayConfig,
    web_search_tool: Any,
) -> dict[str, Any]:
    """Create the peer reviewer subagent definition (dict subagent)."""
    return {
        "name": "peer_reviewer",
        "description": (
            "Semantic critique: assess whether the approach is correct, complete, "
            "and well-justified. Check for missing terms and unjustified assumptions."
        ),
        "system_prompt": load_prompt("peer_reviewer"),
        "model": config.model_for("peer_reviewer"),
        "tools": [python_exec, web_search_tool, physics_checklist],
        "response_format": Verdict,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py::test_peer_reviewer_has_physics_checklist_tool -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 6: Commit**

```bash
git add src/solvay/subagents/peer_reviewer.py tests/test_agent.py
git commit -m "feat: register physics_checklist tool in peer reviewer

The peer reviewer can now call physics_checklist(domain, knowns,
unknowns) to get domain-specific verification checks to execute."
```

---

## Stream B: Claude Code Integration

### Task 6: Create solve-physics skill

**Files:**
- Create: `.claude/skills/solve-physics/SKILL.md`

- [ ] **Step 1: Create the skill directory and file**

Create `.claude/skills/solve-physics/SKILL.md`:

```markdown
---
description: Solve a physics problem using Solvay AI multi-agent solver. Use when the user asks to solve, derive, or analyze a physics problem.
---

The user wants to solve a physics problem using Solvay. Follow these steps:

1. **Identify the problem statement** from the user's message. If unclear, ask them to clarify.

2. **Choose the model** based on context:
   - Default (local): `ollama:qwen3.6:35b-a3b-coding-mxfp8`
   - If user requests Claude: `anthropic:claude-sonnet-4-6`
   - If user requests Gemini: `google_genai:gemini-2.5-flash`
   - If user specifies a model, use that model string directly.

3. **Run Solvay** using Bash:
   ```
   uv run solvay solve --model "MODEL" "PROBLEM_STATEMENT"
   ```
   This may take 5-30 minutes depending on the model and problem complexity.
   Use a 600-second timeout.

4. **Read the report**: Solvay writes a `.qd` report file. Read it to get the full derivation.

5. **Present the solution**: Show the final answer and key derivation steps to the user.
   Note any caveats or issues flagged by the peer reviewer.

6. **Verify if needed**: If the user asks for verification, you can:
   - Check specific calculations using Python
   - Re-run with a different model for comparison
   - Point to specific steps in the .qd report
```

- [ ] **Step 2: Verify the skill is discoverable**

Run: `ls -la .claude/skills/solve-physics/SKILL.md`
Expected: File exists

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/solve-physics/SKILL.md
git commit -m "feat: add solve-physics Claude Code skill

Lets Claude Code users invoke Solvay via /solve-physics or
automatically when a physics problem is detected."
```

---

### Task 7: Add fastmcp dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml` to add `fastmcp` to the dependencies list. Add it after `tavily-python`:

```
    "tavily-python>=0.5",
    "fastmcp>=2.0",
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: fastmcp installed successfully

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from fastmcp import FastMCP; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add fastmcp for MCP server support"
```

---

### Task 8: Create MCP server

**Files:**
- Create: `src/solvay/mcp_server.py`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_server.py`:

```python
"""Tests for the Solvay MCP server tool handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSolvePhysicsTool:
    def test_solve_returns_structured_result(self, tmp_path: Path) -> None:
        from solvay.mcp_server import _solve_physics_impl

        mock_collector = MagicMock()
        mock_collector.final_answer = "v = 14 m/s"
        mock_collector.total_s = 42.0
        mock_collector.subagent_runs = []

        with (
            patch("solvay.mcp_server.create_solvay_agent") as mock_create,
            patch("solvay.mcp_server.parse_stream") as mock_stream,
            patch("solvay.mcp_server.RunCollector", return_value=mock_collector),
            patch("solvay.mcp_server.generate_quarkdown", return_value="# Report"),
        ):
            mock_stream.return_value = iter([])
            result = _solve_physics_impl(
                problem="A ball is thrown upward at 14 m/s",
                model=None,
                output_dir=str(tmp_path),
            )

        assert result["answer"] == "v = 14 m/s"
        assert result["duration_s"] == 42.0
        assert "report_path" in result

    def test_solve_uses_custom_model(self, tmp_path: Path) -> None:
        from solvay.mcp_server import _solve_physics_impl

        mock_collector = MagicMock()
        mock_collector.final_answer = "42"
        mock_collector.total_s = 1.0
        mock_collector.subagent_runs = []

        with (
            patch("solvay.mcp_server.create_solvay_agent") as mock_create,
            patch("solvay.mcp_server.parse_stream", return_value=iter([])),
            patch("solvay.mcp_server.RunCollector", return_value=mock_collector),
            patch("solvay.mcp_server.generate_quarkdown", return_value=""),
        ):
            _solve_physics_impl(
                problem="test",
                model="anthropic:claude-sonnet-4-6",
                output_dir=str(tmp_path),
            )
            config_arg = mock_create.call_args[0][0]
            assert config_arg.default_model == "anthropic:claude-sonnet-4-6"


class TestListModelsTool:
    def test_list_models_returns_structure(self) -> None:
        from solvay.mcp_server import _list_models_impl

        with patch("solvay.mcp_server._check_ollama_models", return_value=[]):
            result = _list_models_impl()

        assert "models" in result
        assert isinstance(result["models"], list)

    def test_list_models_detects_api_keys(self) -> None:
        from solvay.mcp_server import _list_models_impl

        with (
            patch("solvay.mcp_server._check_ollama_models", return_value=[]),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}),
        ):
            result = _list_models_impl()

        providers = [m["provider"] for m in result["models"]]
        assert "anthropic" in providers


class TestGetReportTool:
    def test_get_report_reads_file(self, tmp_path: Path) -> None:
        from solvay.mcp_server import _get_report_impl

        report_file = tmp_path / "test.qd"
        report_file.write_text("# Test Report\nContent here")

        result = _get_report_impl(path=str(report_file))
        assert result["content"] == "# Test Report\nContent here"

    def test_get_report_missing_file(self) -> None:
        from solvay.mcp_server import _get_report_impl

        result = _get_report_impl(path="/nonexistent/path.qd")
        assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the MCP server**

Create `src/solvay/mcp_server.py`:

```python
"""Solvay MCP server: expose physics solving via Model Context Protocol."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from solvay.agent import create_solvay_agent
from solvay.config import SolvayConfig
from solvay.report import generate_quarkdown
from solvay.streaming import RunCollector, parse_stream

DEFAULT_MODEL = "ollama:qwen3.6:35b-a3b-coding-mxfp8"

mcp = FastMCP(
    name="solvay",
    instructions=(
        "Solvay is a multi-agent physics problem solver. Use solve_physics "
        "to solve derivations, calculations, and analytical problems. Use "
        "list_models to see available models. Use get_report to read a "
        "previously generated solution report."
    ),
)


def _check_ollama_models() -> list[dict[str, str]]:
    """Check available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                models.append({"name": f"ollama:{parts[0]}", "provider": "ollama"})
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _solve_physics_impl(
    problem: str,
    model: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Core implementation for solve_physics (testable without MCP decorator)."""
    effective_model = model or os.environ.get("SOLVAY_MODEL") or DEFAULT_MODEL

    model_kwargs: dict[str, Any] = {}
    if effective_model.startswith("ollama:"):
        model_kwargs = {"num_predict": 16384}

    config = SolvayConfig(default_model=effective_model, model_kwargs=model_kwargs)
    agent = create_solvay_agent(config)

    collector = RunCollector(problem=problem, model=effective_model)
    for event in parse_stream(agent, problem):
        collector.accumulate(event)

    qd = generate_quarkdown(collector)
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    report_path = out_dir / f"solvay-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.qd"
    report_path.write_text(qd, encoding="utf-8")

    subagent_names = [run.name for run in collector.subagent_runs]

    return {
        "answer": collector.final_answer,
        "report_path": str(report_path),
        "duration_s": collector.total_s,
        "subagents_called": subagent_names,
        "model": effective_model,
    }


def _list_models_impl() -> dict[str, Any]:
    """Core implementation for list_models."""
    models: list[dict[str, str]] = []

    ollama_models = _check_ollama_models()
    models.extend(ollama_models)

    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append({"name": "anthropic:claude-sonnet-4-6", "provider": "anthropic"})
        models.append({"name": "anthropic:claude-haiku-4-5", "provider": "anthropic"})

    if os.environ.get("GOOGLE_API_KEY"):
        models.append({"name": "google_genai:gemini-2.5-flash", "provider": "google"})

    if os.environ.get("OPENAI_API_KEY"):
        models.append({"name": "openai:gpt-4o", "provider": "openai"})

    return {"models": models}


def _get_report_impl(path: str) -> dict[str, Any]:
    """Core implementation for get_report."""
    report_path = Path(path)
    if not report_path.exists():
        return {"error": f"Report not found: {path}"}

    content = report_path.read_text(encoding="utf-8")
    return {"content": content}


@mcp.tool
def solve_physics(problem: str, model: str | None = None) -> str:
    """Solve a physics problem using Solvay's multi-agent system.

    Args:
        problem: The physics problem statement to solve.
        model: Optional model override. Examples: 'ollama:qwen3.5',
            'anthropic:claude-sonnet-4-6', 'google_genai:gemini-2.5-flash'.
            Defaults to SOLVAY_MODEL env var or local Qwen.

    Returns:
        JSON string with answer, report_path, duration_s, and subagents_called.
    """
    result = _solve_physics_impl(problem=problem, model=model)
    return json.dumps(result, indent=2)


@mcp.tool
def list_models() -> str:
    """List available models for Solvay.

    Returns:
        JSON string with a 'models' array of {name, provider} objects.
    """
    result = _list_models_impl()
    return json.dumps(result, indent=2)


@mcp.tool
def get_report(path: str) -> str:
    """Read a previously generated Solvay report (.qd file).

    Args:
        path: Path to the .qd report file.

    Returns:
        JSON string with the report content or an error message.
    """
    result = _get_report_impl(path=path)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All passing

- [ ] **Step 6: Commit**

```bash
git add src/solvay/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add MCP server for Claude Code integration

Exposes three tools via FastMCP stdio transport:
- solve_physics: run Solvay on a problem statement
- list_models: list available models (Ollama, API keys)
- get_report: read a previously generated .qd report"
```

---

### Task 9: Create MCP configuration and verify end-to-end

**Files:**
- Create: `.mcp.json`

- [ ] **Step 1: Create `.mcp.json`**

Create `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "solvay": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "solvay.mcp_server"]
    }
  }
}
```

- [ ] **Step 2: Verify the MCP server starts**

Run: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | uv run python -m solvay.mcp_server 2>/dev/null | head -1`
Expected: JSON response with server capabilities (or at least no crash)

- [ ] **Step 3: Run the full test suite one final time**

Run: `uv run pytest -x -q`
Expected: All tests passing (132 original + 9 checklist + 5 MCP = ~146)

- [ ] **Step 4: Commit**

```bash
git add .mcp.json
git commit -m "feat: add MCP configuration for Claude Code auto-discovery

Claude Code will detect and load the Solvay MCP server
automatically when working in this project directory."
```

---

## Final Verification

- [ ] **Run full test suite**: `uv run pytest -v` — all passing
- [ ] **Verify no regressions**: `uv run pytest tests/test_agent.py tests/test_solver_loop.py tests/test_streaming.py -v`
- [ ] **Verify MCP server starts cleanly**: `uv run python -m solvay.mcp_server` (Ctrl+C to exit)
- [ ] **Verify skill file exists**: `cat .claude/skills/solve-physics/SKILL.md`
