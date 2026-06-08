# Solvay v0.7: Orchestrator Fixes + Claude Code Integration

**Date**: 2026-06-03
**Status**: Draft
**Scope**: Fix orchestrator routing bugs + integrate Solvay with Claude Code

---

## Problem Statement

Three orchestrator bugs were identified during a stress test (2026-06-02) of Qwen 35B on novel physics problems:

1. **Solver routing failure**: The `solver` subagent is never called. Deepagents v0.6.2 auto-injects a `general-purpose` subagent (graph.py:613-669) whose prominent examples in the task tool description bias the LLM toward calling it instead of `solver`.

2. **Peer reviewer misses analytical errors**: The peer reviewer uses `python_exec` for numerical checks but does not verify boundary conditions (E&M), dominant term extraction (asymptotics), or limiting cases. In the stress test, it failed to catch J dot n-hat != 0 at a conductor surface (Problem 1) and incorrect force scaling (Problem 3).

3. **Parser called twice**: In one run, the parser was invoked twice despite the anti-loop rule in the orchestrator prompt, wasting ~60 seconds.

Additionally, there is no way for Claude Code to invoke Solvay directly. Users must manually run `uv run solvay solve` via Bash.

## Goal 1: Fix Orchestrator Bugs

### 1.1 Disable auto-injected general-purpose subagent

**Root cause**: `create_deep_agent()` auto-adds a "general-purpose" subagent at position 0 of the subagents list (graph.py:669) unless a subagent with that name already exists.

**Fix**: In `agent.py`, add an explicit subagent named `"general-purpose"` that is functionally identical to the solver. This exploits the guard at graph.py:619:

```python
if not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
```

By providing our own `"general-purpose"`, deepagents skips auto-injection. If the orchestrator LLM somehow still calls `general-purpose`, it routes to the solver logic rather than a generic agent.

**File**: `src/solvay/agent.py`
**Change**: After creating the solver subagent (a `CompiledSubAgent`), create a clone that reuses the same compiled graph runnable:

```python
from deepagents import CompiledSubAgent

solver = create_solver_subagent(config, web_search, url_fetch)
gp_as_solver = CompiledSubAgent(
    name="general-purpose",
    description=solver["description"],
    runnable=solver["runnable"],
)
# Include gp_as_solver in the subagents list passed to create_deep_agent
```

This shares the compiled LangGraph, adding no memory or build overhead.

### 1.2 Orchestrator prompt hardening

**File**: `src/solvay/prompts/orchestrator.md`

Add/strengthen these rules:
- Explicit rejection: "Calling `general-purpose` is forbidden and will fail."
- Parser guard: "If you have already received a ProblemSpec JSON, do NOT call parser again under any circumstances."
- Step tracking: Add numbered checkboxes so the LLM can track which steps are complete.
- Timeout budget: "You have a maximum of 6 subagent calls total. Plan accordingly."

### 1.3 Enhanced peer reviewer prompt

**File**: `src/solvay/prompts/peer_reviewer.md`

Add domain-specific analytical verification instructions:

**Electromagnetism**:
- Compute J dot n-hat at all boundaries (must be zero for finite conductors)
- Check div J = 0 inside conductors
- Verify that the electric field satisfies boundary conditions

**Asymptotic scaling**:
- If the answer claims F ~ d^-n, differentiate the full expression and verify which term actually dominates at large d
- Check limiting cases: d -> 0, d -> infinity, parameters -> 0 or infinity

**All domains**:
- Dimensional analysis on every intermediate and final result
- Check known limiting cases (classical limit, non-relativistic limit, etc.)
- Verify energy/momentum conservation where applicable

### 1.4 Physics checklist tool

**New file**: `src/solvay/tools/physics_checklist.py`

A tool that generates domain-specific verification checks based on the ProblemSpec.

```python
def physics_checklist(domain: str, knowns: list[str], unknowns: list[str]) -> dict:
    """Generate domain-specific verification checks for a physics problem.

    Args:
        domain: Physics domain from ProblemSpec (em, mechanics, qm, thermo, etc.)
        knowns: List of known quantities.
        unknowns: List of unknown quantities to solve for.

    Returns:
        dict with keys:
        - checks: list of {name, description, python_code} verification items
        - domain_notes: domain-specific gotchas and common errors
    """
```

Domain check registry:

| Domain | Checks generated |
|--------|-----------------|
| `em` | Boundary conditions (J dot n = 0), div J = 0, Poynting conservation, gauge consistency |
| `mechanics` | Energy conservation, momentum conservation, limiting cases (m -> 0, m -> inf) |
| `qm` | Normalization, hermiticity of operators, correspondence principle, uncertainty relations |
| `thermo` | Second law (dS >= 0), limiting temperatures (T -> 0, T -> inf), equation of state consistency |
| `waves` | Dispersion relation consistency, group vs phase velocity, energy flux conservation |
| `relativity` | Lorentz invariance, proper vs coordinate quantities, non-relativistic limit |

Each check includes a `python_code` field with executable SymPy/NumPy code that the peer reviewer can run via `python_exec`.

**Register**: Add to peer_reviewer subagent's tools list in `subagents/peer_reviewer.py`.

**Tests**: `tests/test_physics_checklist.py` — verify that each domain produces valid checks, that python_code fields are syntactically valid, and that unknown domains return a generic fallback checklist.

## Goal 2: Claude Code Integration

### 2.1 Phase 1: Skill (immediate use)

**New file**: `.claude/skills/solve-physics/SKILL.md`

A Claude Code skill that invokes `uv run solvay solve` via Bash and presents the results. Triggered by `/solve-physics "problem"` or automatically when Claude Code detects a physics problem.

The skill:
1. Runs `uv run solvay solve --model "$MODEL" "$PROBLEM"`
2. Waits for completion
3. Reads the generated .qd report
4. Presents the solution in the conversation

Default model: `ollama:qwen3.6:35b-a3b-coding-mxfp8` (local Qwen).
Override: `--model anthropic:claude-sonnet-4-6`, `--model google_genai:gemini-2.5-flash`, etc.

### 2.2 Phase 2: MCP Server

**New file**: `src/solvay/mcp_server.py`

A stdio MCP server built with the `mcp` Python package (FastMCP) exposing three tools:

**`solve_physics(problem: str, model?: str) -> dict`**
- Creates `SolvayConfig`, builds agent via `create_solvay_agent()`
- Streams events via `parse_stream()` into `RunCollector`
- Returns `{answer: str, report_path: str, duration_s: float, subagents: list}`
- Model defaults: `model` param -> `$SOLVAY_MODEL` env var -> `ollama:qwen3.6:35b-a3b-coding-mxfp8`

**`list_models() -> dict`**
- Checks Ollama for local models (`ollama list`)
- Checks environment for API keys (ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.)
- Returns `{models: [{name, provider, available}]}`

**`get_report(path: str) -> dict`**
- Reads a .qd report file
- Returns `{content: str, metadata: dict}`
- For follow-up questions about previously solved problems

**Entry point**: `python -m solvay.mcp_server` (adds `__main__.py` entry)

**Configuration**: Add `.mcp.json` to project root:
```json
{
  "mcpServers": {
    "solvay": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "solvay.mcp_server"],
      "cwd": "."
    }
  }
}
```

**Dependency**: Add `mcp>=1.0` to `pyproject.toml` dependencies.

**Tests**: `tests/test_mcp_server.py` — unit tests for tool handlers using mocked agent.

### 2.3 Model routing

Model resolution chain (applies to both skill and MCP):

1. Explicit `model` parameter (CLI flag or MCP tool argument)
2. `$SOLVAY_MODEL` environment variable
3. Default: `ollama:qwen3.6:35b-a3b-coding-mxfp8`

Model strings use deepagents' provider prefix format:
- `ollama:model-name` — local Ollama
- `anthropic:claude-sonnet-4-6` — Anthropic API
- `google_genai:gemini-2.5-flash` — Google AI
- `vertexai:claude-sonnet-4-6` — Vertex AI
- `openai:gpt-4o` — OpenAI

## Files Changed

| File | Action | Goal |
|------|--------|------|
| `src/solvay/agent.py` | Modify | Add GP-as-solver clone (1.1) |
| `src/solvay/prompts/orchestrator.md` | Modify | Harden anti-loop, anti-GP rules (1.2) |
| `src/solvay/prompts/peer_reviewer.md` | Modify | Add analytical verification checklist (1.3) |
| `src/solvay/tools/physics_checklist.py` | Create | Domain-specific verification generator (1.4) |
| `src/solvay/subagents/peer_reviewer.py` | Modify | Register physics_checklist tool (1.4) |
| `tests/test_physics_checklist.py` | Create | Tests for checklist tool (1.4) |
| `.claude/skills/solve-physics/SKILL.md` | Create | Phase 1 skill (2.1) |
| `src/solvay/mcp_server.py` | Create | Phase 2 MCP server (2.2) |
| `tests/test_mcp_server.py` | Create | MCP server tests (2.2) |
| `pyproject.toml` | Modify | Add `mcp` dependency (2.2) |
| `.mcp.json` | Create | MCP server configuration (2.2) |

## Testing Strategy

- All existing 132 tests must continue passing
- New tests for `physics_checklist` tool (unit tests, one per domain)
- New tests for MCP server tool handlers (mocked agent, structured output)
- Integration test: run Solvay on a simple problem and verify solver (not general-purpose) is called
- Manual test: invoke `/solve-physics` from Claude Code and verify end-to-end flow

## Non-Goals

- No changes to the solver loop itself (solve-critique-judge cycle)
- No changes to the consolidator or report format
- No REST API (MCP stdio is sufficient)
- No UI changes to the CLI
- No changes to the benchmark system
