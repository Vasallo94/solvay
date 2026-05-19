# Native DeepAgents Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Solvay toward a native DeepAgents harness with filesystem-backed memory, backend routing, permissions, session workspace conventions, and operational harness notes.

**Architecture:** Add a small `solvay.harness` module that owns DeepAgents memory paths, backend construction, permissions, and harness-note formatting. Wire `create_solvay_agent()` to pass `memory=`, `backend=`, and `permissions=` into `create_deep_agent()`. Update runtime prompts to use `/workspace/lab_notebook.md` and `/memories/solvay/harness_notes.md` instead of relying on custom notebook injection as the primary channel.

**Tech Stack:** Python 3.14, DeepAgents, LangChain, LangGraph, Pydantic, Typer, pytest, ruff, mypy.

---

## File Structure

- Create `src/solvay/harness.py`
  - Defines constants for native DeepAgents paths.
  - Builds `CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(...)})`.
  - Builds filesystem permission rules.
  - Formats operational harness notes.
  - Provides a simple secret-pattern guard for note content.
- Modify `src/solvay/config.py`
  - Add `HarnessConfig` to configure native memory, backend mode, and harness note behavior.
  - Add it to `SolvayConfig`.
- Modify `src/solvay/agent.py`
  - Use harness helpers when calling `create_deep_agent()`.
- Modify prompts in `src/solvay/prompts/*.md`
  - Replace generic `lab_notebook.md` references with `/workspace/lab_notebook.md`.
  - Add operational harness note rules where relevant.
- Add `tests/test_harness.py`
  - Unit tests for paths, permissions, backend shape, note formatting, and secret blocking.
- Modify existing tests only where signatures/config shape change.

---

### Task 1: Harness Config and Native Path Constants

**Files:**
- Modify: `src/solvay/config.py`
- Create: `src/solvay/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write failing tests for path constants and default config**

Add this to `tests/test_harness.py`:

```python
"""Tests for native DeepAgents harness configuration."""

from __future__ import annotations

from solvay.config import HarnessConfig, SolvayConfig
from solvay.harness import (
    HARNESS_NOTES_PATH,
    LONG_TERM_MEMORY_PATH,
    SESSION_NOTEBOOK_PATH,
    WORKSPACE_ROOT,
    memory_paths,
)


def test_harness_paths_are_absolute_deepagents_paths() -> None:
    assert WORKSPACE_ROOT == "/workspace"
    assert SESSION_NOTEBOOK_PATH == "/workspace/lab_notebook.md"
    assert LONG_TERM_MEMORY_PATH == "/memories/solvay/AGENTS.md"
    assert HARNESS_NOTES_PATH == "/memories/solvay/harness_notes.md"


def test_default_config_enables_native_harness_memory() -> None:
    config = SolvayConfig()
    assert isinstance(config.harness, HarnessConfig)
    assert config.harness.native_memory_enabled is True
    assert memory_paths(config.harness) == [
        "/memories/solvay/AGENTS.md",
        "/memories/solvay/harness_notes.md",
    ]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py -q
```

Expected: import errors for `HarnessConfig` and `solvay.harness`.

- [ ] **Step 3: Add config and constants**

In `src/solvay/config.py`, add:

```python
@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for native DeepAgents harness features."""

    native_memory_enabled: bool = True
    memory_namespace: str = "solvay"
    persist_harness_notes: bool = True
```

Add this field to `SolvayConfig`:

```python
harness: HarnessConfig = field(default_factory=HarnessConfig)
```

Create `src/solvay/harness.py`:

```python
"""Native DeepAgents harness helpers: memory paths, backends, permissions, notes."""

from __future__ import annotations

from solvay.config import HarnessConfig

WORKSPACE_ROOT = "/workspace"
SESSION_NOTEBOOK_PATH = f"{WORKSPACE_ROOT}/lab_notebook.md"
LONG_TERM_MEMORY_PATH = "/memories/solvay/AGENTS.md"
HARNESS_NOTES_PATH = "/memories/solvay/harness_notes.md"


def memory_paths(config: HarnessConfig) -> list[str]:
    """Return native DeepAgents memory files for the Solvay runtime."""
    if not config.native_memory_enabled:
        return []
    paths = [LONG_TERM_MEMORY_PATH]
    if config.persist_harness_notes:
        paths.append(HARNESS_NOTES_PATH)
    return paths
```

- [ ] **Step 4: Run test**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py -q
```

Expected: pass.

---

### Task 2: Backend and Permission Builders

**Files:**
- Modify: `src/solvay/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write failing tests for backend and permissions**

Append to `tests/test_harness.py`:

```python
from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from solvay.harness import build_backend, build_permissions


def test_build_backend_routes_memories_to_store_backend() -> None:
    backend = build_backend()
    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.default, StateBackend)
    assert "/memories/" in backend.routes
    assert isinstance(backend.routes["/memories/"], StoreBackend)


def test_build_permissions_allow_workspace_and_deny_source_writes() -> None:
    permissions = build_permissions()
    assert all(isinstance(rule, FilesystemPermission) for rule in permissions)
    assert any(rule.mode == "allow" and "/workspace/**" in rule.paths for rule in permissions)
    assert any(rule.mode == "deny" and "/src/**" in rule.paths for rule in permissions)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py -q
```

Expected: import errors for `build_backend` and `build_permissions`.

- [ ] **Step 3: Implement backend and permissions**

Add to `src/solvay/harness.py`:

```python
from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend


def build_backend() -> CompositeBackend:
    """Build the default native DeepAgents backend for Solvay."""
    return CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend()},
    )


def build_permissions() -> list[FilesystemPermission]:
    """Return conservative filesystem permission rules for runtime agents."""
    return [
        FilesystemPermission(operations=["read", "write"], paths=["/workspace/**"]),
        FilesystemPermission(operations=["read"], paths=["/memories/**"]),
        FilesystemPermission(operations=["write"], paths=["/src/**", "/tests/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/docs/**"], mode="deny"),
    ]
```

- [ ] **Step 4: Run test**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py -q
```

Expected: pass.

---

### Task 3: Harness Note Formatter and Secret Guard

**Files:**
- Modify: `src/solvay/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write failing tests for note formatting**

Append to `tests/test_harness.py`:

```python
import pytest

from solvay.harness import HarnessNote, format_harness_note


def test_format_harness_note_uses_required_contract() -> None:
    note = HarnessNote(
        role="solver",
        severity="warning",
        symptom="python_exec rejected an import.",
        context="Verifier requested numeric integration.",
        likely_cause="The package is absent from the execution environment.",
        suggested_fix="Add the package or document that it is unsupported.",
        artifacts="/workspace/artifacts/run-1/",
        timestamp="2026-05-20T12:34:56Z",
    )
    text = format_harness_note(note)
    assert text.startswith("## [harness-note] 2026-05-20T12:34:56Z role=solver severity=warning")
    assert "Symptom: python_exec rejected an import." in text
    assert "Artifacts: /workspace/artifacts/run-1/" in text


def test_format_harness_note_rejects_secret_like_content() -> None:
    note = HarnessNote(
        role="researcher",
        severity="error",
        symptom="ANTHROPIC_API_KEY=sk-ant-secret leaked.",
        context="Testing secret guard.",
        likely_cause="Bad logging.",
        suggested_fix="Redact environment variables.",
    )
    with pytest.raises(ValueError, match="secret-like"):
        format_harness_note(note)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py -q
```

Expected: import errors for `HarnessNote` and `format_harness_note`.

- [ ] **Step 3: Implement formatter**

Add to `src/solvay/harness.py`:

```python
import datetime as dt
import re
from dataclasses import dataclass
from typing import Literal

HarnessNoteSeverity = Literal["info", "warning", "error"]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"[A-Z0-9_]*API_KEY\s*=", re.IGNORECASE),
    re.compile(r"(password|token|secret)\s*[:=]", re.IGNORECASE),
]


@dataclass(frozen=True)
class HarnessNote:
    """A structured operational note from a runtime agent to harness developers."""

    role: str
    severity: HarnessNoteSeverity
    symptom: str
    context: str
    likely_cause: str
    suggested_fix: str
    artifacts: str | None = None
    timestamp: str | None = None


def _contains_secret_like_content(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def format_harness_note(note: HarnessNote) -> str:
    """Format a harness note as markdown, rejecting secret-like content."""
    timestamp = note.timestamp or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    fields = [
        note.role,
        note.severity,
        note.symptom,
        note.context,
        note.likely_cause,
        note.suggested_fix,
        note.artifacts or "",
    ]
    joined = "\n".join(fields)
    if _contains_secret_like_content(joined):
        raise ValueError("Harness note contains secret-like content")
    return (
        f"## [harness-note] {timestamp} role={note.role} severity={note.severity}\n"
        f"Symptom: {note.symptom}\n"
        f"Context: {note.context}\n"
        f"Likely cause: {note.likely_cause}\n"
        f"Suggested fix: {note.suggested_fix}\n"
        f"Artifacts: {note.artifacts or 'none'}\n"
    )
```

- [ ] **Step 4: Run test**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py -q
```

Expected: pass.

---

### Task 4: Wire Native Harness into `create_solvay_agent`

**Files:**
- Modify: `src/solvay/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write failing test that `create_deep_agent` receives native harness args**

Add to `tests/test_agent.py`:

```python
from unittest.mock import patch

from deepagents.backends import CompositeBackend

from solvay.agent import create_solvay_agent
from solvay.harness import HARNESS_NOTES_PATH, LONG_TERM_MEMORY_PATH


def test_create_agent_wires_native_memory_backend_and_permissions() -> None:
    with patch("solvay.agent.create_deep_agent") as fake_create:
        fake_create.return_value = object()
        create_solvay_agent()

    kwargs = fake_create.call_args.kwargs
    assert kwargs["memory"] == [LONG_TERM_MEMORY_PATH, HARNESS_NOTES_PATH]
    assert isinstance(kwargs["backend"], CompositeBackend)
    assert kwargs["permissions"]
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_agent.py::test_create_agent_wires_native_memory_backend_and_permissions -q
```

Expected: failure because `memory`, `backend`, and `permissions` are not passed.

- [ ] **Step 3: Wire harness helpers**

In `src/solvay/agent.py`, import:

```python
from solvay.harness import build_backend, build_permissions, memory_paths
```

Pass these arguments to `create_deep_agent()`:

```python
memory=memory_paths(config.harness),
backend=build_backend(),
permissions=build_permissions(),
```

- [ ] **Step 4: Run tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_agent.py tests/test_solver_loop.py -q
```

Expected: pass.

---

### Task 5: Prompt Migration to Native Paths and Harness Notes

**Files:**
- Modify: `src/solvay/prompts/orchestrator.md`
- Modify: `src/solvay/prompts/parser.md`
- Modify: `src/solvay/prompts/researcher.md`
- Modify: `src/solvay/prompts/solver.md`
- Modify: `src/solvay/prompts/verifier.md`
- Modify: `src/solvay/prompts/peer_reviewer.md`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write failing prompt tests**

Create `tests/test_prompts.py`:

```python
"""Tests for prompt-level native harness conventions."""

from __future__ import annotations

from solvay.subagents import load_prompt


def test_runtime_prompts_reference_native_workspace_notebook() -> None:
    for name in ["orchestrator", "parser", "researcher", "solver", "verifier", "peer_reviewer"]:
        prompt = load_prompt(name)
        assert "/workspace/lab_notebook.md" in prompt


def test_operational_roles_reference_harness_notes() -> None:
    for name in ["orchestrator", "researcher", "solver", "verifier", "peer_reviewer"]:
        prompt = load_prompt(name)
        assert "/memories/solvay/harness_notes.md" in prompt
        assert "Symptom:" in prompt
        assert "Suggested fix:" in prompt
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_prompts.py -q
```

Expected: failures because prompts still use older notebook references and lack harness notes.

- [ ] **Step 3: Update prompts**

For each runtime prompt:

- Replace `lab_notebook.md` with `/workspace/lab_notebook.md`.
- Add this short rule block to operational roles:

```md
## Operational harness notes

If a tool fails, a path is confusing, dependencies are missing, structured
output repeatedly fails, or the harness behavior blocks the task, append a short
note to `/memories/solvay/harness_notes.md` using:

Symptom: ...
Context: ...
Likely cause: ...
Suggested fix: ...
Artifacts: ...

Never include secrets, raw API keys, passwords, or private credentials.
```

- [ ] **Step 4: Run prompt tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_prompts.py -q
```

Expected: pass.

---

### Task 6: Full Verification

**Files:**
- Existing files touched by Tasks 1-5

- [ ] **Step 1: Run focused harness tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run pytest tests/test_harness.py tests/test_agent.py tests/test_prompts.py tests/test_solver_loop.py -q
```

Expected: pass.

- [ ] **Step 2: Run quality gates**

Run:

```bash
env UV_CACHE_DIR=/tmp/uvcache uv run ruff check .
env UV_CACHE_DIR=/tmp/uvcache uv run mypy src tests
env UV_CACHE_DIR=/tmp/uvcache uv run pytest
git diff --check
```

Expected:

- ruff: `All checks passed!`
- mypy: `Success: no issues found`
- pytest: all non-live tests pass, with live smoke still skipped unless explicitly enabled.
- diff check: no output.

---

## Self-Review

Spec coverage:

- Native DeepAgents memory paths: Tasks 1 and 4.
- Backend routing: Task 2.
- Permissions: Task 2.
- Harness notes: Task 3 and Task 5.
- Prompt/session workspace migration: Task 5.
- Supervisor as native DeepAgent: Task 4 and Task 5.
- Solver deterministic loop preservation: Task 4 verification includes solver loop tests.
- Ollama readiness: preserved as follow-up after this plan; this plan builds the harness foundation needed before the live run.

Known scope boundary:

- This plan does not replace `python_exec` with a DeepAgents sandbox. It intentionally keeps `python_exec` until live testing proves the native sandbox path can cover scientific workloads.
- This plan does not remove `persistence.py` yet. It stops adding new dependency on it and prepares a later removal after native memory is proven.

