"""Tests for native DeepAgents harness configuration."""

from __future__ import annotations

import pytest
from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from solvay.config import HarnessConfig, SolvayConfig
from solvay.harness import (
    HARNESS_NOTES_PATH,
    LONG_TERM_MEMORY_PATH,
    SESSION_NOTEBOOK_PATH,
    WORKSPACE_ROOT,
    HarnessNote,
    build_backend,
    build_permissions,
    format_harness_note,
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


def test_build_backend_routes_memories_to_filesystem_backend() -> None:
    backend = build_backend()
    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.default, StateBackend)
    assert "/memories/" in backend.routes
    assert isinstance(backend.routes["/memories/"], FilesystemBackend)


def test_build_permissions_allow_workspace_and_deny_source_writes() -> None:
    permissions = build_permissions()
    assert all(isinstance(rule, FilesystemPermission) for rule in permissions)
    assert any(rule.mode == "allow" and "/workspace/**" in rule.paths for rule in permissions)
    assert any(rule.mode == "deny" and "/src/**" in rule.paths for rule in permissions)


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
