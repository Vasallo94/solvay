"""Native DeepAgents harness helpers: memory paths, backends, permissions, notes."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Literal

from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from solvay.config import HarnessConfig

HarnessNoteSeverity = Literal["info", "warning", "error"]

WORKSPACE_ROOT = "/workspace"
SESSION_NOTEBOOK_PATH = f"{WORKSPACE_ROOT}/lab_notebook.md"
LONG_TERM_MEMORY_PATH = "/memories/solvay/AGENTS.md"
HARNESS_NOTES_PATH = "/memories/solvay/harness_notes.md"
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


def memory_paths(config: HarnessConfig) -> list[str]:
    """Return native DeepAgents memory files for the Solvay runtime."""
    if not config.native_memory_enabled:
        return []
    paths = [LONG_TERM_MEMORY_PATH]
    if config.persist_harness_notes:
        paths.append(HARNESS_NOTES_PATH)
    return paths


def build_backend() -> CompositeBackend:
    """Build the default native DeepAgents backend for Solvay."""
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": FilesystemBackend(
                root_dir=".solvay/memories",
                virtual_mode=True,
            )
        },
    )


def build_permissions() -> list[FilesystemPermission]:
    """Return conservative filesystem permission rules for runtime agents."""
    return [
        FilesystemPermission(operations=["read", "write"], paths=["/workspace/**"]),
        FilesystemPermission(operations=["read"], paths=["/memories/**"]),
        FilesystemPermission(operations=["write"], paths=["/src/**", "/tests/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/docs/**"], mode="deny"),
    ]


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
