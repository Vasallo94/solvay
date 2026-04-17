"""Cross-run persistence: ~/.solvay/ journal + session I/O."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from solvay.config import PersistenceConfig
from solvay.schemas import JournalEntry


def _solvay_dir(config: PersistenceConfig) -> Path:
    """Resolve and create the solvay data directory."""
    path = Path(config.solvay_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sessions_dir(config: PersistenceConfig) -> Path:
    """Resolve and create the sessions subdirectory."""
    path = _solvay_dir(config) / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_journal_snapshot(
    config: PersistenceConfig,
    max_entries: int = 10,
) -> str:
    """Load the most recent entries from lab_journal.md.

    Args:
        config: Persistence configuration.
        max_entries: Maximum entries to include in the snapshot.

    Returns:
        A string with recent journal entries, or empty string if no journal.
    """
    journal_path = _solvay_dir(config) / "lab_journal.md"
    if not journal_path.exists():
        return ""

    content = journal_path.read_text(encoding="utf-8")
    entries = [e for e in content.split("\n---\n") if e.strip()]
    recent = entries[-max_entries:]
    return "\n---\n".join(recent)


def append_journal_entries(
    config: PersistenceConfig,
    entries: list[JournalEntry],
) -> None:
    """Append consolidated entries to lab_journal.md.

    Args:
        config: Persistence configuration.
        entries: List of JournalEntry objects to append.
    """
    if not entries:
        return

    journal_path = _solvay_dir(config) / "lab_journal.md"
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    for entry in entries:
        lines.append(
            f"\n---\n## [{entry.role}] iter {entry.iteration} ({timestamp})\n"
            f"{entry.content}\n"
        )

    with open(journal_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def save_session_notebook(
    config: PersistenceConfig,
    problem_slug: str,
    notebook_content: str,
) -> Path:
    """Save the full session notebook for auditability.

    Args:
        config: Persistence configuration.
        problem_slug: A slug derived from the problem statement.
        notebook_content: Full lab_notebook.md content from the session.

    Returns:
        Path to the saved session file.
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    slug = _slugify(problem_slug)
    filename = f"{timestamp}_{slug}.md"
    path = _sessions_dir(config) / filename
    path.write_text(notebook_content, encoding="utf-8")
    return path


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len]
