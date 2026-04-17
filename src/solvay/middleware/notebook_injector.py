"""Middleware that injects the last N lab notebook entries into subagent context."""

from __future__ import annotations

from solvay.config import NotebookConfig


def get_recent_notebook_entries(
    notebook_content: str,
    max_entries: int = 20,
) -> str:
    """Extract the last N entries from the lab notebook.

    Entries are delimited by '## [' headers. Returns the last max_entries
    entries as a single string.

    Args:
        notebook_content: Full contents of lab_notebook.md.
        max_entries: Maximum number of entries to return.

    Returns:
        A string containing the last N entries.
    """
    if not notebook_content.strip():
        return ""

    entries: list[str] = []
    current_entry: list[str] = []

    for line in notebook_content.split("\n"):
        if line.startswith("## [") and current_entry:
            entries.append("\n".join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)

    if current_entry:
        entries.append("\n".join(current_entry))

    recent = entries[-max_entries:]
    return "\n\n".join(recent)


def build_notebook_context(
    notebook_content: str,
    config: NotebookConfig | None = None,
) -> str:
    """Build the notebook context string to inject into subagent prompts.

    Args:
        notebook_content: Full contents of lab_notebook.md.
        config: Notebook configuration (uses defaults if None).

    Returns:
        A formatted context string with recent notebook entries.
    """
    if config is None:
        config = NotebookConfig()

    recent = get_recent_notebook_entries(
        notebook_content, max_entries=config.max_injected_entries
    )

    if not recent:
        return (
            "\n\n---\n"
            "**Lab Notebook:** (empty -- you are the first agent this session)\n"
            "---\n"
        )

    return (
        "\n\n---\n"
        "**Recent Lab Notebook Entries** (read before acting; append after acting):\n\n"
        f"{recent}\n"
        "---\n"
    )
