"""Tests for prompt-level native harness conventions."""

from __future__ import annotations

from solvay.subagents import load_prompt


def test_runtime_prompts_reference_native_workspace_notebook() -> None:
    for name in ["orchestrator", "parser", "researcher", "solver", "consolidator"]:
        prompt = load_prompt(name)
        assert "/workspace/lab_notebook.md" in prompt


def test_operational_roles_reference_harness_notes() -> None:
    for name in ["orchestrator", "researcher", "solver"]:
        prompt = load_prompt(name)
        assert "/memories/solvay/harness_notes.md" in prompt
        assert "Symptom:" in prompt
        assert "Suggested fix:" in prompt


def test_critic_prompts_do_not_reference_filesystem_paths() -> None:
    """Critics run without filesystem tools; their prompts must not order
    them to read or write files they cannot touch."""
    for name in ["verifier", "peer_reviewer"]:
        prompt = load_prompt(name)
        assert "/workspace/lab_notebook.md" not in prompt
        assert "/memories/solvay/harness_notes.md" not in prompt
