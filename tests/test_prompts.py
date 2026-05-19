"""Tests for prompt-level native harness conventions."""

from __future__ import annotations

from solvay.subagents import load_prompt


def test_runtime_prompts_reference_native_workspace_notebook() -> None:
    for name in [
        "orchestrator",
        "parser",
        "researcher",
        "solver",
        "verifier",
        "peer_reviewer",
        "consolidator",
    ]:
        prompt = load_prompt(name)
        assert "/workspace/lab_notebook.md" in prompt


def test_operational_roles_reference_harness_notes() -> None:
    for name in ["orchestrator", "researcher", "solver", "verifier", "peer_reviewer"]:
        prompt = load_prompt(name)
        assert "/memories/solvay/harness_notes.md" in prompt
        assert "Symptom:" in prompt
        assert "Suggested fix:" in prompt
