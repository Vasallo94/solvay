"""Tests for the Solvay config fingerprint."""

from __future__ import annotations

from pathlib import Path

import pytest

from solvay.benchmark.fingerprint import compute_fingerprint
from solvay.config import SolvayConfig


def test_fingerprint_is_deterministic() -> None:
    cfg = SolvayConfig()
    a = compute_fingerprint(cfg)
    b = compute_fingerprint(cfg)
    assert a == b
    assert a.startswith("sha256:")


def test_fingerprint_changes_with_model_override() -> None:
    base = SolvayConfig()
    other = SolvayConfig(default_model="openai:gpt-4o")
    assert compute_fingerprint(base) != compute_fingerprint(other)


def test_fingerprint_changes_with_prompt_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Copy the real prompts dir to tmp, tweak one file, point the resolver there.
    import shutil

    import solvay.subagents as subagents

    real_prompts = Path(subagents.__file__).parent.parent / "prompts"
    fake_prompts = tmp_path / "prompts"
    shutil.copytree(real_prompts, fake_prompts)

    cfg = SolvayConfig()
    monkeypatch.setattr("solvay.benchmark.fingerprint._PROMPTS_DIR", fake_prompts)
    before = compute_fingerprint(cfg)

    target = fake_prompts / "solver.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nMORE TEXT\n", encoding="utf-8")
    after = compute_fingerprint(cfg)

    assert before != after
