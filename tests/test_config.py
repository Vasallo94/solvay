"""Tests for SolvayConfig model resolution.

Precedence rules for ``model_for(role)``:

  1. Per-role entry in ``SolvayConfig.models``
  2. ``SolvayConfig.default_model`` (global override, e.g. passed via CLI)
  3. ``$SOLVAY_MODEL`` environment variable
  4. Hardcoded ``DEFAULT_MODEL`` fallback
"""

from __future__ import annotations

import pytest

from solvay.config import DEFAULT_MODEL, SolvayConfig


class TestModelResolution:
    def test_hardcoded_default_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOLVAY_MODEL", raising=False)
        config = SolvayConfig()
        assert config.model_for("orchestrator") == DEFAULT_MODEL

    def test_env_var_overrides_hardcoded_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOLVAY_MODEL", "ollama:qwen3.5")
        config = SolvayConfig()
        assert config.model_for("orchestrator") == "ollama:qwen3.5"
        assert config.model_for("solver") == "ollama:qwen3.5"

    def test_default_model_field_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOLVAY_MODEL", "ollama:qwen3.5")
        config = SolvayConfig(default_model="anthropic:claude-opus-4-1")
        assert config.model_for("orchestrator") == "anthropic:claude-opus-4-1"

    def test_per_role_entry_beats_default_model_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOLVAY_MODEL", "ollama:qwen3.5")
        config = SolvayConfig(
            default_model="anthropic:claude-opus-4-1",
            models={"solver": "google_genai:gemini-2.5-pro"},
        )
        assert config.model_for("solver") == "google_genai:gemini-2.5-pro"
        # Other roles still fall through to default_model, not env, not hardcoded.
        assert config.model_for("orchestrator") == "anthropic:claude-opus-4-1"
