"""Unit tests for solvay.mcp_server — all agent interactions are mocked."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from solvay.mcp_server import (
    _get_report_impl,
    _list_models_impl,
    _solve_physics_impl,
)
from solvay.streaming import RunCollector, RunFinished, SubagentFinished, SubagentStarted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_events(final_answer: str = "v = 42 m/s") -> list:
    return [
        SubagentStarted(name="parser", t=0.0),
        SubagentFinished(name="parser", duration_s=1.0),
        SubagentStarted(name="solver", t=1.0),
        SubagentFinished(name="solver", duration_s=2.0),
        RunFinished(total_s=3.0, final_answer=final_answer),
    ]


# ---------------------------------------------------------------------------
# Test: solve_physics returns structured result
# ---------------------------------------------------------------------------


class TestSolvePhysicsImpl:
    def test_solve_returns_structured_result(self, tmp_path: Path) -> None:
        """Verify that _solve_physics_impl returns expected keys and values."""
        with (
            patch("solvay.mcp_server.create_solvay_agent") as mock_create,
            patch("solvay.mcp_server.parse_stream") as mock_stream,
            patch("solvay.mcp_server.generate_quarkdown") as mock_qd,
        ):
            mock_create.return_value = MagicMock()
            mock_stream.return_value = iter(_mock_events("v = 42 m/s"))
            mock_qd.return_value = ".docname {Test}\n"

            result = _solve_physics_impl(
                problem="A ball is dropped from height 90 m. Find v at impact.",
                model=None,
                output_dir=tmp_path,
            )

        assert "answer" in result
        assert "report_path" in result
        assert "duration_s" in result
        assert "subagents_called" in result
        assert "model" in result

        assert result["answer"] == "v = 42 m/s"
        assert result["duration_s"] == pytest.approx(3.0)
        assert "parser" in result["subagents_called"]
        assert "solver" in result["subagents_called"]

        report_path = Path(result["report_path"])
        assert report_path.exists()
        assert report_path.suffix == ".qd"

    def test_solve_uses_custom_model(self, tmp_path: Path) -> None:
        """Verify that a custom model string is passed into SolvayConfig."""
        custom_model = "anthropic:claude-opus-4-5"

        with (
            patch("solvay.mcp_server.create_solvay_agent") as mock_create,
            patch("solvay.mcp_server.parse_stream") as mock_stream,
            patch("solvay.mcp_server.generate_quarkdown") as mock_qd,
        ):
            mock_create.return_value = MagicMock()
            mock_stream.return_value = iter(_mock_events())
            mock_qd.return_value = ""

            _solve_physics_impl(
                problem="Simple problem",
                model=custom_model,
                output_dir=tmp_path,
            )

        # SolvayConfig is the first positional arg to create_solvay_agent
        config_passed = mock_create.call_args[0][0]
        assert config_passed.default_model == custom_model

    def test_solve_applies_ollama_model_kwargs(self, tmp_path: Path) -> None:
        """Verify num_predict cap is applied for Ollama models."""
        ollama_model = "ollama:qwen3:7b"

        with (
            patch("solvay.mcp_server.create_solvay_agent") as mock_create,
            patch("solvay.mcp_server.parse_stream") as mock_stream,
            patch("solvay.mcp_server.generate_quarkdown") as mock_qd,
        ):
            mock_create.return_value = MagicMock()
            mock_stream.return_value = iter(_mock_events())
            mock_qd.return_value = ""

            _solve_physics_impl(
                problem="Simple problem",
                model=ollama_model,
                output_dir=tmp_path,
            )

        config_passed = mock_create.call_args[0][0]
        assert config_passed.model_kwargs.get("num_predict") == 16384


# ---------------------------------------------------------------------------
# Test: list_models
# ---------------------------------------------------------------------------


class TestListModelsImpl:
    def test_list_models_returns_structure(self) -> None:
        """Verify that _list_models_impl returns a dict with a models list."""
        with patch("solvay.mcp_server._check_ollama_models") as mock_ollama:
            mock_ollama.return_value = [
                {"name": "ollama:qwen3:7b", "provider": "ollama"},
            ]

            result = _list_models_impl()

        assert "models" in result
        assert isinstance(result["models"], list)
        assert any(m["provider"] == "ollama" for m in result["models"])

    def test_list_models_detects_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that provider entries are added when API keys are present."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("solvay.mcp_server._check_ollama_models") as mock_ollama:
            mock_ollama.return_value = []
            result = _list_models_impl()

        providers = [m["provider"] for m in result["models"]]
        assert "anthropic" in providers
        assert "google" not in providers
        assert "openai" not in providers


# ---------------------------------------------------------------------------
# Test: get_report
# ---------------------------------------------------------------------------


class TestGetReportImpl:
    def test_get_report_reads_file(self, tmp_path: Path) -> None:
        """Verify that an existing .qd file is returned as content."""
        report = tmp_path / "test-report.qd"
        report.write_text(".docname {Hello}\n\n# Title\n", encoding="utf-8")

        result = _get_report_impl(str(report))

        assert "content" in result
        assert ".docname {Hello}" in result["content"]

    def test_get_report_missing_file(self, tmp_path: Path) -> None:
        """Verify that a missing file returns an error dict."""
        nonexistent = tmp_path / "does-not-exist.qd"

        result = _get_report_impl(str(nonexistent))

        assert "error" in result
        assert "content" not in result
