"""Smoke tests for the solvay-bench CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from solvay.benchmark.cli import app


def test_cli_app_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "solvay-bench" in result.stdout.lower() or "benchmark" in result.stdout.lower()
