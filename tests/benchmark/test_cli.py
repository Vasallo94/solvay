"""Smoke tests for the solvay-bench CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from solvay.benchmark.cli import app


def test_cli_app_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "solvay-bench" in result.stdout.lower() or "benchmark" in result.stdout.lower()


def test_cli_run_invokes_matrix(tmp_path: Path) -> None:
    from solvay.benchmark.profiles import PROFILES, Profile, ProfileResult, register

    def runner(problem, model, config):  # type: ignore[no-untyped-def]
        return ProfileResult(
            answer_raw="4.905",
            answer_extracted="4.905",
            elapsed_seconds=0.0,
            tokens={"input": 0, "output": 0, "cache_read": 0},
        )

    register(Profile(name="cli-fake", description="", runner=runner, cost_estimate="low"))
    out = tmp_path / "run.jsonl"
    runner_cli = CliRunner()
    result = runner_cli.invoke(
        app,
        [
            "run",
            "--profiles",
            "cli-fake",
            "--models",
            "anthropic:claude-sonnet-4-6",
            "--problems",
            "benchmark/problems",
            "--out",
            str(out),
        ],
    )
    PROFILES.pop("cli-fake", None)
    assert result.exit_code == 0, result.stdout
    assert out.exists()
