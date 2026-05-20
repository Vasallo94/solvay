"""Tests for the Solvay CLI module (dotenv auto-loading, etc.)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from solvay.cli import app


class TestDotenvAutoLoad:
    """Importing ``solvay.cli`` must trigger ``load_dotenv()`` so that
    LangSmith / Anthropic / Tavily credentials placed in a project-root
    ``.env`` flow into the process before langchain libraries read them.

    The test spawns a subprocess because ``load_dotenv()`` runs at import
    time and is not repeatable within a single test session.
    """

    def test_cli_import_loads_dotenv_from_cwd(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SOLVAY_TEST_MARKER=from-dotenv\n")
        script = (
            "import os\n"
            "import solvay.cli  # noqa: F401 -- import triggers load_dotenv()\n"
            "print(os.environ.get('SOLVAY_TEST_MARKER', 'missing'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "from-dotenv"

    def test_shell_env_overrides_dotenv(self, tmp_path: Path) -> None:
        """Explicit shell exports must win over ``.env`` (override=False)."""
        (tmp_path / ".env").write_text("SOLVAY_TEST_MARKER=from-dotenv\n")
        script = (
            "import os\nimport solvay.cli  # noqa: F401\nprint(os.environ['SOLVAY_TEST_MARKER'])\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env={"SOLVAY_TEST_MARKER": "from-shell", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "from-shell"


def test_notebook_content_prefers_native_workspace_path() -> None:
    from solvay.cli import _notebook_content

    files = {
        "/lab_notebook.md": {"content": "legacy"},
        "/workspace/lab_notebook.md": {"content": "native"},
    }
    assert _notebook_content(files) == "native"


def test_bench_command_uses_new_matrix_runner(tmp_path: Path) -> None:
    from solvay.benchmark.config import BenchConfig
    from solvay.benchmark.profiles import PROFILES, Profile, ProfileResult, register
    from solvay.benchmark.schema import Problem

    def runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
        return ProfileResult(
            answer_raw="2",
            answer_extracted="2",
            elapsed_seconds=0.0,
            tokens={"input": 0, "output": 0, "cache_read": 0},
        )

    register(Profile(name="cli-main-fake", description="", runner=runner, cost_estimate="low"))
    out = tmp_path / "main-bench.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--profiles",
            "cli-main-fake",
            "--models",
            "ollama:qwen3.5",
            "--problems",
            "benchmark/problems",
            "--out",
            str(out),
        ],
    )
    PROFILES.pop("cli-main-fake", None)

    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert "Wrote" in result.stdout


def test_bench_help_mentions_ollama_model_strings() -> None:
    result = CliRunner().invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "ollama:" in result.stdout


import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from solvay.cli import app
from solvay.streaming import RunFinished, SubagentFinished, SubagentStarted


def _mock_events():
    return iter([
        SubagentStarted(name="parser", t=0.0),
        SubagentFinished(name="parser", duration_s=5.0),
        SubagentStarted(name="consolidator", t=5.0),
        SubagentFinished(name="consolidator", duration_s=3.0),
        RunFinished(total_s=8.0, final_answer="v = 14.0 m/s"),
    ])


class TestSolveReportGeneration:
    def test_report_always_written_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: _mock_events())
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: ".docname {Test}\n")

        result = CliRunner().invoke(app, ["solve", "test problem"])

        assert result.exit_code == 0, result.output
        qd_files = list(tmp_path.glob("solvay-report-*.qd"))
        assert len(qd_files) == 1
        assert ".docname" in qd_files[0].read_text()

    def test_report_written_to_custom_output_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: _mock_events())
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: ".docname {Test}\n")

        out = tmp_path / "my-report.qd"
        result = CliRunner().invoke(app, ["solve", "test problem", "--output", str(out)])

        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_verbose_flag_shows_tool_call_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from solvay.streaming import ToolCallMade, ToolResultReceived

        events = iter([
            SubagentStarted(name="parser", t=0.0),
            ToolCallMade(subagent="parser", tool="python_exec", args_preview="h=10"),
            ToolResultReceived(subagent="parser", tool="python_exec", result_preview="h=10.0"),
            SubagentFinished(name="parser", duration_s=5.0),
            RunFinished(total_s=5.0, final_answer="v = 14 m/s"),
        ])

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: events)
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: "")

        result = CliRunner().invoke(app, ["solve", "-v", "test problem"])

        assert "python_exec" in result.output
        assert "h=10" in result.output

    def test_non_verbose_hides_tool_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from solvay.streaming import ToolCallMade

        events = iter([
            SubagentStarted(name="parser", t=0.0),
            ToolCallMade(subagent="parser", tool="python_exec", args_preview="secret_arg"),
            SubagentFinished(name="parser", duration_s=5.0),
            RunFinished(total_s=5.0, final_answer="v = 14 m/s"),
        ])

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: events)
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: "")

        result = CliRunner().invoke(app, ["solve", "test problem"])

        assert "secret_arg" not in result.output

    def test_report_saved_line_in_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("solvay.cli.create_solvay_agent", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("solvay.cli.parse_stream", lambda *a, **kw: _mock_events())
        monkeypatch.setattr("solvay.cli.generate_quarkdown", lambda *a, **kw: "")

        result = CliRunner().invoke(app, ["solve", "test problem"])

        assert "Report saved" in result.output
