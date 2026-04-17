"""Tests for the Solvay CLI module (dotenv auto-loading, etc.)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
