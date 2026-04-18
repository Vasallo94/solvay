"""Typer CLI for the Solvay benchmark suite."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="solvay-bench",
    help="Generate, run, and report on the Solvay benchmark suite.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Solvay benchmark suite."""


if __name__ == "__main__":
    app()
