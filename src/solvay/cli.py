"""Typer CLI for Solvay: solve problems and run benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="solvay",
    help="Multi-agent physics problem solver.",
    no_args_is_help=True,
)


@app.command()
def solve(
    statement: str | None = typer.Argument(None, help="The physics problem statement."),
    file: Path | None = typer.Option(None, "--file", "-f", help="Read problem from a text file."),
    show_notebook: bool = typer.Option(
        False, "--show-notebook", help="Show full lab notebook after solving."
    ),
    trace: Path | None = typer.Option(
        None, "--trace", help="Dump full trace as JSON to this file."
    ),
    no_persist: bool = typer.Option(
        False, "--no-persist", help="Disable cross-run journal persistence."
    ),
) -> None:
    """Solve a physics problem using the multi-agent system."""
    if file is not None:
        if not file.exists():
            typer.echo(f"Error: file not found: {file}", err=True)
            raise typer.Exit(code=1)
        problem = file.read_text(encoding="utf-8").strip()
    elif statement is not None:
        problem = statement
    else:
        typer.echo("Error: provide a problem statement or --file.", err=True)
        raise typer.Exit(code=1)

    if not problem:
        typer.echo("Error: empty problem statement.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Solving: {problem[:80]}{'...' if len(problem) > 80 else ''}\n")

    from solvay.agent import create_solvay_agent
    from solvay.config import PersistenceConfig, SolvayConfig
    from solvay.persistence import (
        load_journal_snapshot,
        save_session_notebook,
    )

    persistence_config = PersistenceConfig(enabled=not no_persist)
    config = SolvayConfig(persistence=persistence_config)

    journal_snapshot = ""
    if persistence_config.enabled:
        journal_snapshot = load_journal_snapshot(persistence_config)

    agent = create_solvay_agent(config)

    user_message = problem
    if journal_snapshot:
        user_message += (
            f"\n\n---\nPrevious session learnings (from lab journal):\n{journal_snapshot}\n---"
        )

    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    final_message = result["messages"][-1].content
    typer.echo("=" * 60)
    typer.echo(final_message)
    typer.echo("=" * 60)

    if show_notebook:
        files = result.get("files", {})
        notebook = files.get("/lab_notebook.md", {}).get("content", "(no notebook found)")
        typer.echo("\n--- Lab Notebook ---")
        typer.echo(notebook)

    if trace is not None:
        trace.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        typer.echo(f"\nTrace saved to {trace}")

    if persistence_config.enabled:
        files = result.get("files", {})
        notebook_content = files.get("/lab_notebook.md", {}).get("content", "")
        if notebook_content:
            save_session_notebook(persistence_config, problem[:40], notebook_content)


@app.command()
def bench(
    problems: Path = typer.Option(
        "benchmark/problems",
        "--problems",
        "-p",
        help="Directory containing benchmark problem JSON files.",
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output results JSON file."),
) -> None:
    """Run the benchmark suite against curated physics problems."""
    if not problems.exists():
        typer.echo(f"Error: problems directory not found: {problems}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Running benchmark from {problems}...")

    sys.path.insert(0, str(Path.cwd()))
    from benchmark.run_bench import run_benchmark

    results = run_benchmark(problems_dir=problems)

    if out is not None:
        out.write_text(json.dumps(results, default=str, indent=2), encoding="utf-8")
        typer.echo(f"Results saved to {out}")
    else:
        typer.echo(json.dumps(results, default=str, indent=2))


if __name__ == "__main__":
    app()
