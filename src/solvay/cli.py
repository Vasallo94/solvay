"""Typer CLI for Solvay: solve problems and run benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load variables from a project-root ``.env`` file (ANTHROPIC_API_KEY,
# TAVILY_API_KEY, LANGSMITH_*, SOLVAY_MODEL, ...) before any langchain /
# langgraph / langsmith imports fire, so those libraries pick the values up
# on first import. ``override=False`` (the default) means real shell exports
# still win over ``.env`` -- handy for one-off CLI overrides.
load_dotenv()

app = typer.Typer(
    name="solvay",
    help="Multi-agent physics problem solver.",
    no_args_is_help=True,
)


def _notebook_content(files: Mapping[str, object]) -> str:
    """Return the run notebook content from native or legacy DeepAgents paths."""
    for path in ("/workspace/lab_notebook.md", "/lab_notebook.md"):
        entry = files.get(path)
        if isinstance(entry, dict):
            content = entry.get("content")
            if isinstance(content, str):
                return content
    return ""


@app.command()
def solve(
    statement: str | None = typer.Argument(None, help="The physics problem statement."),
    file: Path | None = typer.Option(None, "--file", "-f", help="Read problem from a text file."),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Override the model for all roles. Accepts any string understood "
            "by langchain.chat_models.init_chat_model, e.g. "
            "'anthropic:claude-sonnet-4-6', 'openai:gpt-4o', "
            "'google_genai:gemini-2.5-pro', or 'ollama:qwen3.5'. "
            "Takes precedence over the SOLVAY_MODEL env var."
        ),
    ),
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
    config = SolvayConfig(persistence=persistence_config, default_model=model)
    typer.echo(f"Model: {config.model_for('orchestrator')}\n")

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
        notebook = _notebook_content(files) or "(no notebook found)"
        typer.echo("\n--- Lab Notebook ---")
        typer.echo(notebook)

    if trace is not None:
        trace.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        typer.echo(f"\nTrace saved to {trace}")

    if persistence_config.enabled:
        files = result.get("files", {})
        notebook_content = _notebook_content(files)
        if notebook_content:
            save_session_notebook(persistence_config, problem[:40], notebook_content)


@app.command()
def bench(
    problems: Path = typer.Option(
        "benchmark/problems",
        "--problems",
        "-p",
        help="Root directory containing benchmark problem JSON files.",
    ),
    profiles: str = typer.Option(
        "solvay-full",
        "--profiles",
        help="Comma-separated profile names, or 'all'.",
    ),
    models: str = typer.Option(
        "",
        "--models",
        help=(
            "Comma-separated model strings for langchain.chat_models.init_chat_model, "
            "for example 'ollama:qwen3.5'. Defaults to Solvay's default model."
        ),
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help="Filter problems to this domain subdirectory, e.g. mechanics.",
    ),
    repeat: int = typer.Option(1, "--repeat", min=1, help="Repeats per matrix cell."),
    confirm_cost: bool = typer.Option(
        False,
        "--confirm-cost",
        help="Confirm running a matrix above the benchmark cost threshold.",
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output results JSONL file."),
) -> None:
    """Run the benchmark matrix with the new solvay.benchmark runner."""
    from solvay.benchmark.cli import _import_builtin_profiles
    from solvay.benchmark.config import BenchConfig
    from solvay.benchmark.profiles import PROFILES
    from solvay.benchmark.runner import MatrixSpec, run_matrix
    from solvay.benchmark.schema import load_problems_dir
    from solvay.config import DEFAULT_MODEL

    root = problems / domain if domain else problems
    if not root.exists():
        typer.echo(f"Error: problems directory not found: {root}", err=True)
        raise typer.Exit(code=1)

    _import_builtin_profiles()
    all_problems = load_problems_dir(root)
    if not all_problems:
        typer.echo(f"No problems found under {root}.", err=True)
        raise typer.Exit(code=1)

    profile_names = (
        list(PROFILES) if profiles == "all" else [p.strip() for p in profiles.split(",")]
    )
    for profile_name in profile_names:
        if profile_name not in PROFILES:
            typer.echo(
                f"Unknown profile: {profile_name}. Known: {sorted(PROFILES)}",
                err=True,
            )
            raise typer.Exit(code=2)

    model_list = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
    config = BenchConfig()
    invocations = len(all_problems) * len(profile_names) * len(model_list) * repeat
    if invocations > config.cost_guard_threshold and not confirm_cost:
        typer.echo(
            f"Matrix would make {invocations} invocations "
            f"(threshold {config.cost_guard_threshold}). "
            "Pass --confirm-cost to proceed.",
            err=True,
        )
        raise typer.Exit(code=3)

    out_path = out or Path("benchmark/runs") / "latest.jsonl"
    spec = MatrixSpec(
        problems=all_problems,
        profile_names=profile_names,
        models=model_list,
        repeats=repeat,
    )
    path = run_matrix(spec, out_path=out_path, config=config)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
