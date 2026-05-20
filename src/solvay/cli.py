"""Typer CLI for Solvay: solve problems and run benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load variables from a project-root ``.env`` file (ANTHROPIC_API_KEY,
# TAVILY_API_KEY, LANGSMITH_*, SOLVAY_MODEL, ...) before any langchain /
# langgraph / langsmith imports fire, so those libraries pick the values up
# on first import. ``override=False`` (the default) means real shell exports
# still win over ``.env`` -- handy for one-off CLI overrides.
load_dotenv()

# Disable LangSmith tracing when no API key is configured to avoid noisy
# 403 errors on every model call.
import os as _os
if not _os.environ.get("LANGSMITH_API_KEY"):
    _os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    _os.environ.setdefault("LANGSMITH_TRACING", "false")

app = typer.Typer(
    name="solvay",
    help="Multi-agent physics problem solver.",
    no_args_is_help=True,
)


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
    trace: Path | None = typer.Option(
        None, "--trace", help="Dump full trace as JSON to this file."
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
    from solvay.config import SolvayConfig

    config = SolvayConfig(default_model=model)
    typer.echo(f"Model: {config.model_for('orchestrator')}\n")

    agent = create_solvay_agent(config)

    result = agent.invoke({"messages": [{"role": "user", "content": problem}]})

    final_message = result["messages"][-1].content
    typer.echo("=" * 60)
    typer.echo(final_message)
    typer.echo("=" * 60)

    if trace is not None:
        trace.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        typer.echo(f"\nTrace saved to {trace}")


@app.command()
def chat(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the model for all roles.",
    ),
) -> None:
    """Interactive REPL — chat with the Solvay physics solver."""
    import time

    from solvay.agent import create_solvay_agent
    from solvay.config import SolvayConfig

    config = SolvayConfig(default_model=model)
    typer.echo(f"Solvay  model: {config.model_for('orchestrator')}")
    typer.echo("Type a physics problem. Ctrl+C to exit.\n")

    agent = create_solvay_agent(config)
    messages: list[dict[str, str]] = []

    while True:
        try:
            problem = input("\033[1;32mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nBye!")
            break

        if not problem:
            continue

        messages.append({"role": "user", "content": problem})
        final_content = ""
        seen_subagents: list[str] = []
        subagent_start = time.monotonic()

        try:
            for chunk in agent.stream(
                {"messages": list(messages)},
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            ):
                if not isinstance(chunk, dict):
                    continue

                ns = chunk.get("ns", ())
                data = chunk.get("data", {})

                if not isinstance(data, dict):
                    continue

                for _node, update in data.items():
                    if not isinstance(update, dict):
                        continue

                    raw_msgs = update.get("messages", [])
                    if hasattr(raw_msgs, "value"):
                        raw_msgs = raw_msgs.value
                    if not isinstance(raw_msgs, list):
                        continue

                    for msg in raw_msgs:
                        # Detect task() tool calls → subagent dispatch
                        for tc in getattr(msg, "tool_calls", []):
                            if tc.get("name") == "task":
                                st = tc.get("args", {}).get("subagent_type", "")
                                if st and st not in seen_subagents:
                                    if seen_subagents:
                                        elapsed = time.monotonic() - subagent_start
                                        typer.echo(
                                            f"  \033[32m✓\033[0m {seen_subagents[-1]}"
                                            f"  ({elapsed:.0f}s)"
                                        )
                                    seen_subagents.append(st)
                                    subagent_start = time.monotonic()
                                    typer.echo(f"  \033[36m⟳\033[0m {st}...")

                        # Track final AI message from root namespace
                        if not ns:
                            content = getattr(msg, "content", None)
                            if content and isinstance(content, str):
                                final_content = content

            if seen_subagents:
                elapsed = time.monotonic() - subagent_start
                typer.echo(
                    f"  \033[32m✓\033[0m {seen_subagents[-1]}  ({elapsed:.0f}s)"
                )

        except KeyboardInterrupt:
            typer.echo("\n\033[33mInterrupted.\033[0m\n")
            messages.pop()
            continue

        if final_content:
            typer.echo(f"\n\033[1;34mSolvay:\033[0m\n{final_content}\n")
            messages.append({"role": "assistant", "content": final_content})
        else:
            typer.echo("\033[33m(no response)\033[0m\n")
            messages.pop()


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
