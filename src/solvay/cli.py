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


def _notebook_content(files: Mapping[str, object]) -> str:
    """Return the run notebook content from native or legacy DeepAgents paths."""
    for path in ("/workspace/lab_notebook.md", "/lab_notebook.md"):
        entry = files.get(path)
        if isinstance(entry, dict):
            content = entry.get("content")
            if isinstance(content, str):
                return content
    return ""


def _stream_verbose(agent: object, problem: str) -> str:
    """Run agent with streaming progress and return the final answer."""
    import time

    t0 = time.monotonic()
    seen: list[str] = []
    step_start = t0
    final_content = ""

    for event in agent.stream(  # type: ignore[union-attr]
        {"messages": [{"role": "user", "content": problem}]},
        stream_mode="debug",
    ):
        if not isinstance(event, dict):
            continue
        etype = event.get("type", "")
        payload = event.get("payload", {})
        elapsed = time.monotonic() - t0

        if etype != "task_result" or not isinstance(payload, dict):
            continue
        result_data = payload.get("result", {})
        if not isinstance(result_data, dict):
            continue

        msgs = result_data.get("messages", [])
        if hasattr(msgs, "value"):
            msgs = msgs.value
        if not isinstance(msgs, list):
            continue

        for m in msgs:
            for tc in getattr(m, "tool_calls", []):
                if tc.get("name") == "task":
                    st = tc.get("args", {}).get("subagent_type", "")
                    if st and st not in seen:
                        if seen:
                            typer.echo(
                                f"  \033[32m✓\033[0m {seen[-1]}"
                                f"  ({time.monotonic() - step_start:.0f}s)"
                            )
                        seen.append(st)
                        step_start = time.monotonic()
                        typer.echo(f"  \033[36m⟳\033[0m {st}...")

            if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", []):
                c = getattr(m, "content", "")
                if isinstance(c, str) and len(c) > 20:
                    final_content = c

    if seen:
        typer.echo(
            f"  \033[32m✓\033[0m {seen[-1]}  ({time.monotonic() - step_start:.0f}s)"
        )

    total = time.monotonic() - t0
    typer.echo(f"\n\033[2mCompleted in {total / 60:.1f} min\033[0m\n")
    return final_content


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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show subagent progress as the pipeline runs."
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

    if verbose:
        final_message = _stream_verbose(agent, problem)
    else:
        result = agent.invoke({"messages": [{"role": "user", "content": problem}]})
        final_message = result["messages"][-1].content

    typer.echo("=" * 60)
    typer.echo(final_message)
    typer.echo("=" * 60)

    if trace is not None:
        if not verbose:
            trace.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        else:
            trace.write_text(json.dumps({"answer": final_message}, indent=2), encoding="utf-8")
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
