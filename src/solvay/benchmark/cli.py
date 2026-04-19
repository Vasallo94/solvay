"""Typer CLI for the Solvay benchmark suite."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

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


@app.command("run")
def run_cmd(
    problems: Annotated[
        Path, typer.Option("--problems", help="Root directory of problem JSONs.")
    ] = Path("benchmark/problems"),
    profiles: Annotated[
        str,
        typer.Option(
            "--profiles",
            help="Comma-separated profile names, or 'all'.",
        ),
    ] = "solvay-full",
    models: Annotated[
        str,
        typer.Option(
            "--models",
            help="Comma-separated model strings for langchain.chat_models.init_chat_model.",
        ),
    ] = "",
    domain: Annotated[
        str | None,
        typer.Option("--domain", help="Filter problems to this domain (subdirectory)."),
    ] = None,
    repeat: Annotated[int, typer.Option("--repeat", min=1)] = 1,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination JSONL (default: benchmark/runs/<id>.jsonl)."),
    ] = None,
    confirm_cost: Annotated[
        bool,
        typer.Option("--confirm-cost", help="Confirm running a matrix above the cost threshold."),
    ] = False,
) -> None:
    """Execute the (problems x profiles x models x repeats) matrix."""
    # Import locally so the module tree cost isn't paid on --help.
    from solvay.benchmark.config import BenchConfig
    from solvay.benchmark.profiles import PROFILES
    from solvay.benchmark.runner import MatrixSpec, run_matrix
    from solvay.benchmark.schema import load_problems_dir
    from solvay.config import DEFAULT_MODEL

    # Force import of the built-in profiles so they self-register. The try/except
    # lets earlier phases run before Phase 2 is merged.
    try:
        import solvay.benchmark.profiles.bare
        import solvay.benchmark.profiles.prompted
        import solvay.benchmark.profiles.solvay_full  # type: ignore[import-not-found]  # noqa: F401, RUF100
        import solvay.benchmark.profiles.solvay_noweb  # type: ignore[import-not-found]  # noqa: F401, RUF100
        import solvay.benchmark.profiles.tooled  # noqa: F401
    except ImportError:
        pass

    root = problems / domain if domain else problems
    all_problems = load_problems_dir(root)
    if not all_problems:
        typer.echo(f"No problems found under {root}.", err=True)
        raise typer.Exit(1)

    profile_names = (
        list(PROFILES) if profiles == "all" else [p.strip() for p in profiles.split(",")]
    )
    for pn in profile_names:
        if pn not in PROFILES:
            typer.echo(f"Unknown profile: {pn}. Known: {sorted(PROFILES)}", err=True)
            raise typer.Exit(2)

    model_list = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
    cfg = BenchConfig()
    invocations = len(all_problems) * len(profile_names) * len(model_list) * repeat
    if invocations > cfg.cost_guard_threshold and not confirm_cost:
        typer.echo(
            f"Matrix would make {invocations} invocations "
            f"(threshold {cfg.cost_guard_threshold}). "
            "Pass --confirm-cost to proceed.",
            err=True,
        )
        raise typer.Exit(3)

    out_path = out or Path("benchmark/runs") / (
        dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S") + ".jsonl"
    )
    spec = MatrixSpec(
        problems=all_problems,
        profile_names=profile_names,
        models=model_list,
        repeats=repeat,
    )
    path = run_matrix(spec, out_path=out_path, config=cfg)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
