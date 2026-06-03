"""FastMCP stdio server exposing Solvay physics solver as MCP tools."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from solvay.agent import create_solvay_agent
from solvay.config import SolvayConfig
from solvay.report import generate_quarkdown
from solvay.streaming import RunCollector, parse_stream

mcp = FastMCP("Solvay Physics Solver")

_OLLAMA_DEFAULT_MODEL = "ollama:qwen3.6:35b-a3b-coding-mxfp8"


# ---------------------------------------------------------------------------
# Implementation functions (separated from decorators for testability)
# ---------------------------------------------------------------------------


def _solve_physics_impl(
    problem: str,
    model: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Core implementation for solve_physics tool.

    Args:
        problem: The physics problem statement to solve.
        model: Optional model override string.
        output_dir: Directory for the .qd report file. Defaults to cwd.

    Returns:
        Dict with answer, report_path, duration_s, subagents_called, model keys.
    """
    # Resolve model following the precedence chain
    effective_model = model or os.environ.get("SOLVAY_MODEL", _OLLAMA_DEFAULT_MODEL)

    # Cap Ollama generation to prevent Qwen thinking-mode runaway
    model_kwargs: dict[str, Any] = {}
    if effective_model.startswith("ollama:"):
        model_kwargs = {"num_predict": 16384}

    config = SolvayConfig(default_model=model, model_kwargs=model_kwargs)

    agent = create_solvay_agent(config)
    collector = RunCollector(problem=problem, model=effective_model)

    for event in parse_stream(agent, problem):
        collector.accumulate(event)

    qd = generate_quarkdown(collector)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_name = f"solvay-report-{timestamp}.qd"
    out_dir = output_dir or Path.cwd()
    report_path = out_dir / report_name
    report_path.write_text(qd, encoding="utf-8")

    return {
        "answer": collector.final_answer,
        "report_path": str(report_path),
        "duration_s": collector.total_s,
        "subagents_called": [run.name for run in collector.subagent_runs],
        "model": effective_model,
    }


def _check_ollama_models() -> list[dict[str, str]]:
    """Query the local Ollama instance for available models.

    Returns:
        List of {name, provider} dicts, or empty list if Ollama is unavailable.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        models: list[dict[str, str]] = []
        lines = result.stdout.strip().splitlines()
        # Skip header line (NAME   ID   SIZE   MODIFIED)
        for line in lines[1:]:
            parts = line.split()
            if parts:
                name = parts[0]
                models.append({"name": f"ollama:{name}", "provider": "ollama"})
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _list_models_impl() -> dict[str, Any]:
    """Core implementation for list_models tool.

    Returns:
        Dict with models key containing list of {name, provider} dicts.
    """
    models: list[dict[str, str]] = []

    # Check Ollama
    models.extend(_check_ollama_models())

    # Check for API keys indicating cloud providers
    api_key_providers = [
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic:claude-sonnet-4-6"),
        ("GOOGLE_API_KEY", "google", "google_genai:gemini-2.5-pro"),
        ("OPENAI_API_KEY", "openai", "openai:gpt-4o"),
    ]
    for env_var, provider, example_model in api_key_providers:
        if os.environ.get(env_var):
            models.append({"name": example_model, "provider": provider})

    return {"models": models}


def _get_report_impl(path: str) -> dict[str, Any]:
    """Core implementation for get_report tool.

    Args:
        path: Path to the .qd report file.

    Returns:
        Dict with content key on success, or error key on failure.
    """
    report_path = Path(path)
    if not report_path.exists():
        return {"error": f"File not found: {path}"}
    content = report_path.read_text(encoding="utf-8")
    return {"content": content}


# ---------------------------------------------------------------------------
# MCP tool decorators
# ---------------------------------------------------------------------------


@mcp.tool
def solve_physics(problem: str, model: str | None = None) -> str:
    """Solve a physics problem using the Solvay multi-agent system.

    Args:
        problem: The physics problem statement to solve.
        model: Optional model override (e.g. 'anthropic:claude-sonnet-4-6',
               'ollama:qwen3.5', 'openai:gpt-4o'). Falls back to SOLVAY_MODEL
               env var or the default Ollama model.

    Returns:
        JSON string with answer, report_path, duration_s, subagents_called, model.
    """
    result = _solve_physics_impl(problem, model)
    return json.dumps(result)


@mcp.tool
def list_models() -> str:
    """List models available to Solvay (Ollama and API-key providers).

    Returns:
        JSON string with models array of {name, provider} objects.
    """
    result = _list_models_impl()
    return json.dumps(result)


@mcp.tool
def get_report(path: str) -> str:
    """Read a previously generated Solvay .qd report file.

    Args:
        path: Absolute or relative path to the .qd report file.

    Returns:
        JSON string with content key on success, or error key if file missing.
    """
    result = _get_report_impl(path)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
