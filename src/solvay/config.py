"""Central configuration for Solvay: models, budgets, defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Role = Literal[
    "orchestrator",
    "parser",
    "researcher",
    "solver",
    "verifier",
    "peer_reviewer",
    "consolidator",
]

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
"""Fallback model used when nothing else is configured. Accepts any string
understood by :func:`langchain.chat_models.init_chat_model`, e.g.
``"anthropic:claude-sonnet-4-6"``, ``"openai:gpt-4o"``,
``"google_genai:gemini-2.5-pro"``, or ``"ollama:qwen3.5"``."""

ENV_MODEL_VAR = "SOLVAY_MODEL"
"""Environment variable consulted when neither ``models[role]`` nor
``default_model`` is set. Useful for running against local Ollama without
touching code: ``SOLVAY_MODEL=ollama:qwen3.5 uv run solvay solve ...``."""


@dataclass(frozen=True)
class SolverLoopConfig:
    """Configuration for the solver-critique loop."""

    max_iterations: int = 3


@dataclass(frozen=True)
class PythonExecConfig:
    """Configuration for the sandboxed Python execution tool."""

    timeout_seconds: int = 30
    memory_limit_mb: int = 512


@dataclass(frozen=True)
class NotebookConfig:
    """Configuration for notebook injection middleware."""

    max_injected_entries: int = 20


@dataclass(frozen=True)
class PersistenceConfig:
    """Configuration for cross-run persistence."""

    solvay_dir: str = "~/.solvay"
    enabled: bool = True


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for native DeepAgents harness features."""

    native_memory_enabled: bool = True
    memory_namespace: str = "solvay"
    persist_harness_notes: bool = True


@dataclass
class SolvayConfig:
    """Top-level configuration for a Solvay run.

    Model resolution in :meth:`model_for` follows this precedence:

    1. ``models[role]`` -- explicit per-role override
    2. ``default_model`` -- global override (typically set from ``--model`` CLI flag)
    3. ``$SOLVAY_MODEL`` environment variable
    4. :data:`DEFAULT_MODEL` hardcoded fallback
    """

    models: dict[Role, str] = field(default_factory=lambda: {})
    default_model: str | None = None
    solver_loop: SolverLoopConfig = field(default_factory=SolverLoopConfig)
    python_exec: PythonExecConfig = field(default_factory=PythonExecConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)

    def model_for(self, role: Role) -> str:
        """Return the model string for a given role, walking the precedence ladder."""
        if role in self.models:
            return self.models[role]
        if self.default_model is not None:
            return self.default_model
        env_override = os.environ.get(ENV_MODEL_VAR)
        if env_override:
            return env_override
        return DEFAULT_MODEL
