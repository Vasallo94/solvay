"""Central configuration for Solvay: models, budgets, defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

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
    harness: HarnessConfig = field(default_factory=HarnessConfig)

    def model_for(self, role: Role) -> "str | BaseChatModel":
        """Return the model for a given role, walking the precedence ladder.

        Returns a string for most providers (passed to init_chat_model by deepagents).
        Returns a BaseChatModel instance for ``vertexai:`` models, because
        ChatAnthropicVertex is not registered with init_chat_model.
        """
        if role in self.models:
            model = self.models[role]
        elif self.default_model is not None:
            model = self.default_model
        else:
            env_override = os.environ.get(ENV_MODEL_VAR)
            model = env_override if env_override else DEFAULT_MODEL

        if isinstance(model, str) and model.startswith("vertexai:"):
            return _build_vertex_model(model)
        return model


def resolve_model(model: "str | BaseChatModel") -> "BaseChatModel":
    """Return a BaseChatModel, calling init_chat_model if given a string."""
    from langchain_core.language_models import BaseChatModel as _BaseChatModel

    if isinstance(model, _BaseChatModel):
        return model
    from langchain.chat_models import init_chat_model

    return init_chat_model(model)


def _build_vertex_model(model_string: str) -> "BaseChatModel":
    """Instantiate ChatAnthropicVertex from a ``vertexai:<model-name>`` string."""
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    model_name = model_string.removeprefix("vertexai:")
    project = os.environ.get("VERTEX_PROJECT", "")
    location = os.environ.get("VERTEX_LOCATION", "eu")
    if not project:
        raise ValueError(
            "VERTEX_PROJECT env var is required when using a vertexai: model."
        )
    return ChatAnthropicVertex(
        model_name=model_name,
        project=project,
        location=location,
    )
