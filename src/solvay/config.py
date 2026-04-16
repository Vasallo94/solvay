"""Central configuration for Solvay: models, budgets, defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal[
    "orchestrator", "parser", "researcher", "solver",
    "verifier", "peer_reviewer", "consolidator",
]

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


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


@dataclass
class SolvayConfig:
    """Top-level configuration for a Solvay run."""

    models: dict[Role, str] = field(default_factory=lambda: {})
    solver_loop: SolverLoopConfig = field(default_factory=SolverLoopConfig)
    python_exec: PythonExecConfig = field(default_factory=PythonExecConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    def model_for(self, role: Role) -> str:
        """Return the model string for a given role, falling back to default."""
        return self.models.get(role, DEFAULT_MODEL)
