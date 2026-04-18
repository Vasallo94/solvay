"""Benchmark-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass

from solvay.config import DEFAULT_MODEL


@dataclass(frozen=True)
class BenchConfig:
    """Configuration for the benchmark suite.

    All LLM-consuming components of the benchmark read their model strings
    from here. Kept separate from :class:`solvay.config.SolvayConfig` because
    these knobs are meta-uses of LLMs (generation, evaluation) rather than
    solver-time choices.
    """

    composer_model: str = DEFAULT_MODEL
    probe_model: str = DEFAULT_MODEL
    error_classifier_model: str = DEFAULT_MODEL
    composer_max_attempts: int = 3
    cost_guard_threshold: int = 100
