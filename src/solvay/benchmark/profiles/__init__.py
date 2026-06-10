"""Profile registry: encapsulate the different experimental conditions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Problem


def aggregate_tokens(messages: Sequence[Any]) -> dict[str, int]:
    """Sum LangChain ``usage_metadata`` across the messages of an agent run."""
    totals = {"input": 0, "output": 0, "cache_read": 0}
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        totals["input"] += int(usage.get("input_tokens", 0) or 0)
        totals["output"] += int(usage.get("output_tokens", 0) or 0)
        details = usage.get("input_token_details")
        if isinstance(details, dict):
            totals["cache_read"] += int(details.get("cache_read", 0) or 0)
    return totals


@dataclass(frozen=True)
class ProfileResult:
    """Raw output of a single (problem, profile, model) invocation."""

    answer_raw: str
    answer_extracted: str | None
    elapsed_seconds: float
    tokens: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None
    error: str | None = None


ProfileRunner = Callable[[Problem, str, BenchConfig], ProfileResult]


@dataclass(frozen=True)
class Profile:
    """What apparatus runs against a problem."""

    name: str
    description: str
    runner: ProfileRunner
    cost_estimate: Literal["low", "medium", "high"]


PROFILES: dict[str, Profile] = {}


def register(profile: Profile) -> None:
    PROFILES[profile.name] = profile


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile: {name!r}. Known: {sorted(PROFILES)}")
    return PROFILES[name]


def all_profiles() -> list[Profile]:
    return list(PROFILES.values())
