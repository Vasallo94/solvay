"""Profile registry: encapsulate the different experimental conditions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Problem


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


def extract_text(content: object) -> str:
    """Extract plain text from a model response content field.

    Handles both plain strings and Responses API content-block lists
    (list of dicts with 'type' and 'text' keys).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts) if parts else str(content)
    return str(content)


PROFILES: dict[str, Profile] = {}


def register(profile: Profile) -> None:
    PROFILES[profile.name] = profile


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile: {name!r}. Known: {sorted(PROFILES)}")
    return PROFILES[name]


def all_profiles() -> list[Profile]:
    return list(PROFILES.values())
