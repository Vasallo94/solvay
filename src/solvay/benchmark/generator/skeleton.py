"""Input specification for a single generation attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GenerationSkeleton:
    domain: str
    compose: list[str]
    subdomain: str | None = None
    difficulty: Literal["easy", "intermediate", "hard"] = "intermediate"
    approach: Literal["backward", "forward"] = "backward"
    seed: int | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.compose:
            raise ValueError("compose must contain at least one concept")
