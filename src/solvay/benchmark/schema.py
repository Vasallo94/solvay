"""Pydantic models and JSON IO for benchmark problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["synthetic", "olympiad", "textbook"]
    origin: str
    generated_at: str
    seed: int | None = None


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["sympy_equivalence", "numeric_eval", "dimensional_only"]
    script: str | None = None


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["symbolic", "numeric"]
    value: str
    unit: str | None = None
    tolerance_rel: float = 0.01
    verification: Verification


class Contamination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checked_at: str
    method: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["clean", "suspect", "contaminated"]


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: int
    source: Source
    domain: str
    subdomain: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "intermediate", "hard"] = "intermediate"
    statement: str
    given: dict[str, str] = Field(default_factory=dict)
    find: str
    expected: Expected
    contamination: Contamination | None = None
    notes: str | None = None


def load_problem(path: Path) -> Problem:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Problem.model_validate(data)


def load_problems_dir(root: Path) -> list[Problem]:
    problems: list[Problem] = []
    for json_path in sorted(Path(root).rglob("*.json")):
        problems.append(load_problem(json_path))
    return problems


def dump_problem(problem: Problem, path: Path) -> None:
    path.write_text(problem.model_dump_json(indent=2) + "\n", encoding="utf-8")
