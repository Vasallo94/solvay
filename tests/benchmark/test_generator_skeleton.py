"""Tests for the generation skeleton."""

from __future__ import annotations

from solvay.benchmark.generator.skeleton import GenerationSkeleton


def test_skeleton_defaults() -> None:
    sk = GenerationSkeleton(domain="mechanics", compose=["pendulum"])
    assert sk.approach == "backward"
    assert sk.difficulty == "intermediate"


def test_skeleton_rejects_empty_compose() -> None:
    import pytest

    with pytest.raises(ValueError):
        GenerationSkeleton(domain="mechanics", compose=[])
