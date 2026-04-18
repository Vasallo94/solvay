"""Tests for the profile registry."""

from __future__ import annotations

import pytest

from solvay.benchmark.profiles import (
    PROFILES,
    Profile,
    ProfileResult,
    get_profile,
    register,
)


def test_profile_result_fields() -> None:
    r = ProfileResult(
        answer_raw="42",
        answer_extracted="42",
        elapsed_seconds=0.1,
        tokens={"input": 1, "output": 1, "cache_read": 0},
        trace_id=None,
        error=None,
    )
    assert r.answer_raw == "42"


def test_register_adds_profile() -> None:
    def runner(problem, model, config):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    p = Profile(name="unit-test", description="desc", runner=runner, cost_estimate="low")
    register(p)
    try:
        assert get_profile("unit-test") is p
    finally:
        PROFILES.pop("unit-test", None)


def test_get_profile_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("does-not-exist")
