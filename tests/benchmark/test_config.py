"""Tests for BenchConfig."""

from __future__ import annotations

from solvay.benchmark.config import BenchConfig


def test_bench_config_defaults() -> None:
    cfg = BenchConfig()
    assert cfg.composer_model
    assert cfg.probe_model
    assert cfg.error_classifier_model
    assert cfg.cost_guard_threshold == 100


def test_bench_config_overrides() -> None:
    cfg = BenchConfig(composer_model="openai:gpt-5", cost_guard_threshold=500)
    assert cfg.composer_model == "openai:gpt-5"
    assert cfg.cost_guard_threshold == 500
