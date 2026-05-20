"""Tests for the contamination probe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.probe import probe_contamination
from solvay.benchmark.schema import Problem


def _problem() -> Problem:
    return Problem.model_validate(
        {
            "id": "t",
            "version": 1,
            "source": {
                "kind": "synthetic",
                "origin": "t",
                "generated_at": "2026-04-18T00:00:00Z",
                "seed": 0,
            },
            "domain": "mechanics",
            "subdomain": None,
            "tags": [],
            "difficulty": "easy",
            "statement": "Strange unique problem X.",
            "given": {},
            "find": "x",
            "expected": {
                "kind": "numeric",
                "value": "1",
                "unit": None,
                "tolerance_rel": 0.01,
                "verification": {"method": "numeric_eval", "script": None},
            },
            "contamination": None,
            "notes": None,
        }
    )


def test_probe_returns_clean_when_model_denies_recognition() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.side_effect = [
        MagicMock(content='{"recognized": false, "source": null}'),
        MagicMock(content='{"continuation": "completely different text"}'),
        MagicMock(content='{"answer": "0"}'),
    ]
    with patch("solvay.benchmark.generator.probe.init_chat_model", return_value=fake_chat):
        result = probe_contamination(_problem(), BenchConfig())
    assert result.verdict == "clean"
    assert 0.0 <= result.score < 0.34


def test_probe_returns_contaminated_when_model_cites_source() -> None:
    fake_chat = MagicMock()
    fake_chat.invoke.side_effect = [
        MagicMock(content='{"recognized": true, "source": "Tipler 4.1"}'),
        MagicMock(content='{"continuation": "Strange unique problem X."}'),
        MagicMock(content='{"answer": "1"}'),
    ]
    with patch("solvay.benchmark.generator.probe.init_chat_model", return_value=fake_chat):
        result = probe_contamination(_problem(), BenchConfig())
    assert result.verdict == "contaminated"
    assert result.score > 0.66
