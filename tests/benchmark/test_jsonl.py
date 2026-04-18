"""Tests for the JSONL run writer/reader."""

from __future__ import annotations

from pathlib import Path

from solvay.benchmark.jsonl import RunHeader, RunRecord, RunWriter, read_run


def _header() -> RunHeader:
    return RunHeader(
        run_id="2026-04-18-test",
        solvay_version="0.1.0",
        git_commit="abc1234",
        git_dirty=False,
        config_fingerprint="sha256:deadbeef",
        started_at="2026-04-18T00:00:00Z",
    )


def _record(problem_id: str = "p1") -> RunRecord:
    return RunRecord(
        problem_id=problem_id,
        profile="bare",
        model="anthropic:claude-sonnet-4-6",
        repeat_idx=0,
        answer_raw="42",
        answer_extracted="42",
        correct=True,
        elapsed_seconds=0.5,
        tokens={"input": 100, "output": 50, "cache_read": 0},
        trace_id=None,
        error=None,
    )


def test_writer_emits_header_then_records(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = RunWriter(path, _header())
    writer.write(_record("p1"))
    writer.write(_record("p2"))
    writer.close()

    header, records = read_run(path)
    assert header.run_id == "2026-04-18-test"
    assert [r.problem_id for r in records] == ["p1", "p2"]


def test_writer_is_append_safe(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    w = RunWriter(path, _header())
    w.write(_record("p1"))
    w.close()

    # Re-open to append: keep the same header, do not rewrite it.
    w2 = RunWriter(path, _header(), append=True)
    w2.write(_record("p2"))
    w2.close()

    _, records = read_run(path)
    assert [r.problem_id for r in records] == ["p1", "p2"]
