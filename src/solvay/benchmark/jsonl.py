"""JSONL writer/reader for benchmark runs.

Format: the first line of every run file is a header record tagged
``"kind": "header"``. Every subsequent line is a data record tagged
``"kind": "record"``. Both types are self-describing so the file is
forward-compatible with future fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class RunHeader:
    run_id: str
    solvay_version: str
    git_commit: str
    git_dirty: bool
    config_fingerprint: str
    started_at: str


@dataclass(frozen=True)
class RunRecord:
    problem_id: str
    profile: str
    model: str
    repeat_idx: int
    answer_raw: str
    answer_extracted: str | None
    correct: bool
    elapsed_seconds: float
    tokens: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None
    error: str | None = None
    grading_method: str = "regex"


class RunWriter:
    """Append-safe JSONL writer. Writes header on first open if file is new."""

    def __init__(self, path: Path, header: RunHeader, append: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        mode = "a" if append or not is_new else "w"
        self._fp: IO[str] = self._path.open(mode, encoding="utf-8")
        if is_new:
            self._fp.write(json.dumps({"kind": "header", **asdict(header)}) + "\n")
            self._fp.flush()

    def write(self, record: RunRecord) -> None:
        self._fp.write(json.dumps({"kind": "record", **asdict(record)}) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_run(path: Path) -> tuple[RunHeader, list[RunRecord]]:
    header: RunHeader | None = None
    records: list[RunRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.pop("kind")
        if kind == "header":
            header = RunHeader(**obj)
        elif kind == "record":
            records.append(RunRecord(**obj))
    if header is None:
        raise ValueError(f"No header found in {path}")
    return header, records
