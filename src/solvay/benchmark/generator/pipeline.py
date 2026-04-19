"""Generator orchestrator: compose -> verify -> probe -> write."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.composer import compose_problem
from solvay.benchmark.generator.probe import probe_contamination
from solvay.benchmark.generator.skeleton import GenerationSkeleton
from solvay.benchmark.generator.verifier import verify_problem
from solvay.benchmark.schema import dump_problem


@dataclass
class GenerationReport:
    attempted: int = 0
    written: int = 0
    rejected_verification: int = 0
    rejected_compose: int = 0
    contamination_breakdown: dict[str, int] = field(default_factory=dict)
    written_paths: list[Path] = field(default_factory=list)


def generate_batch(
    skeletons: list[GenerationSkeleton],
    out_dir: Path,
    config: BenchConfig,
) -> GenerationReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = GenerationReport()
    for sk in skeletons:
        report.attempted += 1
        try:
            problem = compose_problem(sk, config)
        except Exception:
            report.rejected_compose += 1
            continue
        outcome = verify_problem(problem)
        if not outcome.ok:
            report.rejected_verification += 1
            continue
        contamination = probe_contamination(problem, config)
        problem = problem.model_copy(update={"contamination": contamination})
        report.contamination_breakdown[contamination.verdict] = (
            report.contamination_breakdown.get(contamination.verdict, 0) + 1
        )
        path = out_dir / f"{problem.id}.json"
        dump_problem(problem, path)
        report.written += 1
        report.written_paths.append(path)
    return report
