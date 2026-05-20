"""'solvay-full' profile: complete Solvay system with web access."""

from __future__ import annotations

import time

from solvay.agent import create_solvay_agent
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.schema import Problem
from solvay.config import SolvayConfig
from solvay.tools.python_exec import reset_exec_state


def solvay_full_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    reset_exec_state()
    scfg = SolvayConfig(default_model=model)
    agent = create_solvay_agent(scfg)
    t0 = time.monotonic()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": problem.statement}]})
    except Exception as exc:
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    final = result["messages"][-1]
    return ProfileResult(
        answer_raw=str(getattr(final, "content", "")),
        answer_extracted=None,
        elapsed_seconds=time.monotonic() - t0,
        tokens={"input": 0, "output": 0, "cache_read": 0},
    )


register(
    Profile(
        name="solvay-full",
        description="Full Solvay system with web search enabled.",
        runner=solvay_full_runner,
        cost_estimate="high",
    )
)
