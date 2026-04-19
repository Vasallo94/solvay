"""'solvay-noweb' profile: full Solvay with Tavily web search disabled."""

from __future__ import annotations

import os
import time

from solvay.agent import create_solvay_agent
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.schema import Problem
from solvay.config import PersistenceConfig, SolvayConfig
from solvay.tools.python_exec import reset_exec_state


def solvay_noweb_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    reset_exec_state()
    saved = os.environ.pop("TAVILY_API_KEY", None)
    try:
        scfg = SolvayConfig(
            default_model=model,
            persistence=PersistenceConfig(enabled=False),
        )
        agent = create_solvay_agent(scfg)
        t0 = time.monotonic()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": problem.statement}]}
            )
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
    finally:
        if saved is not None:
            os.environ["TAVILY_API_KEY"] = saved


register(
    Profile(
        name="solvay-noweb",
        description="Full Solvay system with web search disabled (feeds metric A).",
        runner=solvay_noweb_runner,
        cost_estimate="high",
    )
)
