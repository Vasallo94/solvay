"""'bare' profile: raw model, no system prompt, no tools."""

from __future__ import annotations

import time

from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, register
from solvay.benchmark.schema import Problem


def bare_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    chat = init_chat_model(model, temperature=0)
    t0 = time.monotonic()
    try:
        response = chat.invoke([{"role": "user", "content": problem.statement}])
    except Exception as exc:
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    text = getattr(response, "content", "")
    usage = getattr(response, "usage_metadata", {}) or {}
    tokens = {
        "input": int(usage.get("input_tokens", 0) or 0),
        "output": int(usage.get("output_tokens", 0) or 0),
        "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
    }
    return ProfileResult(
        answer_raw=str(text),
        answer_extracted=None,
        elapsed_seconds=time.monotonic() - t0,
        tokens=tokens,
    )


register(
    Profile(
        name="bare",
        description="Raw model, no system prompt, no tools.",
        runner=bare_runner,
        cost_estimate="low",
    )
)
