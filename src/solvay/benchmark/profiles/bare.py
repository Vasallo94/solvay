"""'bare' profile: raw model, no system prompt, no tools."""

from __future__ import annotations

import time

from langchain.chat_models import init_chat_model

from solvay.benchmark.answer_format import (
    append_final_answer_instruction,
    extract_final_answer,
)
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, aggregate_tokens, register
from solvay.benchmark.schema import Problem


def bare_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    chat = init_chat_model(model, temperature=0)
    content = append_final_answer_instruction(problem.statement)
    t0 = time.monotonic()
    try:
        response = chat.invoke([{"role": "user", "content": content}])
    except Exception as exc:
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    text = str(getattr(response, "content", ""))
    return ProfileResult(
        answer_raw=text,
        answer_extracted=extract_final_answer(text),
        elapsed_seconds=time.monotonic() - t0,
        tokens=aggregate_tokens([response]),
    )


register(
    Profile(
        name="bare",
        description="Raw model, no system prompt, no tools.",
        runner=bare_runner,
        cost_estimate="low",
    )
)
