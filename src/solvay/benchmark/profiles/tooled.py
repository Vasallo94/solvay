"""'tooled' profile: single agent with local tools, no web, no orchestration."""

from __future__ import annotations

import time

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from solvay.benchmark.answer_format import (
    append_final_answer_instruction,
    extract_final_answer,
)
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import Profile, ProfileResult, aggregate_tokens, register
from solvay.benchmark.profiles.system_prompt import PHYSICIST_SYSTEM_PROMPT
from solvay.benchmark.schema import Problem
from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec, reset_exec_state


def tooled_runner(problem: Problem, model: str, config: BenchConfig) -> ProfileResult:
    reset_exec_state()
    chat = init_chat_model(model, temperature=0)
    agent = create_react_agent(
        chat,
        tools=[python_exec, check_dimensions],
        prompt=PHYSICIST_SYSTEM_PROMPT,
    )
    content = append_final_answer_instruction(problem.statement)
    t0 = time.monotonic()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    except Exception as exc:
        return ProfileResult(
            answer_raw="",
            answer_extracted=None,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    messages = result["messages"]
    text = str(getattr(messages[-1], "content", ""))
    return ProfileResult(
        answer_raw=text,
        answer_extracted=extract_final_answer(text),
        elapsed_seconds=time.monotonic() - t0,
        tokens=aggregate_tokens(messages),
    )


register(
    Profile(
        name="tooled",
        description="Single agent with sympy/python_exec/dimensional tools, no web.",
        runner=tooled_runner,
        cost_estimate="medium",
    )
)
