"""Contamination probe: estimate how likely the target model has seen the problem."""

from __future__ import annotations

import datetime as dt
import difflib
import json
import re

from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Contamination, Problem

PROBE_METHOD_ID = "model-recall-probe-v1"


def _parse_json(text: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _recognition_probe(chat, problem: Problem) -> float:
    msg = (
        "Do you recognize the following physics problem? If yes, name the source "
        "(textbook, competition, year). Answer in JSON with keys 'recognized' "
        "(bool) and 'source' (string or null).\n\n"
        f"Problem: {problem.statement}"
    )
    resp = chat.invoke([{"role": "user", "content": msg}])
    obj = _parse_json(str(getattr(resp, "content", "")))
    return 1.0 if obj.get("recognized") and obj.get("source") else 0.0


def _continuation_probe(chat, problem: Problem) -> float:
    cut = max(20, int(len(problem.statement) * 0.3))
    head = problem.statement[:cut]
    tail = problem.statement[cut:]
    msg = (
        "Continue the following physics problem statement as you believe it is "
        "originally written. Return JSON with a single key 'continuation'.\n\n"
        f"Begin: {head}"
    )
    resp = chat.invoke([{"role": "user", "content": msg}])
    obj = _parse_json(str(getattr(resp, "content", "")))
    guess = str(obj.get("continuation", ""))
    if not tail or not guess:
        return 0.0
    return difflib.SequenceMatcher(None, tail.lower(), guess.lower()).ratio()


def _cold_answer_probe(chat, problem: Problem) -> float:
    msg = (
        "Give ONLY the final numerical or symbolic answer to this physics problem, "
        "no work shown. Return JSON with a single key 'answer'.\n\n"
        f"Problem: {problem.statement}"
    )
    resp = chat.invoke([{"role": "user", "content": msg}])
    obj = _parse_json(str(getattr(resp, "content", "")))
    guess = str(obj.get("answer", "")).strip()
    truth = problem.expected.value.strip()
    if not guess:
        return 0.0
    return 1.0 if guess == truth else 0.0


def probe_contamination(problem: Problem, config: BenchConfig) -> Contamination:
    chat = init_chat_model(config.probe_model, temperature=0)
    scores = [
        _recognition_probe(chat, problem),
        _continuation_probe(chat, problem),
        _cold_answer_probe(chat, problem),
    ]
    score = sum(scores) / len(scores)
    if score < 0.34:
        verdict: str = "clean"
    elif score < 0.67:
        verdict = "suspect"
    else:
        verdict = "contaminated"
    return Contamination(
        checked_at=dt.datetime.now(dt.UTC).isoformat(),
        method=PROBE_METHOD_ID,
        score=round(score, 3),
        verdict=verdict,  # type: ignore[arg-type]
    )
