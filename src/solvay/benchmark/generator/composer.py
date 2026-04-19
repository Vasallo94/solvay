"""LLM-driven authoring of benchmark problems."""

from __future__ import annotations

import json
import re

from langchain.chat_models import init_chat_model
from pydantic import ValidationError

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.generator.skeleton import GenerationSkeleton
from solvay.benchmark.schema import Problem


_COMPOSER_SYSTEM = """\
You are a physics problem-writer for a benchmark that measures reasoning capacity.
Your job: produce ONE problem that composes the listed concepts in a way that does
not appear in any standard textbook. The problem MUST be answerable in closed form
or with a numerical integration.

Requirements:
- Emit ONLY a single JSON object matching the target schema. No prose, no markdown,
  no code fences.
- When approach="backward", first pick the closed-form answer, then construct a
  statement that leads to it. This is the default — use it whenever feasible.
- When approach="forward", compose the statement freely, then solve it yourself and
  include the verified answer.
- `expected.value` must be a sympy-parseable expression (for `kind=symbolic`) or a
  plain number (for `kind=numeric`).
- `expected.verification.method` is one of `sympy_equivalence`, `numeric_eval`, or
  `dimensional_only`.
- Do NOT set a `contamination` field — that is filled in later by the probe.
- Use `id` of the form `synth-<domain-short>-<4-digit-hex>` using the provided seed.
"""


_SCHEMA_EXAMPLE = """\
{
  "id": "synth-mech-0001",
  "version": 1,
  "source": {"kind": "synthetic", "origin": "solvay-generator-v0.1",
             "generated_at": "2026-04-18T00:00:00Z", "seed": 42},
  "domain": "mechanics",
  "subdomain": "rigid-body-dynamics",
  "tags": ["pendulum", "lorentz"],
  "difficulty": "intermediate",
  "statement": "...",
  "given": {"L": "1 m"},
  "find": "...",
  "expected": {
    "kind": "symbolic",
    "value": "sqrt(g/L)",
    "unit": "rad/s",
    "tolerance_rel": 0.01,
    "verification": {"method": "sympy_equivalence", "script": null}
  },
  "contamination": null,
  "notes": null
}
"""


def _extract_json(text: str) -> str:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    return stripped


def compose_problem(skeleton: GenerationSkeleton, config: BenchConfig) -> Problem:
    """Author one problem. Retries up to ``composer_max_attempts`` on parse/validation failures."""
    chat = init_chat_model(config.composer_model, temperature=0.7)
    user = (
        f"Skeleton:\n{json.dumps(skeleton.__dict__, default=list, indent=2)}\n\n"
        f"Target schema example:\n{_SCHEMA_EXAMPLE}\n"
    )

    last_error: str | None = None
    for _ in range(config.composer_max_attempts):
        messages = [
            {"role": "system", "content": _COMPOSER_SYSTEM},
            {
                "role": "user",
                "content": user
                + (f"\n\nPrevious attempt failed: {last_error}\nFix it." if last_error else ""),
            },
        ]
        response = chat.invoke(messages)
        raw = _extract_json(str(getattr(response, "content", "")))
        try:
            payload = json.loads(raw)
            return Problem.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:400]
            continue
    raise RuntimeError(
        f"Composer failed after {config.composer_max_attempts} attempts: {last_error}"
    )
