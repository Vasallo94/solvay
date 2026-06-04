"""Two-stage physics answer grader: SymPy equivalence check with LLM-judge fallback."""

from __future__ import annotations

import json
import re

import sympy
from langchain.chat_models import init_chat_model

from solvay.benchmark.config import BenchConfig
from solvay.benchmark.schema import Expected, Problem

# Regex patterns used for extraction
_NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_SYMBOLIC_PATTERN = re.compile(
    r"[a-zA-Z_]\w*(?:\s*[\*\+\-/\^]\s*[a-zA-Z_\d\.\(\)]*)+"
)


def sympy_grade(answer_raw: str, expected: Expected) -> bool | None:
    """Attempt to grade answer_raw against expected using SymPy.

    Returns:
        True  – answer is correct.
        False – answer is definitively wrong.
        None  – could not determine (no parseable expression found, or
                expected itself could not be parsed).
    """
    if expected.kind == "numeric":
        return _grade_numeric(answer_raw, expected)
    elif expected.kind == "symbolic":
        return _grade_symbolic(answer_raw, expected)
    return None


def _grade_numeric(answer_raw: str, expected: Expected) -> bool | None:
    """Grade a numeric answer."""
    # Parse expected value
    try:
        target = float(sympy.sympify(expected.value).evalf())
    except Exception:
        return None

    # Extract all floating-point numbers from the answer
    numbers = _NUMERIC_PATTERN.findall(answer_raw)
    if not numbers:
        return None

    tol = expected.tolerance_rel
    for num_str in numbers:
        try:
            candidate = float(num_str)
        except ValueError:
            continue
        # Relative tolerance check (handle zero target edge-case)
        if target == 0.0:
            if abs(candidate) <= tol:
                return True
        else:
            if abs(candidate - target) / abs(target) <= tol:
                return True

    return False


def _grade_symbolic(answer_raw: str, expected: Expected) -> bool | None:
    """Grade a symbolic answer."""
    # Parse expected expression
    try:
        expected_expr = sympy.sympify(expected.value)
    except Exception:
        return None

    candidates: list[str] = []

    # Collect symbolic expressions
    for match in _SYMBOLIC_PATTERN.finditer(answer_raw):
        candidates.append(match.group())

    # Also try plain numbers in case the symbolic answer evaluates to a constant
    for match in _NUMERIC_PATTERN.finditer(answer_raw):
        candidates.append(match.group())

    if not candidates:
        return None

    any_parsed = False
    for raw_candidate in candidates:
        # Strip trailing punctuation that the regex may have captured
        raw_candidate = raw_candidate.rstrip(".,;:!?)")
        # Replace caret power notation with Python notation
        normalised = raw_candidate.replace("^", "**")
        try:
            candidate_expr = sympy.sympify(normalised)
            any_parsed = True
        except Exception:
            continue

        try:
            diff = sympy.simplify(candidate_expr - expected_expr)
            if diff == 0:
                return True
        except Exception:
            continue

    if not any_parsed:
        return None

    return False


def llm_grade(
    answer_raw: str,
    expected: Expected,
    problem: Problem,
    config: BenchConfig,
) -> bool:
    """Grade answer_raw using an LLM as judge.

    Returns True if the LLM decides the answer is correct, False otherwise.
    Unparseable LLM responses are treated as incorrect (False).
    """
    grader_model = getattr(config, "grader_model", "fake-model")
    llm = init_chat_model(grader_model)

    unit_str = expected.unit or ""
    prompt = (
        "Compare this physics answer to the expected answer.\n\n"
        f"Problem: {problem.statement}\n"
        f"Student answer: {answer_raw}\n"
        f"Expected: {expected.value} {unit_str}\n\n"
        'Are they mathematically equivalent? Return JSON: {"correct": true/false, "reason": "..."}'
    )

    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        return bool(data["correct"])
    except Exception:
        return False


def evaluate_correct(
    answer_raw: str,
    expected: Expected,
    problem: Problem,
    config: BenchConfig,
) -> tuple[bool, str]:
    """Grade an answer using SymPy first, falling back to LLM-judge.

    Returns:
        (correct, grading_method) where grading_method is "sympy" or "llm-judge".
    """
    sympy_result = sympy_grade(answer_raw, expected)
    if sympy_result is not None:
        return (sympy_result, "sympy")

    llm_result = llm_grade(answer_raw, expected, problem, config)
    return (llm_result, "llm-judge")
