"""FINAL ANSWER convention: instruction appended to prompts and extractor.

Every benchmark profile asks the model to end its response with a single
``FINAL ANSWER: <value> <unit>`` line. Evaluation then runs against that
declared answer instead of fishing numbers out of the full transcript.
"""

from __future__ import annotations

import re

FINAL_ANSWER_INSTRUCTION = (
    "End your response with a single line in exactly this format:\n"
    "FINAL ANSWER: <value> <unit>\n"
    "Use a plain number (or a symbolic expression) and SI-style units. "
    "Write that line once, as the last line of your response."
)

_FINAL_ANSWER_RE = re.compile(
    r"\*{0,2}final answer:?\*{0,2}\s*:?\s*(?P<answer>[^\n]+)",
    re.IGNORECASE,
)


def append_final_answer_instruction(statement: str) -> str:
    """Return the problem statement with the FINAL ANSWER instruction appended."""
    return f"{statement}\n\n{FINAL_ANSWER_INSTRUCTION}"


def extract_final_answer(text: str) -> str | None:
    """Extract the declared final answer from a model response.

    Takes the LAST ``FINAL ANSWER:`` line (models sometimes restate it after
    a correction), tolerating markdown bold and case variations. Returns the
    trimmed answer text without surrounding ``$`` math delimiters, or None
    when no such line exists.
    """
    matches = _FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    answer = answer.strip("$").strip()
    return answer or None
