"""Deterministic behavior fingerprint for a Solvay configuration.

Two runs with the same fingerprint are considered "the same Solvay" for
regression-comparison purposes. The fingerprint changes if and only if
something that affects model output changes: prompt text, subagent roster,
model-by-role mapping, or tool module identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from solvay.config import Role, SolvayConfig

_PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"
_SUBAGENT_ROSTER: tuple[Role, ...] = (
    "orchestrator",
    "parser",
    "researcher",
    "solver",
    "verifier",
    "peer_reviewer",
    "consolidator",
)
_TOOL_MODULES: tuple[str, ...] = (
    "solvay.tools.dimensional",
    "solvay.tools.python_exec",
    "solvay.tools.url_fetch",
)


def _hash_prompts() -> str:
    h = hashlib.sha256()
    for md in sorted(_PROMPTS_DIR.glob("*.md")):
        h.update(md.name.encode("utf-8"))
        h.update(b"\0")
        h.update(md.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _hash_tools() -> str:
    h = hashlib.sha256()
    for mod in _TOOL_MODULES:
        name = mod.rsplit(".", 1)[-1]
        path = Path(__file__).resolve().parent.parent / "tools" / f"{name}.py"
        if path.exists():
            h.update(mod.encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def compute_fingerprint(config: SolvayConfig) -> str:
    """Return a sha256 fingerprint of the runtime behavior surface."""
    model_map = {role: config.model_for(role) for role in _SUBAGENT_ROSTER}
    payload = {
        "prompts_hash": _hash_prompts(),
        "tools_hash": _hash_tools(),
        "subagents": list(_SUBAGENT_ROSTER),
        "model_map": model_map,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()
