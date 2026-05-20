"""End-to-end smoke test with a real LLM. Opt-in via RUN_LIVE_TESTS=1."""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("RUN_LIVE_TESTS", "0") == "1"


@pytest.mark.skipif(not LIVE, reason="Set RUN_LIVE_TESTS=1 to run")
def test_simple_mechanics_problem() -> None:
    """Solve a simple inclined plane problem end-to-end."""
    from solvay.agent import create_solvay_agent
    from solvay.config import SolvayConfig

    config = SolvayConfig()
    agent = create_solvay_agent(config)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A block of mass 2 kg slides down a frictionless incline "
                        "of angle 30 degrees. Find the acceleration of the block. "
                        "Express your answer in m/s^2."
                    ),
                }
            ]
        }
    )

    final_message = result["messages"][-1].content
    assert isinstance(final_message, str)
    assert len(final_message) > 50

    # Answer should be in the right ballpark (g*sin(30) is approx 4.9)
    assert "4.9" in final_message or "4.905" in final_message
