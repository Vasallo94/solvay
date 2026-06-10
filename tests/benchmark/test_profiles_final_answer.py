"""Tests for the FINAL ANSWER convention and token aggregation in profiles."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from solvay.benchmark.answer_format import FINAL_ANSWER_INSTRUCTION
from solvay.benchmark.config import BenchConfig
from solvay.benchmark.profiles import aggregate_tokens, get_profile
from tests.benchmark.test_profile_bare import _problem


def _ai_message(content: str, input_tokens: int = 0, output_tokens: int = 0) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_token_details": {"cache_read": 0},
    }
    return msg


class TestAggregateTokens:
    def test_sums_usage_across_messages(self) -> None:
        messages = [_ai_message("a", 10, 5), _ai_message("b", 20, 7)]
        totals = aggregate_tokens(messages)
        assert totals == {"input": 30, "output": 12, "cache_read": 0}

    def test_ignores_messages_without_dict_usage(self) -> None:
        plain = MagicMock()
        plain.usage_metadata = None
        totals = aggregate_tokens([plain, _ai_message("a", 1, 1)])
        assert totals == {"input": 1, "output": 1, "cache_read": 0}


class TestBareProfileFinalAnswer:
    def test_appends_instruction_and_extracts_answer(self) -> None:
        fake_chat = MagicMock()
        response = MagicMock()
        response.content = "Work...\nFINAL ANSWER: 2"
        response.usage_metadata = {"input_tokens": 3, "output_tokens": 4}
        fake_chat.invoke.return_value = response
        with patch("solvay.benchmark.profiles.bare.init_chat_model", return_value=fake_chat):
            result = get_profile("bare").runner(
                _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
            )
        sent = fake_chat.invoke.call_args[0][0][0]["content"]
        assert FINAL_ANSWER_INSTRUCTION in sent
        assert result.answer_extracted == "2"
        assert result.tokens["input"] == 3


class TestSolvayFullProfileFinalAnswer:
    def test_extracts_answer_and_aggregates_tokens(self) -> None:
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {
            "messages": [
                _ai_message("delegating...", 100, 20),
                _ai_message("Method...\nFINAL ANSWER: 4.905 m/s^2", 50, 30),
            ]
        }
        with patch(
            "solvay.benchmark.profiles.solvay_full.create_solvay_agent",
            return_value=fake_agent,
        ):
            result = get_profile("solvay-full").runner(
                _problem(), "anthropic:claude-sonnet-4-6", BenchConfig()
            )
        sent = fake_agent.invoke.call_args[0][0]["messages"][0]["content"]
        assert FINAL_ANSWER_INSTRUCTION in sent
        assert result.answer_extracted == "4.905 m/s^2"
        assert result.tokens == {"input": 150, "output": 50, "cache_read": 0}
