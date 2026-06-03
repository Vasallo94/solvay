"""Tests for the SubagentCallLimitMiddleware."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage

from solvay.middleware import SubagentCallLimitMiddleware


def _make_request(subagent_type: str, call_id: str = "call_1") -> MagicMock:
    req = MagicMock()
    req.tool_call = {
        "id": call_id,
        "name": "task",
        "args": {"subagent_type": subagent_type, "description": "test"},
    }
    return req


def _make_non_task_request(tool_name: str = "web_search") -> MagicMock:
    req = MagicMock()
    req.tool_call = {"id": "call_1", "name": tool_name, "args": {"query": "test"}}
    return req


class TestSubagentCallLimitMiddleware:
    def test_allows_first_parser_call(self) -> None:
        mw = SubagentCallLimitMiddleware()
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))
        req = _make_request("parser", "c1")

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result.content == "ok"

    def test_blocks_second_parser_call(self) -> None:
        mw = SubagentCallLimitMiddleware()
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))

        mw.wrap_tool_call(_make_request("parser", "c1"), handler)
        result = mw.wrap_tool_call(_make_request("parser", "c2"), handler)

        assert isinstance(result, ToolMessage)
        assert "REJECTED" in result.content
        assert "parser" in result.content
        assert handler.call_count == 1

    def test_allows_two_solver_calls(self) -> None:
        mw = SubagentCallLimitMiddleware()
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))

        mw.wrap_tool_call(_make_request("solver", "c1"), handler)
        mw.wrap_tool_call(_make_request("solver", "c2"), handler)

        assert handler.call_count == 2

    def test_blocks_third_solver_call(self) -> None:
        mw = SubagentCallLimitMiddleware()
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))

        mw.wrap_tool_call(_make_request("solver", "c1"), handler)
        mw.wrap_tool_call(_make_request("solver", "c2"), handler)
        result = mw.wrap_tool_call(_make_request("solver", "c3"), handler)

        assert "REJECTED" in result.content
        assert handler.call_count == 2

    def test_passes_through_non_task_tools(self) -> None:
        mw = SubagentCallLimitMiddleware()
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))
        req = _make_non_task_request("web_search")

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result.content == "ok"

    def test_custom_limits(self) -> None:
        mw = SubagentCallLimitMiddleware(limits={"parser": 3})
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))

        for i in range(3):
            mw.wrap_tool_call(_make_request("parser", f"c{i}"), handler)

        assert handler.call_count == 3

        result = mw.wrap_tool_call(_make_request("parser", "c4"), handler)
        assert "REJECTED" in result.content

    def test_unknown_subagent_type_not_limited(self) -> None:
        mw = SubagentCallLimitMiddleware()
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))

        for i in range(10):
            mw.wrap_tool_call(_make_request("unknown_type", f"c{i}"), handler)

        assert handler.call_count == 10
