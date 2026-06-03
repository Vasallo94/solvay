"""Custom middleware for the Solvay orchestrator."""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

SUBAGENT_CALL_LIMITS: dict[str, int] = {
    "parser": 1,
    "researcher": 2,
    "solver": 2,
    "peer_reviewer": 1,
    "consolidator": 1,
    "general-purpose": 2,
}


class SubagentCallLimitMiddleware(AgentMiddleware):  # type: ignore[type-arg]
    """Enforce per-subagent-type call limits on the ``task`` tool.

    Tracks how many times each ``subagent_type`` has been dispatched
    and short-circuits with an error message when the limit is exceeded.
    """

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        super().__init__()
        self._limits = limits or SUBAGENT_CALL_LIMITS
        self._counts: dict[str, int] = {}

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call.get("name")
        if tool_name != "task":
            return handler(request)

        subagent_type = request.tool_call.get("args", {}).get("subagent_type", "")
        limit = self._limits.get(subagent_type)

        if limit is not None:
            current = self._counts.get(subagent_type, 0)
            if current >= limit:
                return ToolMessage(
                    content=(
                        f"REJECTED: subagent '{subagent_type}' has already been called "
                        f"{current} time(s) (limit: {limit}). Move to the next step."
                    ),
                    tool_call_id=request.tool_call["id"],
                    name="task",
                    status="error",
                )
            self._counts[subagent_type] = current + 1

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call.get("name")
        if tool_name != "task":
            return await handler(request)

        subagent_type = request.tool_call.get("args", {}).get("subagent_type", "")
        limit = self._limits.get(subagent_type)

        if limit is not None:
            current = self._counts.get(subagent_type, 0)
            if current >= limit:
                return ToolMessage(
                    content=(
                        f"REJECTED: subagent '{subagent_type}' has already been called "
                        f"{current} time(s) (limit: {limit}). Move to the next step."
                    ),
                    tool_call_id=request.tool_call["id"],
                    name="task",
                    status="error",
                )
            self._counts[subagent_type] = current + 1

        return await handler(request)
