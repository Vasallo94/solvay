"""Main agent factory: wires orchestrator, subagents, tools, and middleware."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from deepagents import SubAgent, create_deep_agent
from tavily import TavilyClient

from solvay.config import SolvayConfig
from solvay.subagents import load_prompt
from solvay.subagents.consolidator import create_consolidator_subagent
from solvay.subagents.parser import create_parser_subagent
from solvay.subagents.peer_reviewer import create_peer_reviewer_subagent
from solvay.subagents.researcher import create_researcher_subagent
from solvay.subagents.solver_loop import create_solver_subagent
from solvay.subagents.verifier import create_verifier_subagent
from solvay.tools.url_fetch import url_fetch


def _create_web_search_tool() -> Any:
    """Create the Tavily web search tool function."""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def web_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news"] = "general",
        include_raw_content: bool = False,
    ) -> Any:
        """Search the web using Tavily.

        Args:
            query: Search query string.
            max_results: Maximum number of results.
            topic: Search topic category.
            include_raw_content: Whether to include raw page content.

        Returns:
            Search results from Tavily.
        """
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return web_search


def create_solvay_agent(
    config: SolvayConfig | None = None,
) -> Any:
    """Create the fully wired Solvay orchestrator agent.

    Args:
        config: Solvay configuration. Uses defaults if None.

    Returns:
        A compiled deepagents agent ready to invoke.
    """
    if config is None:
        config = SolvayConfig()

    web_search = _create_web_search_tool()

    parser = create_parser_subagent(config)
    researcher = create_researcher_subagent(config, web_search, url_fetch)
    solver = create_solver_subagent(config)
    verifier = create_verifier_subagent(config)
    peer_reviewer = create_peer_reviewer_subagent(config, web_search)
    consolidator = create_consolidator_subagent(config)

    agent = create_deep_agent(
        model=config.model_for("orchestrator"),
        tools=[web_search],
        system_prompt=load_prompt("orchestrator"),
        subagents=[
            cast(SubAgent, parser),
            cast(SubAgent, researcher),
            solver,
            cast(SubAgent, verifier),
            cast(SubAgent, peer_reviewer),
            cast(SubAgent, consolidator),
        ],
    )

    return agent
