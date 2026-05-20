# Solvay -- AI-Dev Conventions

## Language rule (non-negotiable)

ALL code, identifiers, comments, docstrings, prompts, schemas, commit messages,
branch names, documentation, and log strings are written in **English**.

## Project overview

Multi-agent physics problem solver built on deepagents. See
`docs/superpowers/specs/2026-04-15-solvay-design.md` for the full design spec.

## How to run

    uv sync                          # install deps
    uv run solvay solve "..."        # solve a problem
    uv run pytest                    # run tests
    uv run solvay bench              # run benchmark

## How to add a subagent

1. Define response schema in `src/solvay/schemas.py`.
2. Write system prompt in `src/solvay/prompts/<role>.md`.
3. Create subagent factory in `src/solvay/subagents/<role>.py`.
4. Register in `src/solvay/agent.py` inside `create_solvay_agent()`.
5. Add tests.

## How to add a tool

1. Write function with typed signature and docstring in `src/solvay/tools/`.
2. Register in `src/solvay/agent.py` (add to the relevant subagent's `tools` list).
3. Add tests in `tests/test_tools.py`.

## Working with deepagents

Canonical docs: https://docs.langchain.com/oss/python/deepagents/overview
For API lookups prefer Context7 over web search.
Never rely on memory for deepagents API; always verify against current docs.

## Interactive mode

    uv run solvay chat               # interactive REPL with streaming
