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

## Model compatibility

- **Azure OpenAI** (`azure_openai:gpt-5.5-codex`): Full pipeline works. Credentials in `.env`. Best option for quality.
- **Ollama local models**: Orchestrator routing works, but solver uses **toolless mode** (raw BaseChatModel, no `create_agent`) because local models generate malformed XML tool calls on complex problems. Simple problems work; complex E&M may be slow due to thinking mode.
- **Mistral Small 24B** (`ollama:mistral-small:24b`): Installed, untested. Best local candidate for reliable tool calling.

## Key architecture decisions

- **GP-as-solver clone**: A CompiledSubAgent named `"general-purpose"` reuses the solver graph to prevent deepagents from auto-injecting its own GP subagent (see `agent.py`).
- **SubagentCallLimitMiddleware**: Per-type call limits (parser:1, solver:2, etc.) in `middleware.py`. Prevents orchestrator loops.
- **Solver prepare node**: Parses ProblemSpec/ResearchBrief from HumanMessage in `solver_loop.py` because deepagents task tool only passes `messages` to CompiledSubAgents.
- **Toolless solver for Ollama**: `_is_local_model()` in `solver_loop.py` switches between `create_agent(tools=[...])` (API models) and raw `BaseChatModel` (local models).

## Benchmark

    uv run solvay-bench run --profiles bare,prompted,solvay-full --models "azure_openai:gpt-5.5-codex" --out results.jsonl
    uv run solvay-bench compare results.jsonl

Profiles: `bare` (no prompt), `prompted` (physicist prompt), `tooled` (single agent), `solvay-full` (pipeline), `solvay-noweb` (pipeline without web search).
Problems: `benchmark/problems/` — 9 hard physics problems across E&M, waves, quantum.
Grading: SymPy equivalence first, LLM-as-judge fallback.
