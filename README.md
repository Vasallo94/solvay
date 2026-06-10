# Solvay

Multi-agent physics problem solver built on [deepagents](https://github.com/hwchase17/deepagents) and [LangGraph](https://github.com/langchain-ai/langgraph).

An orchestrator delegates each physics problem to a pipeline of specialized sub-agents — **parser**, **researcher**, **solver**, **consolidator** — that share a virtual filesystem (the "lab notebook") and a toolkit of sandboxed `python_exec`, symbolic dimension checks, and web search. The solver runs a conversational review loop: it submits each draft to a `request_review` tool that consults two internal critics (**verifier** and **peer reviewer**) in parallel under a code-enforced iteration budget.

## Status

Early prototype. The end-to-end pipeline runs against Anthropic models; local Ollama models currently complete the run as a single LLM call without orchestrating sub-agents (see `FUTURE.md`).

## Requirements

- Python **3.14**
- [uv](https://github.com/astral-sh/uv) for dependency management
- An Anthropic API key (recommended) or a local Ollama install
- Optional: Tavily API key for web search, LangSmith account for tracing

## Setup

```bash
uv sync --extra dev
cp .env.example .env
# edit .env with your keys
```

`.env` is auto-loaded by the CLI (see `src/solvay/cli.py`) before any LangChain import, so keys flow in without manual `export`.

## Usage

```bash
# Solve a one-liner
uv run solvay solve "A 2 kg block slides down a 30° incline with μ=0.1. Find its speed after 3 m."

# Read the problem from a file, see the full lab notebook, dump a trace
uv run solvay solve --file problem.md --show-notebook --trace trace.json

# Pick a different model (any string accepted by langchain.chat_models.init_chat_model)
uv run solvay solve --model "openai:gpt-4o" "..."
uv run solvay solve --model "ollama:qwen3.5"  "..."

# Run the benchmark suite through the matrix runner
uv run solvay bench --problems benchmark/problems --profiles solvay-noweb --models "ollama:qwen3.5" --out results.jsonl
```

For a first local Ollama smoke test, start Ollama, pull the model, then run one
small no-web benchmark profile:

```bash
ollama pull qwen3.5
uv run solvay bench \
  --problems benchmark/problems \
  --profiles solvay-noweb \
  --models "ollama:qwen3.5" \
  --out benchmark/runs/ollama-smoke.jsonl
```

## Observability

Set the LangSmith vars in `.env` (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=solvay`) and every run shows up in the LangSmith UI with the full sub-agent tree, tool calls, and messages. EU-region accounts need `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.

## Project layout

```
src/solvay/
├── agent.py              # deepagents orchestrator + middleware wiring
├── cli.py                # Typer CLI (solve, bench)
├── config.py             # SolvayConfig, PersistenceConfig, model routing
├── middleware/           # notebook injection, cross-run persistence
├── prompts/              # markdown system prompts for each role
├── subagents/            # parser, researcher, solver (+ critics), ...
└── tools/                # python_exec, dimensional check, url_fetch
tests/                    # pytest suite (ruff + mypy --strict clean)
benchmark/                # curated physics problems + runner
docs/superpowers/         # design spec and implementation plan
```

## Development

```bash
uv run pytest           # tests
uv run ruff check .     # lint
uv run mypy src tests   # type-check (strict)
```

Pre-commit hooks run ruff, mypy, and detect-secrets on every commit (`uv run pre-commit install`).

## License

Private repository — all rights reserved for now.
