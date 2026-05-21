# Solvay — run targets
# Usage:
#   make solve P="your problem here"
#   make solve P="your problem" M=ollama:gemma4:31b
#   make solve-file F=my_problem.txt
#   make solve-qwen P="your problem"
#   make solve-gemma P="your problem"
#   make test
#   make bench

# ── Models ───────────────────────────────────────────────────
QWEN   := ollama:qwen3.6:35b-a3b-coding-mxfp8
GEMMA  := ollama:gemma4:31b
VERTEX := vertexai:claude-sonnet-4-6
M      ?= $(QWEN)

# ── Commands ─────────────────────────────────────────────────
RUN := uv run solvay solve -v

.PHONY: solve solve-qwen solve-gemma solve-file test bench help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

solve: ## Solve P="problem" with model M
	@test -n "$(P)" || (echo "Usage: make solve P=\"your problem\"" && exit 1)
	$(RUN) -m "$(M)" "$(P)"

solve-qwen: ## Solve P="problem" with Qwen 3.6 35B
	@test -n "$(P)" || (echo "Usage: make solve-qwen P=\"your problem\"" && exit 1)
	$(RUN) -m "$(QWEN)" "$(P)"

solve-gemma: ## Solve P="problem" with Gemma4 31B
	@test -n "$(P)" || (echo "Usage: make solve-gemma P=\"your problem\"" && exit 1)
	$(RUN) -m "$(GEMMA)" "$(P)"

solve-vertex: ## Solve P="problem" with Claude Sonnet via Vertex AI
	@test -n "$(P)" || (echo "Usage: make solve-vertex P=\"your problem\"" && exit 1)
	$(RUN) -m "$(VERTEX)" "$(P)"

solve-file: ## Solve from F=path.txt with model M
	@test -n "$(F)" || (echo "Usage: make solve-file F=problem.txt" && exit 1)
	$(RUN) -m "$(M)" --file "$(F)"

test: ## Run test suite
	uv run pytest tests/ -q

bench: ## Run benchmark suite
	uv run solvay bench
