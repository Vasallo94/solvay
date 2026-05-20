# Solvay — run targets
# Usage:
#   make solve                        # default model, default problem, verbose
#   make solve PROBLEM="your problem"
#   make solve-qwen
#   make solve-gemma
#   make solve MODEL=anthropic:claude-sonnet-4-6
#   make solve-file FILE=my_problem.txt
#   make quantum                      # preset: quantum tunneling
#   make mechanics                    # preset: free fall
#   make em                           # preset: electromagnetic wave
#   make thermo                       # preset: Carnot engine
#   make test                         # run test suite
#   make bench                        # run benchmark

# ── Models ───────────────────────────────────────────────────
QWEN   := ollama:qwen3.6:35b-a3b-coding-mxfp8
GEMMA  := ollama:gemma4:31b
MODEL  ?= $(QWEN)

# ── Default problem ──────────────────────────────────────────
PROBLEM ?= A ball is thrown vertically upward with an initial velocity of 20 m/s. \
Ignoring air resistance, what maximum height does it reach? Use g = 9.8 m/s².

# ── Preset problems ──────────────────────────────────────────
define PROBLEM_QUANTUM
An electron with kinetic energy of 5 eV approaches a rectangular potential barrier \
of height V₀ = 8 eV and width L = 0.5 nm. Calculate: \
A) The transmission coefficient T. \
B) If 10⁶ electrons/s hit the barrier, how many tunnel through per second? \
C) How much does T change if the barrier is doubled to 1 nm?
endef

define PROBLEM_MECHANICS
A 2 kg block slides down a frictionless inclined plane of angle 30° and height 5 m. \
At the bottom it collides elastically with a 3 kg block at rest on a flat surface \
with friction coefficient μ = 0.2. How far does the 3 kg block slide before stopping?
endef

define PROBLEM_EM
A plane electromagnetic wave of frequency 100 MHz propagates in vacuum. \
Calculate: A) The wavelength. B) The magnitude of the magnetic field if \
the electric field amplitude is E₀ = 500 V/m. C) The average Poynting vector magnitude.
endef

define PROBLEM_THERMO
A Carnot engine operates between a hot reservoir at 600 K and a cold reservoir at 300 K. \
If the engine absorbs 2000 J of heat per cycle, calculate: \
A) The efficiency. B) The work done per cycle. C) The entropy change of the universe per cycle.
endef

export PROBLEM_QUANTUM PROBLEM_MECHANICS PROBLEM_EM PROBLEM_THERMO

# ── Commands ─────────────────────────────────────────────────
RUN := uv run solvay solve -v

.PHONY: solve solve-qwen solve-gemma solve-file quantum mechanics em thermo test bench help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

solve: ## Solve with MODEL and PROBLEM (both overridable)
	$(RUN) -m "$(MODEL)" "$(PROBLEM)"

solve-qwen: ## Solve with Qwen 3.6 35B
	$(RUN) -m "$(QWEN)" "$(PROBLEM)"

solve-gemma: ## Solve with Gemma4 31B
	$(RUN) -m "$(GEMMA)" "$(PROBLEM)"

solve-file: ## Solve from FILE=path.txt
	$(RUN) -m "$(MODEL)" --file "$(FILE)"

quantum: ## Preset: quantum tunneling barrier
	$(RUN) -m "$(MODEL)" "$$PROBLEM_QUANTUM"

mechanics: ## Preset: elastic collision on incline
	$(RUN) -m "$(MODEL)" "$$PROBLEM_MECHANICS"

em: ## Preset: electromagnetic wave
	$(RUN) -m "$(MODEL)" "$$PROBLEM_EM"

thermo: ## Preset: Carnot engine
	$(RUN) -m "$(MODEL)" "$$PROBLEM_THERMO"

test: ## Run test suite
	uv run pytest tests/ -q

bench: ## Run benchmark suite
	uv run solvay bench
