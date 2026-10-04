# Typical workflows. `make` (or `make help`) lists the targets.
# Variables can be set on the command line, e.g.:
#   make parse Q="blue velvet sofa"
#   make dev BACKEND=ollaya MODEL=jeb:4b
#   make eval BACKEND=ollama MODEL=tev1 SCHEME=embedded
# Run only one model on the GPU at a time (see `make status`).

SHELL := /bin/bash
.DEFAULT_GOAL := help

BACKEND ?= ollama
MODEL   ?=
SCHEME  ?= embedded
SPLIT   ?= dev
LIMIT   ?=
OVERWRITE ?=
STRICT  ?=
Q       ?= cheap grey oak coffee table with storage
V       ?= -v

PY      := uv run python
MODELS_DIR    ?= $(HOME)/sidm-models
OLLAYA        := $(MODELS_DIR)/ollaya/bin/ollaya
OLLAYA_ENV    := OLLAYA_MODELS=$(MODELS_DIR)/ollaya-models OLLAYA_HOST=127.0.0.1:11435

# Resolved lazily (only targets that use them pay for the Python call).
MODEL_RESOLVED = $(or $(MODEL),$(shell $(PY) -c "from sidm.ollama_client import DEFAULT_MODELS; print(DEFAULT_MODELS['$(BACKEND)'])" 2>/dev/null))
RUN            = $(SCHEME)@$(BACKEND):$(MODEL_RESOLVED)
PRED           = $(shell $(PY) -c "from sidm.evaluate import pred_path; print(pred_path('$(1)', '$(MODEL_RESOLVED)', '$(SCHEME)', '$(BACKEND)'))" 2>/dev/null)
LIMIT_ARG      = $(if $(LIMIT),--limit $(LIMIT))

.PHONY: help doctor runs setup schema parse data preview run dev eval tune report bench results mcnemar example-doc \
        ollama-pull ollama-rm ollama-unload mlx-convert mlx-serve mlx-stop ollaya-serve ollaya-pull ollaya-rm ollaya-stop decider-setup decider-serve decider-stop status

help: ## List targets and variables
	@echo "Usage: make <target> [BACKEND=ollama|mlx|ollaya|decider|jev] [MODEL=...] [SCHEME=embedded|router] [SPLIT=dev|eval] [LIMIT=N] [OVERWRITE=1] [Q=\"query\"]"
	@echo "New here? Start with: make doctor, make runs; shipped results: results/results_eval.md, results/mcnemar_eval.txt"
	@awk 'BEGIN {FS = ":.*?## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-z0-9-]+:.*?## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Start here
doctor: ## What this machine has: servers, models, keys, data; which runs can be re-run (fixes included)
	@$(PY) -m sidm.doctor

runs: ## Status of every run: stored dev/eval predictions, decoding settings, last update
	@$(PY) -m sidm.runs

##@ Setup
setup: ## Install the project environment (uv sync)
	uv sync

schema: ## Rebuild data/schema.json and run the vocabulary checks
	$(PY) -m sidm.schema

##@ Single query
parse: ## Parse one query: Q="..." [BACKEND MODEL SCHEME V=-v|-vv|-vvv|--json]
	uv run sidm $(V) --backend $(BACKEND) $(if $(MODEL),--model $(MODEL)) --scheme $(SCHEME) "$(Q)"

##@ Dataset (needs qwen3:14b in Ollama; don't run alongside a GPU eval)
data: ## Generate / resume the 1,100-row dataset (data/eval_raw.jsonl)
	$(PY) -u -m sidm.gen_dataset --n 1100 --out data/eval_raw.jsonl --workers 2

preview: ## Write the readable dump data/preview.txt
	$(PY) -m sidm.preview

##@ Evaluation of one run (<SCHEME>@<BACKEND>:<MODEL>; resumable)
run: ## Run/resume the parser on SPLIT and write that run's table [LIMIT=N] [OVERWRITE=1 = re-run from scratch]
	$(PY) -u -m sidm.results_table --runs $(RUN) --run --split $(SPLIT) $(LIMIT_ARG) $(if $(OVERWRITE),--overwrite) \
	  --md results/results_$(SPLIT)_$(subst :,-,$(subst @,_,$(RUN))).md

dev: ## `run` on the 100 dev queries (resumes; OVERWRITE=1 re-runs from scratch)
	@$(MAKE) --no-print-directory run SPLIT=dev

eval: ## `run` on the 1,000 eval queries (resumes; OVERWRITE=1 re-runs from scratch)
	@$(MAKE) --no-print-directory run SPLIT=eval

tune: ## Tune this run's decoding on its dev predictions -> results/tuned_settings.json
	$(PY) -m sidm.evaluate tune --apply --pred "$(call PRED,dev)"

report: ## Detailed report (per-attribute, confusions, worst queries) for SPLIT
	$(PY) -m sidm.evaluate report --pred "$(call PRED,$(SPLIT))" \
	  --md "results/report_$(basename $(notdir $(call PRED,$(SPLIT)))).md"

bench: ## Clean latency benchmark: first LIMIT (default 30) dev queries, nothing else running
	$(PY) -u -m sidm.evaluate run --split dev --limit $(or $(LIMIT),30) --backend $(BACKEND) \
	  --model $(MODEL_RESOLVED) --scheme $(SCHEME) --out "results/bench/$(BACKEND)-$(subst :,-,$(MODEL_RESOLVED)).jsonl"

##@ Results (from saved predictions; no model calls)
results: ## Regenerate tables results_{eval,dev}.{md,json}, mcnemar_*.txt, report_eval_*.md (runs: src/sidm/runs.py) [STRICT=1]
	scripts/make_results.sh $(if $(STRICT),--strict)

mcnemar: ## Print the pairwise McNemar tests of results/results_SPLIT.json [METRIC=category|full] [SIG=1]
	@$(PY) -m sidm.results_table --show-tests results/results_$(SPLIT).json $(if $(METRIC),--metric $(METRIC)) $(if $(SIG),--significant)

example-doc: ## Rebuild docs/example_request.md from cached responses
	$(PY) -m sidm.example_doc

##@ Model servers and memory (one model on the GPU at a time)
ollama-pull: ## Pull MODEL into Ollama
	ollama pull $(MODEL_RESOLVED)

ollama-rm: ## Delete MODEL from Ollama's disk store (e.g. MODEL=qwen3:14b; re-pull with ollama-pull)
	@test -n "$(MODEL)" || { echo "set MODEL=<name>, e.g. make ollama-rm MODEL=qwen3:14b"; exit 1; }
	@df -h ~ | tail -1
	ollama rm $(MODEL)
	@df -h ~ | tail -1

ollama-unload: ## Unload every model loaded in Ollama
	@for m in $$(ollama ps | awk 'NR>1 {print $$1}'); do ollama stop "$$m"; echo "unloaded $$m"; done

mlx-convert: ## Prepare nimble's 8-bit MLX model (download, merge, quantize, verify)
	uv run --project mlx_backend python mlx_backend/convert_model.py all --q-bits 8

mlx-serve: ## Serve nimble's MLX ParallelScorer on :11500 (foreground; unloads Ollama's nimble)
	mlx_backend/serve.sh --mode cached_serial

mlx-stop: ## Stop the MLX server
	-pkill -f mlx_backend/server.py

ollaya-serve: ## Serve Ollaya on :11435 (foreground; unloads Ollama's models)
	scripts/ollaya_serve.sh

ollaya-pull: ## Pull MODEL into Ollaya
	$(OLLAYA_ENV) $(OLLAYA) pull $(MODEL_RESOLVED)

ollaya-rm: ## Delete MODEL from Ollaya's store (e.g. MODEL=von)
	@test -n "$(MODEL)" || { echo "set MODEL=<name>, e.g. make ollaya-rm MODEL=von"; exit 1; }
	$(OLLAYA_ENV) $(OLLAYA) rm $(MODEL)

ollaya-stop: ## Stop the Ollaya server (unloads its models)
	-$(OLLAYA_ENV) $(OLLAYA) stop

decider-setup: ## Create the Strands Decider environment (decider_backend/, Python 3.12, MLX)
	cd decider_backend && uv sync

decider-serve: ## Serve Strands Decider on :11600 (foreground; unloads Ollama/Ollaya; DEVICE=mps fallback)
	decider_backend/serve.sh

decider-stop: ## Stop the Strands Decider server
	-pkill -f "strands-decider serve"

status: ## Show which models are loaded where, and free disk
	@echo "== Ollama";  ollama ps 2>/dev/null || echo "not running"
	@echo "== Ollaya";  $(OLLAYA_ENV) $(OLLAYA) ps 2>/dev/null || echo "not running"
	@echo "== MLX";     curl -s -m 2 localhost:11500/health || echo "not running"; echo
	@echo "== Strands Decider"; curl -s -m 2 -o /dev/null -w "running (HTTP %{http_code})\n" localhost:11600/docs || echo "not running"
	@echo "== Disk";    df -h ~ | tail -1
