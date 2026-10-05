# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
This is an experiment, not a product. It parses a furniture search query into a **category** (a node in a product-type ontology), **filters** (categorical attribute=value pairs) and **residual words**. It uses a Jev-style decision model (`nimble`, 9B) through Ollama ≥ 0.35's `POST /v1/systemone`, sending **one request per query**. A synthetic eval set is generated with `qwen3:14b`. `README.md` has the current results; `backlog.md` has findings and next experiments.

## Commands
`make doctor` (what this machine has, which runs can be re-run, with fixes) and `make runs` (stored predictions per run) are the entry points. `make help` lists the typical workflows (`parse`, `dev`, `eval`, `tune`, `report`, `bench`, `sweep`, `results`, the server start/stop targets and `status`). They take the variables `BACKEND`, `MODEL`, `SCHEME`, `SPLIT`, `LIMIT` and `Q`. The underlying commands:
The project is stdlib-only (Python ≥ 3.9) and has no tests or linter. Run everything through uv from the repo root.
```bash
uv run sidm -v "cheap grey oak coffee table with storage"      # one query; -v/-vv/-vvv, --json, --scheme, --untuned
uv run python -m sidm.gen_dataset --n 1100 --out data/eval_raw.jsonl --workers 2   # resumable
uv run python -m sidm.evaluate run --split dev --scheme embedded --out results/dev_nimble_embedded.jsonl
uv run python -m sidm.evaluate report --pred results/eval_nimble_embedded.jsonl --md results/report_embedded.md
uv run python -m sidm.evaluate tune --pred results/dev_nimble_embedded.jsonl     # sweep the no-category threshold (dev only)
uv run python -m sidm.evaluate compare --pred results/eval_nimble_router.jsonl results/eval_nimble_embedded.jsonl
scripts/make_results.sh                       # regenerate results/results_{eval,dev}.{md,json} (edit its run list)
uv run python -m sidm.results_table --runs embedded@ollama:nimble embedded@mlx:nimble [--run]   # ad-hoc table
uv run --project mlx_backend python mlx_backend/convert_model.py all --q-bits 8     # one-time MLX model prep
mlx_backend/serve.sh                                                                # MLX backend server
uv run python -m sidm.schema                  # rebuild data/schema.json and run the vocabulary collision asserts
```

## Architecture (`src/sidm/`)
- **`schema.py`** is the single source of truth for the ontology (L1 → L2 → optional L3), the 12 attributes with surface forms, attribute applicability per L1/L2, and the residual vocabulary.
  - Import-time asserts enforce that no synonym is shared by two nodes and that residual phrases don't collide with attribute or category words.
  - Every other module reads from here.
- **`gen_dataset.py`**:
  - Samples a gold intent, then has qwen3 verbalize it as a query whose spans label *every* word.
  - `validate()` checks:
    - spans cover every word, in order;
    - the category and filter text map back to the gold labels;
    - width numbers fall in their bucket;
    - residual spans contain no category or attribute term (`check_residual_leaks`).
  - A few safe cases are relabeled automatically; any other failure is retried with the error fed back to qwen3.
  - Rows are appended to `data/eval_raw.jsonl`, and finished ids are skipped on restart.
  - **Word-labeling rule:** a connecting word belongs to the filter or category phrase it sits in ("**with** storage", "chest **of** drawers"); otherwise it is residual.
- **`parser.py`** builds the request and decodes the answers. Every question is a `choice`:
  - `cat_<group>` ×6: the group's L2 and L3 nodes flattened into one list.
  - `filter_<attr>` ×12: `not_specified` plus the values.
  - `word_<i>`: one per word, choosing category / filter / residual.
  - **Two ablatable category schemes**, defined by `CategoryScheme` (`SCHEMES`, `DEFAULT_SCHEME`):
    - `router` adds the top-level question `cat_L1`.
    - `embedded` (the default) drops `cat_L1` and gives every group question an `other_product` escape.
    - Only `group_question()` and `none_signal()` behave differently between the two; everything else is shared.
  - **Decoding:** `TUNED` (dev-tuned) vs `UNTUNED` (plain argmax).
    - Category: the most confident group answer (`category_candidates`), with "no category" when `none_signal > none_thr`.
    - Filters: a value counts only at probability ≥ 0.95.
    - Word roles: the `category` probability is weighted ×8.
- **`evaluate.py`**: `run` writes `results/<split>_<model>_<scheme>.jsonl` with the **raw answers**. `report`, `tune`, `compare` and `results_table.py` re-decode them through `load_preds()`, so changing decoding never needs a nimble re-run.
  - Files without a `scheme` field are treated as `router`.
- **`results_table.py`**: the main table. For each metric it gives the measure, what it is counted over, the counts and a 95% Wilson interval, plus a paired McNemar test between schemes.
- **`cli.py`**: the `sidm` console script declared in `pyproject.toml`.
- **`mlx_backend/`** is a separate **Python 3.12** uv project (mlx, mlx-lm, transformers), not part of the 3.9 package.
  - `convert_model.py` is a staged, resumable converter: adapter → bf16 merge → MLX quantization with `lm_head` kept unquantized, since `ParallelScorer` requires it.
  - `server.py` exposes nimble's `ParallelScorer` with the same `/v1/systemone` contract, on :11500.
  - The main package selects it with `backend="mlx"` (`ollama_client.BACKENDS`).
  - Prediction files: `results/<split>_mlx-<model>_<scheme>.jsonl`.
  - Per-backend tuned decoding lives in `parser.TUNED_BY_BACKEND`.
  - Run only one local backend at a time: `serve.sh` runs `ollama stop nimble`, because memory is tight.
- **The `decider` backend** is AWS Strands Labs' Strands Decider on :11600, served by `decider_backend/serve.sh` (`make decider-setup` / `decider-serve`). It uses its own Python 3.12 env installed from GitHub at a pinned commit, with weights in `~/sidm-models/hf-home`. It encodes the state once and only adds each question's suffix.
- **The `ollaya` backend** is the Ollaya server on :11435 (`scripts/ollaya_serve.sh`). The binary is `~/sidm-models/ollaya/bin/ollaya`, and models go in `OLLAYA_MODELS=~/sidm-models/ollaya-models`. Pull models before use; Ollaya never pulls implicitly.
- **Oversized prompts are split automatically.** `ollama_client.system_one_split` handles them, e.g. tev1's 2k context in Ollama.
- **Per-model tuned decoding** lives in `results/tuned_settings.json` (keys `"<backend>:<model>"`), written by `evaluate tune --apply` on a dev run.
- **The `jev` backend** is TypeSafe's hosted `/v1/systemone`.
  - It needs `TYPESAFE_API_KEY`; the base URL is `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai`). Both can live in the git-ignored `.env` at the repo root (template `.env.example`), loaded by `ollama_client.load_env_file()` on import; the environment wins over `.env`. Shell scripts and the Makefile don't read `.env`.
  - The default model is `jev-latest`. Each prediction records the versioned `served_model`.
  - Retries on 429/5xx. `results_table` runs can pin a model: `embedded@jev:jev-1.13.0`.
  - To test the code path without a key, point `TYPESAFE_BASE_URL` at local Ollama with `--model nimble`.

## Constraints and gotchas
- **`/v1/systemone` limits:**
  - At most 64 questions per request, 2–26 options per choice and 64 KiB per request.
  - The rendered prompt must fit in nimble's fixed **8194-token** window. `num_ctx` is ignored and an oversized request returns HTTP 400.
  - Keep option descriptions short. The 26-option limit is why each group question flattens L2+L3 and why `group_question` asserts `≤ 26`.
- **Latency is ~15–16 s per query.** The server log (`~/.ollama/logs/server.log`) shows where it goes:
  - **~11 s is one prefill** of a ~5.5k-token prompt containing the state and every question's text (~500 tok/s on the M3 Max).
  - **~0.15–0.2 s per question** after that, as one sequential task each on a single slot; questions are not batched.
  - **The reported ~130k `input_tokens` is accounting** (every question is charged the full prompt), not compute.
  - **Nothing carries over between queries:** the query comes early in the prompt (shared prefix ≈ 86 tokens), and nimble's hybrid/recurrent memory blocks partial prefix reuse.
  - **What reduces latency:** shorter question text and fewer questions.
- **Prompt format, confirmed by nimble's source** (`nimble/scoring/parallel_schema.py`): system → `{"context": state, "schema": all fields}` → `"Requested field: " + <key>`, with one-letter codes A–Z.
  - Every question's text is in the shared prefill. The key is the only per-question cue.
  - nimble's MLX `ParallelScorer` scores all fields in one batched pass; Ollama does them sequentially.
  - nimble was trained on ≤ 2,048-token prompts; ours are ~5.4k.
- **Splits:**
  - ids 0–99 are **dev**, used for all tuning (thresholds, weights, scheme choice).
  - eval is the next 1000 ids. Never tune on eval.
- **Don't run qwen3 generation and nimble at the same time.** Ollama won't swap models while one is busy, and memory is tight (36 GB, heavy swapping). Generation uses `num_ctx=4096` for the same reason.
- **Manual gold fixes:** rows marked `manual_fix` in `flags` were corrected by hand. The original file is `data/eval_raw.before_review.jsonl`, and `data/REVIEW.md` records the dataset review.
- **Synthetic data:** the category tree must be a **product-type ontology, not room-based**. Rooms are only an attribute.
- **Figures:** `src/sidm/figures.py` (`make figures`, matplotlib from the uv group `figures`; the package stays stdlib-only) draws `docs/figures/*.svg` from shipped files only. Colors are fixed per run in `EVAL_RUNS` order. Latency comes from the eval runs; tev1 4B's is labeled inflated (`INFLATED`). The controlled latency sweep is `src/sidm/bench.py` (`make sweep`, `results/bench/sweep_<label>.jsonl`): one request sent with its first N questions, a new first word per request so Ollama can't reuse the prefill. Output is deterministic (fixed hashsalt, no date).
- **Run registry:** `src/sidm/runs.py` (`EVAL_RUNS`, `DEV_RUNS`) defines the runs in the shipped tables; `make results` builds them via `results_table --preset`, plus the readable tests `results/mcnemar_*.txt`.
- **Prediction file names** come only from `evaluate.pred_path` (`results/<split>_<label>_<scheme>.jsonl`). Colons in model tags become `-` in file names (`tev1:0.8b` → `tev1-0.8b`), while run names keep them (`embedded@ollama:tev1:0.8b`).
- **Runs resume** (completed queries are skipped). `OVERWRITE=1` (`--overwrite`) deletes a run's stored predictions for the split and re-runs it from scratch.
- **User-facing errors** use `ollama_client.SidmError(message, fix)`. Entry points are decorated with `@friendly_main`, which prints `error:`/`fix:` without a traceback. Use these for new failure modes that have a known fix.
