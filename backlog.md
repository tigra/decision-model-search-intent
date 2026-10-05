# Backlog

Ideas deliberately left out of the first experiment.

## Comparisons
- **tev1 4B / tev1:0.8b** on the same question set: accuracy vs latency trade-off.
- **Generative LLM baseline** (e.g. qwen3:14b / smaller) returning JSON {category, filters, residual} with a schema-constrained `format`; compare accuracy and latency.
- **Two-stage nimble**: request 1 = category (+ maybe word roles), request 2 = only filters applicable to the predicted category (fewer questions, smaller option lists).

## Modeling variants
- Per-word attribute tagging (`filter:<attr>`) instead of the 3 coarse roles; derive filter spans from it and cross-check with the filter questions.
- Multi-valued filters (e.g. "swivel reclining chair") via `noul` questions per value, or multiple feature questions.
- Numeric dimensions as real ranges ("60 inch", "under 30in", "between 48 and 60") instead of buckets; negation ("no arms").
- Confidence thresholds / abstention using `confidence` and probabilities; calibration curves.
- Consistency constraints between word roles and category/filter answers (e.g. if no word is tagged category, force category=none).
- Prompt/criteria wording ablations (with/without synonyms in descriptions, with/without child names in L1 options).

## Data
- Real query logs, or LLM-paraphrased hard cases (ambiguity: "oak" as material vs leg material, "storage bench").
- Multilingual queries; heavier typo noise.

- **"furniture" in queries without a product type.** Gold labels it residual ("modern plastic furniture"), but every model tags it as category, so no run gets any of the 48 no-category eval queries right (docs/figures/accuracy_by_difficulty.svg). Decide the convention (a category word for the root, or a stop word) and relabel, or describe it in the word-role question.

## Serving
- Concurrency / throughput tests (`OLLAMA_NUM_PARALLEL`), batching several queries per request as separate question groups.
- Question-count vs latency curve; caching of the static criteria prefix.

## Findings from the first run (to act on)
- **Latency (~15 s per query) is the main blocker.** The llama.cpp server log for one request breaks it down:
  - **One prefill (~11 s, ~70%)** of a ~5.5k-token prompt holding the state and the text of every question, at ~500 tok/s on the M3 Max.
  - **Then one sequential task per question** on the single slot: restore a checkpoint, process ~9 tokens, read the answer. That's ~0.15–0.2 s each, about 4–5 s for ~24 questions. The questions are not batched.
  - **The reported ~130k input tokens is accounting, not compute:** every question is charged the full prompt. Cost is roughly linear in total question text plus a constant per question. It is *not* quadratic, as I wrote earlier.
  - **Nothing is reused across queries.** The query comes early in the prompt, so consecutive queries share only ~86 tokens. nimble's hybrid/recurrent memory also prevents partial prefix reuse; llama.cpp logs "forcing full prompt re-processing".
- **Things to try:**
  - Shorter question and option text: cuts the prefill proportionally.
  - Drop the per-word questions and derive residual words from the category and filter answers plus the synonym tables: saves the per-question steps and part of the prefill.
  - Two-stage requests (category first, then applicable filters).
  - tev1 4B / 0.8B: faster prefill.
  - Server side: put the query *after* the questions in the prompt template, so the static ~5k-token prefix could be checkpointed and reused across queries. This would be the biggest win, but it needs an Ollama change.
  - `OLLAMA_NUM_PARALLEL` raises throughput across concurrent queries, not single-query latency.
- **Hard limit: 8194 tokens for the rendered prompt.** `num_ctx` is ignored by /v1/systemone. This caps the total size of the question set, not just the number of questions.
- ~~The top-level `cat_L1` question is unreliable~~ **Done: the `embedded` scheme.** It drops `cat_L1`, and every group question gets an `other_product` option. On the full 1,000-query eval, category accuracy is 0.917 vs 0.836, L1 accuracy 0.977 vs 0.867, and full exact match 0.424 vs 0.373, with latency unchanged (McNemar p < 0.0001). "No category" recall is still only 0.31 for embedded; tune its threshold on more data, or add a direct yes/no question "does the query name a product type?". The remaining category errors are mostly around "no category" (gold-none queries predicted as tables or dressers), where only 15 gold rows exist.
- **Filter false positives** (leg_material, style, color): currently handled by a p≥0.95 threshold. Worth trying: a calibrated threshold per attribute.
- **Word roles under-predict `category`:** currently handled by a ×8 class weight. Worth trying: consistency constraints (words tagged as category should match the predicted category's synonyms).
- **Masking filters by predicted category hurts** when the category is wrong. Only consider masking when category confidence is high.

## How /v1/systemone answers each question: open checks (one request each; run only when nimble is free)
The server log suggests the following (see the README, "Where the latency goes"):
- One prefill holds the preamble, `state` and the text of every question. Each question is then answered from a suffix of only `<question key>` + 1 separator + ~9 constant tokens (probably the chat template's assistant turn), with one logit-biased token as the answer.
- The prefill task is the shared prefix plus exactly 4 tokens (question 1's key + separator). llama.cpp checkpoints 4 tokens before the end of a prompt, so the checkpoint lands exactly at the shared boundary.

Checks:
1. **Is the "+4 tokens" in the prefill deliberate, or a coincidence of 4-token first keys?** (`cat_L1` and `cat_seating` are both 4.) Send one request whose **first** question has a long key, e.g. `filter_firmness_level_of_the_mattress` (≥ 6 tokens). Read `server.log`:
   - Is the prefill still shared + 4?
   - Do questions 2+ restore from the shared boundary, or from the earlier `n − 1024` checkpoint, re-processing ~1k tokens each?
   - If it's a coincidence, a long first key would make every question ~10× slower. In that case, always put a short-keyed question first.
2. **What exactly is in the prompt?** *Mostly answered by nimble's source (`nimble/scoring/parallel_schema.py`):* system → `{"context", "schema": all fields}` → `"Requested field: " + name` → chat template; one-letter codes A–Z. **Still open:** Ollama's own wrapping and the exact ~9-token cue.
   - Run Ollama with `OLLAMA_DEBUG=1` (or read the `/v1/systemone` implementation in Ollama's source) to see the rendered template.
   - Check: the order (preamble → state → questions → key → answer cue), the separator token, whether the ~9-token cue is the ChatML assistant header with an empty think block, and how options are labelled (letters?).

3. **Latency vs number of questions.** This would turn the log-based breakdown into a measured cost model.
   - **Is there caching between queries?** Only when two consecutive prompts share a long prefix.
     - Between our normal queries there is effectively none: the shared prefix is 86 tokens (`sim = 0.016 (86/5450)`), and llama.cpp logs "forcing full prompt re-processing".
     - Re-sending an identical request *is* cached: it took 5.8 s instead of 15 s, because the previous prefill's checkpoint matches.
     - A naive sweep with the same `state` and `questions[:N]` for N = 1, 2, … would hit this cache. Consecutive prompts share preamble + state + the texts of questions 1..N−1, so earlier checkpoints, or llama.cpp's prompt cache, could be restored.
     - **So vary the start of `state` in every request.** The query sits right after the 86-token preamble, so changing its first word, e.g. a nonce like `q17` or a different color word with the same token length and word count, cuts the shared prefix back to ~86 tokens.
     - Check this in `server.log`: each request should show `sim ≈ 86/…` and "forcing full prompt re-processing".
   - **Sweep A, realistic questions.** Take one embedded request (a 7-word query, 25 questions) and send `questions[:N]` for N = 1..25.
     - Repeat each N 3 times with different first words, and shuffle the order of runs to average out thermal or memory drift.
     - Record wall latency and, from the log, the prefill time and tokens and each question task's time.
     - N changes both the prefill size and the number of question tasks, so fit `latency ≈ a + b · prefill_tokens + c · N`.
   - **Sweep B, isolating the per-question cost.** Keep a fixed realistic set, then add K = 0..30 *tiny* dummy questions (2 options, a few tokens each). The prefill barely grows, so the slope is the pure sequential per-question cost. The log suggests ~0.18 s.
   - **Sweep C, isolating prefill cost per token.** Keep N fixed and pad the option descriptions of one question by 0..2k tokens. The slope is seconds per prefill token. The log suggests ~2 ms per token at ~500 tok/s.
   - **Combine with check 1** by running part of the sweep with a long first key.
   - **Expected, if the current model holds:** `latency ≈ ~1.4 s + prefill_tokens / ~500 tok/s + 0.18 s · N`.

Follow-up idea, since the key is the model's only per-question cue:
- Make keys self-describing, e.g. `word_3_oak` or `role_of_oak` instead of `word_3`.
- Measure word-role accuracy on dev; it's currently our weakest metric, at 0.84.

## Schema
- **`leg_color=natural` overlaps `leg_material=wood`.** Its wordings ("natural wood legs", "light wood legs") also express wood legs, which made row 1087 impossible to label with one attribute per span.
  - Option 1: reword it as color-only ("natural legs", "light legs").
  - Option 2: allow a span to carry several attribute=value pairs.

## From nimble's repository (github.com/bespokelabsai/nimble)
- **Ollama answers the fields sequentially; nimble's `ParallelScorer` batches them.** The MLX scorer broadcasts the prefix cache and scores all fields in one batched pass. Ollama runs 25 single-slot tasks at ~0.18 s each, ~4.6 s per query.
  - Options: raise this with Ollama.
  - Or benchmark nimble's own `ParallelScorer` (MLX) on this machine with the same schema. That's the batched reference to compare latency against.
- **Prompts are 2.5× longer than nimble's training limit** (2,048 tokens; ours are ~5.4k).
  - Measure accuracy against prompt length: dev runs with smaller question sets, e.g. category-only or applicable filters only.
  - Treat "≤ 2k-token prompts" as a design target, which supports the two-stage idea.
- **Our "instructions" and option descriptions map onto nimble's schema** (`name`, `description`, `choices[{code, value, description}]`). Our wording could be aligned with how nimble's training schemas phrase fields; examples are in `examples/` in the repo.

## Backend ablation: nimble's own `ParallelScorer` (MLX) next to Ollama
**Status:** implemented (`mlx_backend/`, `--backend mlx`).
- **Latency finding:** no faster than Ollama on this M3 Max at our ~6.5k-token prompts (~16.5 s per query). Prefill dominates, and batched field evaluation doesn't beat serial evaluation, because each branch copies the long attention cache. Numbers are in the README ("Latency on this M3 Max").
- **Accuracy comparison:** see `results/results_eval.md`.

The original plan follows.
**Goal:** run the same questions through nimble's reference batched scorer, keeping the Ollama path, and compare latency and accuracy on the same eval rows.

**Weights:**
- **Repos** (public, Apache-2.0):
  - [`bespokelabs/Bespoke-Nimble-9B`](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B): a LoRA adapter (~165 MiB) on Qwen3.5-9B. The repo's loader downloads the base and merges it once.
  - `bespokelabs/Bespoke-Nimble-9B-v2`: a merged full checkpoint.
- **Which one does Ollama's `nimble` correspond to?** Check, because v2's model card notes a different calibration (T = 1.0).
- **Runtime:** Python **3.12** with the repo's `requirements/mlx.txt`, Apple Silicon only. Setup and usage:
  ```bash
  python3.12 -m venv .venv-mlx && .venv-mlx/bin/pip install -r requirements/mlx.txt
  ```
  `ParallelScorer(**json.load(open(".cache/nimble-model.json")))`, loaded once and reused per context.
- **Memory:** 9B in bf16 is ~18 GB. Run `ollama stop nimble` first. Optionally also try an MLX 8-bit quantization to match the size of Ollama's GGUF, and record precision with every result.

**Design, keeping one code path for everything else:**
- **A small local server in the 3.12 MLX env** (`sidm_mlx_server`, separate from the 3.9 package) that implements the **same `/v1/systemone` contract**.
  - It maps each Jev question to a nimble schema field: question key → `name`, `instructions` → `description`, `criteria` key → choice `value`, criteria text → choice `description`, with codes A, B, … in order.
  - It calls `ParallelScorer`, then returns Jev-shaped `answers` (choice, probabilities and a confidence; define how confidence is computed, since Ollama's definition is unknown) plus timings: prefill time and batched scoring time.
- **On our side, the only change is selecting the endpoint and model.**
  - `ollama_client.system_one(..., backend=...)`, with backends `ollama` (localhost:11434) and `mlx` (the local server).
  - `--backend` on `sidm`, `evaluate run` and `results_table`.
  - Prediction files become `results/<split>_<backend>-<model>_<scheme>.jsonl`, and `results_table` gets one column per (backend, scheme).
  - Parser, decoding, metrics and dev-tuned thresholds stay shared. Re-check `none_thr` on dev for the new backend.
- **Fairness checks:**
  - Same eval rows and same question set.
  - Confirm the mapping reproduces Ollama's answers on a handful of queries; small differences from quantization or calibration are expected.
  - Report precision (bf16 vs the GGUF quantization) alongside the scores.

**What to measure:**
- Latency per query, split into prefill vs scoring of all fields.
- Accuracy on dev and eval with the same `results_table`.
- Peak memory.

**Expected:** the ~4.6 s of sequential per-question tasks should shrink to one batched pass. MLX prefill speed against llama.cpp's ~500 tok/s is unknown.

**Follow-ups this enables:**
- nimble's v2 card allows **up to 255 choices per field**, while Ollama caps at 26. A flat single category question over all 83 nodes becomes possible, as a third category scheme.
- Prompt-length sensitivity: the training limit is 2,048 tokens, but batched scoring makes many-question requests cheap.

## More backends and models for the ablation
**Status:** both are implemented and being evaluated: tev1 via Ollama with automatic request splitting, and Ollaya as `--backend ollaya`. Results are in `results/results_dev.md` / `results/results_eval.md`.

**1. tev1 via Ollama** (Together AI, Qwen3.5 fine-tunes; [repo](https://github.com/togethercomputer/tev1), [HF](https://huggingface.co/togethercomputer/Tev1-4B-experimental)). No code change needed.
- `ollama pull tev1` and `ollama pull tev1:0.8b`, then `--model tev1` or `--model tev1:0.8b`.
- Run on dev, then eval, with the same `results_table`.
- **Its prompt layout is one question per prompt:** `{"state", "question", "options": [{label, key, description}]}` with 2–24 lettered options. Ollama's bundled system prompt: "Evaluate the supplied decision task… Select exactly one listed option. Return only its letter".
  - Our largest group question has 24 options, which is right at that limit.
- **Expectations, to verify in `server.log`:**
  - Each question sees only the state plus its own options (~100–300 tokens), not the whole schema.
  - The state may be shared as a prefix across the questions of a request.
  - Questions still run one at a time, since Ollama is sequential for every model.
  - Prompts stay within tev1's 2,048-token training limit, and questions can't influence each other.
- **Reference point:** an A100 measurement of a single question gave nimble 192 ms, tev1-4B 158 ms and tev1-0.8B 53 ms, with accuracy 76.0 / 70.8 / 59.7 ([sotaaz](https://sotaaz.com/post/ollama-decision-models-en)).
- **Also watch:** whether tev1's calibration still suits our dev-tuned thresholds. Re-tune `none_thr`, the filter p-threshold and the word weight on dev for each model.

**2. Ollaya as a third backend** ([ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)), next to Ollama and nimble's MLX `ParallelScorer`.
- **Wire-compatible:** "`POST /v1/systemone` … wire-identical to TypeSafe", on port 11435, so it plugs into the same `--backend` switch with only the URL and model changed.
- **It claims a single forward pass for all questions**, mainly with encoder classifiers:
  - Laya (ModernBERT-large 421M, or the multilingual mmBERT-base 322M)
  - NLI (DeBERTa-v3-large, ModernBERT-large)
  - GLiClass (DeBERTa-v3-large)
  - Reported latency: "8–10 ms for five questions" on GPU.
- **It also serves decoders:** Qwen3.5 "Decider" 0.8B/2B/4B, Kev (LoRA plus a pointer head), Winnow and Nimble ("JSON schema scoring").
  - Check whether its Nimble path is batched; that would be an alternative to running `ParallelScorer` ourselves.
- **Apple Silicon:** llama.cpp Metal for GGUF, plus ONNX on CPU. Check which models actually run on this Mac.
- **Expected trade-off:** encoders should be far faster, with accuracy unknown on our task, especially word roles and 24-way category questions.
- **Same fairness rules** as the MLX ablation: same rows and questions, decoding re-tuned per model on dev, and precision and size recorded.

## Jev (TypeSafe hosted) backend
**Status:** implemented as `--backend jev`. It runs once an API key is available (`TYPESAFE_API_KEY`).
- **Code path tested** against local Ollama via `TYPESAFE_BASE_URL`.
- **Plan:**
  1. A few-row trial with `--limit`, to check limits, usage and cost.
  2. The 100 dev queries, then `evaluate tune`, recorded in `TUNED_BY_BACKEND["jev"]`.
  3. The 1,000 eval queries, compared with `embedded@ollama:nimble` and `embedded@mlx:nimble` in `results_table`.
- **To verify on the real API:** the question and option limits (we use ≤ 26 options), the rate limits, whether `state` objects are accepted as-is, and the latency distribution with network included.
- **Third-party reference:** 236–276 ms median for a 5-question request ([search result](https://www.kunalganglani.com/blog/ollaya-ollama-decision-setup)). Our requests have ~24 questions.

## Next, from the model comparison (see README "Results")
- **A hybrid:** category from winnow:e4b (0.966) or tev1 4B (0.942), filters and word roles from nimble. That costs two backends per query; measure whole-query accuracy and latency.
- **Encoders need a different word-role formulation,** e.g. one question per word with the word itself as the state, or a token-classification head. As posed now they answer uniformly.
- **winnow's residual detection is poor** (F1 0.18). Check whether its role probabilities just need re-weighting beyond ×16; the tuner's grid stops at 16.

## Developer UX (follow-ups to make doctor / make runs / friendly errors)
- **Fingerprint guard for stale predictions:** store a hash of the question set (and decoding-relevant parser version) in each prediction. Refuse to resume a file whose fingerprint differs from the current code, pointing to `OVERWRITE=1`; `results_table` warns when a run mixes fingerprints.
- **Tuning prerequisite:** `make eval` warns when a run has no per-model tuned settings, since it would fall back to nimble's defaults. A one-shot `make model BACKEND=… MODEL=…` would run check → dev → tune → eval → add to the registry.
- **Long runs:**
  - Wrap `dev`/`eval` in `caffeinate` automatically.
  - Print an ETA up front (rows × measured latency).
  - A GPU guard that refuses to start while another server holds a model (`FORCE=1` to override).
- **Shipping as a repo:** ship the code, `data/eval_raw.jsonl`, the predictions (`results/*.jsonl`, ~75 MB), `tuned_settings.json`, the tables and the docs. Add a `.gitignore` for logs, `.venv`, `results/bench/` scratch and the per-run ad-hoc tables. State in the docs that `make data` builds a *new* dataset, since regeneration isn't deterministic.

## Strands Decider (AWS Strands Labs)
**Status:** implemented as `--backend decider` (`decider_backend/`) and evaluated on dev and all 1,000 eval queries.
- **Result:** whole query 0.193, category 0.743, p50 3.9 s on MLX.
- **Speed:** about 4× faster than nimble on Ollama, thanks to its state-once / question-suffix layout.
- **Weak spots:** category, e.g. "coffee table" → desks, and residual words.

Follow-ups:
- **Question style.** Its training data probably favours short, self-contained questions. Try one category question per L2 group, or fewer options with clearer descriptions.
- **Combinations.** Use it as a fast first stage (category group), or try it on just the filter questions.
- **`--device mps` vs `mlx`** latency, and a check of the `num_slots` limit: we use ≤ 24 options and all were accepted.

## Figures (docs/figures, `make figures`)
- Interactive versions (hover tooltips) as a published HTML page; the SVGs are static.
- Add a jev column to every figure once the Jev run exists (EVAL_RUNS order fixes its color).
