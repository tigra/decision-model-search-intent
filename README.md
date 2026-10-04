# Search intent parsing with Jev-style decision models

An experiment in parsing a furniture e-commerce search query into three parts with a **decision model**: a model that answers typed multiple-choice questions with calibrated probabilities instead of generating text (TypeSafe's Jev API and its open counterparts).

For example:
```
query:     cheap grey oak coffee table with storage
category:  Tables > Coffee tables
filters:   color=grey, material=oak, feature=with_storage
residual:  cheap
words:     cheap/R grey/F oak/F coffee/C table/C with/F storage/F      (C category, F filter, R residual)
```

## Problem formulation
**Given:**
- **A query** `q = w_1 … w_N`: free text split into N whitespace-separated words, possibly with typos.
- **A category ontology** `T`: a product-type tree, L1 group → L2 type → optional L3 subtype. Here it has 6 L1, 36 L2 and 41 L3 nodes, e.g. Seating > Sofas > Sectional sofa.
- **Attributes** `a ∈ A`, each with a finite value set `V_a`. Here there are 12: color, material, leg material, leg color, style, shape, bed size, width, seat count, firmness, feature and room. Numeric width is bucketed into 5 ranges, e.g. 48–71 in.
- **Applicability:** which attributes make sense for which category (e.g. firmness only for mattresses).

**Predict:**
1. **Category** `c ∈ T ∪ {none}`: the **most specific node the query names**, at any depth. "seating" means the L1 group; "grey sectional" means the L3 node. `none` means the query names no product type, e.g. "minimalist furniture".
2. **Filters** `F ⊆ {(a, v) : a ∈ A, v ∈ V_a}`: the attribute values the query explicitly requests, at most one value per attribute (possibly none).
3. **Word roles** `r_i ∈ {category, filter, residual}` for every word `w_i`.
   - **The residual terms** are the words with role `residual`: price, quality and shopping words, brands, recipients ("cheap", "for my mom", "ikea").
   - **Connecting words** take the role of the phrase they belong to: "**with** storage" and "**under** 30 inch" are filter, "chest **of** drawers" is category, and "**for** my mom" is residual.

**Scoring** (definitions with counts are in `results/results_eval.md`):

| Part | Measure |
|---|---|
| Category | exact-node accuracy (depth must match; `none` is a value), plus accuracy at L1 / L2 / L3 |
| Filters | micro precision / recall / F1 over (attribute, value) pairs, plus filter-set exact match |
| Word roles | accuracy over all words; precision / recall / F1 for `residual` |
| **Whole query** | **exactly right:** category, filter set and the set of residual words all correct |

Two further measures: **latency** per query on local hardware, and paired significance tests between systems.

## Solution idea: one request, all questions, one decision model
Each query becomes **one `/v1/systemone` request**. The query is the `state`, and every sub-decision is a named `choice` question about it:
- **Category:** one question per top-level product group, e.g. "which kind of table, if any?". Each lists that group's L2/L3 types, plus "not this group" and "only the general word".
- **Filters:** one question per attribute (color, material, legs, style, size, width, …), each with `not_specified` plus the attribute's values.
- **Word roles:** one question per word of the query ("role of word 3, `oak`?"): `category`, `filter` or `residual`. This gives a token-level tagging, and the words tagged `residual` are the residual terms.

The decision model returns a probability for every option of every question. A small decoder turns those into the parse:
- the most confident category;
- filters above a probability threshold;
- the word roles.

Its thresholds are tuned on a dev split. Because the whole parse is a single typed request, it is easy to ablate, and every part can be scored separately.

One assumption is that Jev's value comes in parallelization of question answering and thus low latency, so it is interesting to see if other decision models have similar properties of a decoder.

## Goals of the study
1. **Feasibility and accuracy.** Can a decision model parse search intent in one request? How accurate is it per part (category, filters, word roles / residual words) and on the whole query?
2. **Question design.** Does the way the category is asked matter: an explicit top-level "which group?" question, or the group decision embedded in the group questions?
3. **Latency on local hardware** (a MacBook Pro M3 Max, 36 GB): where does the time go, and what would reduce it?
4. **Runtimes and models.** The same request served by:
   - Ollama's `nimble`;
   - nimble's own batched MLX scorer (`ParallelScorer`);
   - other open decision models (Together AI's `tev1`, and Ollaya's encoders and decoders);
   - TypeSafe's hosted Jev, pending an API key.

## Key findings
- **Best overall: `nimble` (9B) on Ollama with the `embedded` category scheme.**
  - 1,000 eval queries: category exact 0.917, filters F1 0.912, word roles 0.841, **whole query exactly right 0.424**.
  - It's significantly better on the whole query than every other run (McNemar p < 0.001), mainly thanks to filters and word roles.
- **Embedding the group decision beats an explicit top-level question:** category 0.917 vs 0.836. The top-level question picked wrong groups confidently.
- **Other models win on parts.**
  - `winnow:e4b` (Ollaya) has the best category accuracy (0.966) and is fastest (6 s), but barely finds residual words.
  - `tev1` 4B is also better at category (0.942) and ~30% faster, but weaker at filters and word roles.
- **Encoders and small models fail the word-role questions:** their role probabilities are nearly uniform, so they give an effectively constant answer. They can't resolve "role of word N" against the numbered word list in the state.
- **Latency is dominated by one prefill of the whole question set.** A query takes ~16 s with nimble.
  - Every question's text is in one shared prompt (~5.4k tokens), prefilled once at ~500 tok/s (~11 s). Then each question is answered from a ~13-token suffix (~0.18 s each, sequential).
  - nimble's batched MLX `ParallelScorer` is no faster here (17.4 s): batching the suffixes saves little at this prompt length.
  - The real levers are fewer or shorter questions, or a server that puts the static questions before the query, so they could be cached across queries.
- **The data are synthetic and cleaner than real queries,** so the accuracies are upper bounds.

## First 15 minutes
1. **Set up:** `make setup` installs the Python environment (uv).
2. **Check the machine:** `make doctor` shows which servers, models and keys this machine has, and which of the study's runs it could re-run. Every gap comes with its fix.
3. **See what's stored:** `make runs` lists every run with its stored predictions and decoding settings.
4. **Read the shipped results.** No commands are needed:
   - [`results/results_eval.md`](results/results_eval.md): all metrics on 1,000 eval queries, with counts and confidence intervals;
   - [`results/mcnemar_eval.txt`](results/mcnemar_eval.txt): which runs are significantly better than which;
   - the dev counterparts, `results_dev.md` and `mcnemar_dev.txt`, for all 14 runs;
   - per-run error analysis in `results/report_eval_<run>.md` (per-attribute scores, confusions, worst queries).
5. **Try the parser:** `make ollama-pull MODEL=nimble` (9.5 GB, needs Ollama ≥ 0.35), then `make parse Q="blue velvet sofa for my mom"`.

To re-run models and check the numbers, see "Reproducing the results" below.

## Dataset
- **Schema** (`src/sidm/schema.py`):
  - A product-type ontology: 6 L1 groups, 36 L2 and 41 L3 nodes, each with surface synonyms.
  - 12 categorical attributes with surface forms.
  - A residual vocabulary.
  - Import-time checks ensure that no synonym is shared by two nodes, and that residual words don't collide with category or attribute words.
- **Generation** (`src/sidm/gen_dataset.py`, model `qwen3:14b`):
  - Sample a gold intent (category at any depth, 0–3 filters, 0–2 residual phrases, occasional typos).
  - qwen3 writes a natural query and labels **every** word. A connecting word belongs to the filter or category phrase it sits in ("**with** storage", "chest **of** drawers"); otherwise it's residual.
  - Each row is validated: spans cover every word, labels map back to the gold intent, widths fall in their bucket, and residual words contain no category or attribute term. Failures are retried with the error fed back to qwen3.
- **Size and splits:** 1,100 rows in `data/eval_raw.jsonl`.
  - **dev** = ids 0–99, used for all tuning.
  - **eval** = ids 100–1099, 1,000 queries.
- **Review:** `data/REVIEW.md` records the manual review and a few hand fixes. `data/preview.txt` is a readable dump (`make preview`).

## Request design
### Request shape
This is the default `embedded` scheme; `router` differs only in the category questions.
```
POST /v1/systemone
{
  "model": "nimble",
  "keep_alive": "30m",                            // Ollama only
  "state": {
    "query":      "<query>",                      // original search query
    "words":      ["1: <w_1>", …, "N: <w_N>"],    // the query split into words
    "word_roles": "<role definitions + connecting-word rule>"
  },
  "questions": {
    // category: one question per L1 group g = 1..G
    "cat_<g>": {
      "type": "choice",
      "instructions": "Is a kind of <g> searched? Which exact product type? 'other_product' if …",
      "criteria": {
        "other_product": "a different kind of product, or no product named",
        "unspecified":   "only a general word for <g> (<synonyms>)",
        "<node_1>": "<Name> (<synonym>, <synonym>)[, type not specified]",
        …                                     // all M_g L2/L3 nodes of group g
      }
    },
    // filters: one question per attribute a = 1..A
    "filter_<a>": {
      "type": "choice",
      "instructions": "<attribute description> requested in the query? 'not_specified' if not mentioned.",
      "criteria": {
        "not_specified": null,
        "<value_1>": "<extra synonyms>" | null,
        …                                     // all V_a values of attribute a
      }
    },
    // word roles: one question per query word i = 1..N
    "word_<i>": {
      "type": "choice",
      "instructions": "Role of word <i> \"<w_i>\"?",
      "criteria": { "category": null, "filter": null, "residual": null }
    }
  }
}
```
- **G** is the number of L1 groups, and **M_g** the number of L2 and L3 nodes in group g. Each L2 is listed before its L3 children, and an L2 that has children is described as "type not specified". `cat_<g>` has **M_g + 2** options.
- **A** is the number of attributes and **V_a** the number of values of attribute a. `filter_<a>` has **V_a + 1** options. Width values are described by their inch range, e.g. "48-71 in".
- **N** is the number of whitespace-separated words, capped so the total stays within the API's 64 questions.
- **Total: G + A + N questions.**
  - Each question must have 2–26 options, so M_g ≤ 24 and V_a ≤ 25.
  - The rendered prompt must fit the model's context: 8,194 tokens for nimble.
  - **In our setup:** G = 6, A = 12 and N = 5–8 for a typical query, giving 23–26 questions and about 13 KB of JSON.
- **The `router` scheme** adds `"cat_L1"`, a choice over the G groups plus `"none"`. Its group questions ask "If <g> is searched, which exact product type is named?" and have **no `other_product`** option (M_g + 1 options). Total: 1 + G + A + N.
- **Response:** for each question, the chosen option, a probability per option and a confidence, plus token usage.
- **Full example:** [`docs/example_request.md`](docs/example_request.md) has a real request for each scheme, nimble's response and the decoded result (`make example-doc`).

### What the model actually sees (nimble)
This is nimble's prompt format, per `prepare_prompts` in [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble). That Ollama renders our questions into it exactly like this is inferred from its server log; see "Latency analysis".
```
system:    Classify the context using the supplied schema. … one-letter codes …
user:      {"context": <state>,
            "schema": [ {"name": "<question key>", "description": "<instructions>",
                         "choices": [ {"code": "A", "value": "<criteria key>", "description": "<criteria text>"}, … ]},
                        … one entry per question … ]}

            Requested field: <question key>
assistant: <one token: the code letter, scored over that question's codes>
```
- **Everything before the question key is shared** by all questions of a request and is prefilled once. Each question then appends only its key and the turn switch.
- **`state` comes before the schema,** so nothing is reused between queries.

### Decoding
The decoder lives in `src/sidm/parser.py`.
- **Category:** the most confident option across the G `cat_<g>` questions, ignoring `other_product`.
  - "No category" is predicted when a signal exceeds the scheme's threshold. For `embedded` the signal is 1 − the best candidate's probability; for `router` it is P(`cat_L1` = none).
- **Filters:** a value counts only above a probability threshold. The values are then masked to the attributes applicable to the predicted category.
- **Word roles:** argmax over the three roles, with the `category` probability up-weighted.
- **Thresholds and weights are tuned per model on dev** (`make tune` → `results/tuned_settings.json`), because models are calibrated very differently. The nimble defaults are: no-category threshold 0.70, filter p ≥ 0.95, `category` weight ×8.
- **Schemes:** the two category schemes are an ablation defined in `CategoryScheme` (`src/sidm/parser.py`). Everything else is shared.

## Backends and models
The same request can be served by four ablatable backends (`--backend`). A run is named `<scheme>@<backend>:<model>`, e.g. `embedded@ollaya:jeb:4b`.

| Backend | Server | How a request is scored |
|---|---|---|
| `ollama` (default) | Ollama 0.35, `localhost:11434` | one prefill of the whole question set, then **one question at a time**. Requests that exceed a model's context are split automatically (tev1) |
| `mlx` | `mlx_backend/server.py` (nimble's `ParallelScorer`), `localhost:11500` | one prefill, then fields batched (`--mode parallel`) or one at a time (`--mode cached_serial`) |
| `ollaya` | [Ollaya](https://github.com/ollaya-dev/ollaya) 0.9, `localhost:11435` | per model family: most score all questions in one batched pass; `winnow` shares the state prefix and runs questions sequentially |
| `jev` | TypeSafe's hosted API, `https://api.typesafe.ai` | hosted Jev, needs `TYPESAFE_API_KEY` |

### Models tested
| Model | Backend | Type, size, precision | Evaluated on |
|---|---|---|---|
| `nimble` (Bespoke Labs) | ollama | Qwen3.5-9B LoRA, GGUF | dev + eval, both schemes |
| `nimble` (Bespoke Labs) | mlx | the same adapter merged and converted by us: 8-bit MLX, `lm_head` in bf16 | dev |
| `tev1` (Together AI) | ollama | Qwen3.5-4B fine-tune | dev + eval |
| `tev1:0.8b` (Together AI) | ollama | Qwen3.5-0.8B fine-tune | dev + eval |
| `jeb:4b` | ollaya | Qwen3.5-4B, merged LoRA, GGUF Q8, llama.cpp Metal | dev + eval |
| `winnow:e4b` | ollaya | Gemma 4 E4B fine-tune, GGUF Q8, llama.cpp Metal | dev + eval |
| `decider:2b` | ollaya | Qwen3.5-2B decoder, ONNX on the CPU | dev |
| `decision:eos` | ollaya | Qwen3.5-0.8B with an endpoint head, ONNX on the CPU | dev |
| `kev:0.8b` | ollaya | Qwen3.5-0.8B LoRA with a pointer head, ONNX on the CPU | dev |
| `laya:en` | ollaya | ModernBERT-large encoder (421M), MLX | dev |
| `laya:typed-decisions` | ollaya | ModernBERT-large fine-tune, ONNX on the CPU | dev |
| `von` | ollaya | ModernBERT-large encoder (395M), ONNX on the CPU | dev |
| `nli:modernbert-large` | ollaya | NLI cross-encoder, one pair per option | dev |
| `jev-latest` (TypeSafe) | jev | hosted | not yet: implemented, waiting for an API key |

**Not tested:**
- **Too few options for our 24-option questions:** Ollaya's `jevk5` (≤ 16 options), `cygnet` (≤ 20) and `decider:2b-vision` (≤ 10).
- **Too slow here:** the ~18 GB CPU-only models `nimble`, `jeeves`, `clef`, `clm` and `kev:9b`.
- **Built-in questions only:** `qwen3guard`.

### nimble's MLX `ParallelScorer` (`mlx` backend)
It runs in a separate Python 3.12 environment in `mlx_backend/`. Model files go under `$SIDM_MODELS_DIR` (default `~/sidm-models`).
```bash
make mlx-convert        # once: download the adapter + Qwen3.5-9B, merge, quantize to 8-bit MLX (lm_head kept bf16), verify
make mlx-serve          # unloads Ollama's nimble (memory), serves on :11500
make parse BACKEND=mlx
```
- **The server:** it converts each question into nimble's schema (key → field name, `instructions` → description, criteria → choices) and returns Ollama-shaped answers plus nimble's own timings (`prefill_seconds`, `field_evaluation_seconds`).
- **`convert_model.py` is reusable:** it takes `--repo`/`--revision` (e.g. `bespokelabs/Bespoke-Nimble-9B-v2`), `--q-bits 4|8|none`, `--models-dir`, `--name` and `--keep-intermediate`.
  - Its stages (`download`, `merge`, `quantize`, `verify`, `config`) can each run on their own, and an interrupted run resumes.
  - Peak disk use is ~2× the base model (~40 GB for nimble 9B), with ~10.5 GB left at the end.

### Other models on Ollama (tev1)
`make ollama-pull MODEL=tev1`, then `make parse MODEL=tev1`.
- **Ollama renders tev1 like nimble,** with one shared prompt holding every question (~5.4k tokens). But tev1's context is ~2k tokens.
- **The client splits the request automatically.** The server answers `prompt has N tokens; expected 1–M`, so the client cuts the questions into balanced chunks that fit and merges the answers: 7–8 requests per query for tev1.
  - The chunk count is learned once per model.
  - Each prediction records `n_requests`, and latency is the sum of the requests.

### Ollaya
Installed user-locally, with models under `~/sidm-models/ollaya-models`:
```bash
OLLAYA_INSTALL_DIR=$HOME/sidm-models/ollaya OLLAYA_NO_SERVICE=1 sh install.sh   # install.sh from ollaya.dev
make ollaya-serve                          # unloads Ollama's models, serves on :11435
make ollaya-pull MODEL=jeb:4b
make parse BACKEND=ollaya MODEL=jeb:4b
```

### Jev (TypeSafe hosted)
Needs an API key:
```bash
export TYPESAFE_API_KEY=...                       # sent as "Authorization: Bearer …"
make parse BACKEND=jev
make dev BACKEND=jev LIMIT=5                      # paid API: try a few rows first
make dev BACKEND=jev && make tune BACKEND=jev && make eval BACKEND=jev
```
- **Requests:** the same body as Ollama's, minus `keep_alive`.
- **Retries:** HTTP 429 and 5xx are retried with backoff.
- **Model:** `jev-latest` moves with releases, so each prediction records the versioned model that answered (`served_model`). Pin a version with `MODEL=jev-1.13.0`.

**Run only one model on the GPU at a time.** Memory is tight with 36 GB; `make status` shows what's loaded where.

## Results
- **Full tables:** [`results/results_eval.md`](results/results_eval.md) (1,000 eval queries) and [`results/results_dev.md`](results/results_dev.md) (all runs on the 100 dev queries).
  - Each metric comes with its definition, counts and a 95% Wilson interval.
  - McNemar tests are given against the first run and for all pairs.
- **Machine-readable:** `results/results_eval.json` and `results/results_dev.json`, with per run its settings, every metric with counts and intervals, per-attribute filter scores, the word-role confusion matrix, latency, and all pairwise tests.
- **Per-query predictions with raw answers:** `results/<split>_<label>_<scheme>.jsonl`.
- **Readable significance tests:** `results/mcnemar_eval.txt` and `results/mcnemar_dev.txt`, every pair of runs with the winner or "no significant difference".
- **Detailed per-run reports:** `results/report_eval_<run>.md`, one per eval run, e.g. [`report_eval_nimble_embedded.md`](results/report_eval_nimble_embedded.md). Each has per-attribute filter scores, the word-role confusion matrix, the top category confusions and the 20 worst queries with gold vs predicted parse.

### Eval: 1,000 queries (decoding tuned per model on dev)
| Metric (measure) | nimble embedded (ollama) | nimble router (ollama) | tev1 4B (ollama) | tev1 0.8B (ollama) | jeb:4b (ollaya) | winnow:e4b (ollaya) |
|---|---|---|---|---|---|---|
| Category, exact node (accuracy) | 0.917 | 0.836 | 0.942 | 0.727 | 0.918 | **0.966** |
| Category correct at L1 (accuracy) | 0.977 | 0.867 | **0.987** | 0.809 | 0.973 | 0.985 |
| Filters F1 (micro) | **0.912** | 0.892 | 0.862 | 0.655 | 0.888 | 0.840 |
| Filter set exact match (accuracy) | **0.770** | 0.742 | 0.612 | 0.326 | 0.720 | 0.619 |
| Word-role accuracy | **0.841** | 0.839 | 0.684 | 0.316 | 0.755 | 0.699 |
| Residual words F1 | 0.733 | 0.730 | **0.744** | 0.000 | 0.690 | 0.183 |
| **Whole query exactly right** (accuracy) | **0.424** | 0.373 | 0.336 | 0.090 | 0.365 | 0.245 |
| Latency p50, clean runs | 15.6 s | 15.9 s | 10.7 s | **2.2 s** | 11.0 s | 6.0 s |

- **Latency:** for tev1 4B, jeb and winnow it comes from a clean 30-query benchmark (`make bench`). Their full evals ran in parallel with CPU jobs, which inflated those latencies.
- **tev1 0.8B is the fastest decoder (2.2 s) but far behind:** whole query 0.090, with near-uniform word roles and no residual words found.
- **nimble with the `embedded` scheme is best on the whole query.** Every other run is significantly worse (McNemar p < 0.001).
- **On category, winnow:e4b and tev1 4B are significantly better than nimble** (p < 0.0001 and p = 0.002). jeb:4b ties nimble on category (p = 1.0).
- **jeb:4b beats tev1 4B on the whole query** (116 vs 87 queries, p = 0.049), but tev1 is better at category.
- **A hybrid could combine the strengths:** category from winnow or tev1, filters and word roles from nimble. Not tried yet.

### Dev: all runs (100 queries)
Ranked by whole-query exact match. Latency is from clean runs.

| Model | Backend | Whole query | Category | p50 latency |
|---|---|---|---|---|
| nimble (embedded) | ollama | 0.44 | 0.91 | 16.1 s |
| nimble (embedded) | mlx | 0.43 | 0.92 | 17.4 s |
| tev1 4B | ollama | 0.42 | 0.91 | 10.2 s |
| jeb:4b | ollaya | 0.37 | 0.90 | 11.0 s |
| nimble (router) | ollama | 0.35 | 0.85 | 15.4 s |
| winnow:e4b | ollaya | 0.29 | 0.96 | 6.0 s |
| decider:2b | ollaya (CPU) | 0.24 | 0.82 | 46.7 s |
| decision:eos | ollaya (CPU) | 0.11 | 0.76 | 25.8 s |
| kev:0.8b | ollaya (CPU) | 0.11 | 0.76 | 20.0 s |
| tev1 0.8B | ollama | 0.10 | 0.75 | 2.2 s |
| laya:en | ollaya (MLX encoder) | 0.10 | 0.76 | 1.1 s |
| laya:typed-decisions | ollaya (CPU encoder) | 0.06 | 0.73 | 14.6 s |
| von | ollaya (CPU encoder) | 0.06 | 0.55 | 13.0 s |
| nli:modernbert-large | ollaya (encoder) | 0.04 | 0.54 | 5.7 s |

- **Dev is the tuning set,** so these numbers are optimistic.
- **The encoders and small decoders answer word-role questions almost uniformly** (word accuracy ≈ 0.33, residual F1 = 0), and they are weak on filters.
- **Models scoring at least 0.24 on dev got the 1,000-query eval,** plus tev1 0.8B as the fastest decoder.

### How runs are compared: McNemar's test
McNemar's test checks whether **two runs scored on the same queries** really differ in accuracy, or whether the difference could be chance.

1. **Score each query right or wrong for both runs.**

   | | B right | B wrong |
   |---|---|---|
   | **A right** | both right | **only A right** |
   | **A wrong** | **only B right** | both wrong |

2. **Ignore queries where both runs agree.**
3. **Test the disagreements.** If A and B were equally good, each disagreement would be equally likely to favor either side, like a fair coin. We use the exact two-sided binomial test on the disagreements (`mcnemar_p` in `src/sidm/results_table.py`).

**What *p* means:** the probability of a split at least as lopsided as the observed one, *if the two runs were really equally accurate*.
- **p < 0.05** means the difference is significant, and the run with more "only me right" queries is better.
- **A large p** means this test can't tell the runs apart. It doesn't prove they are equal.
- **Caveats:** p doesn't measure *how big* a difference is. And with many pairs tested, about 1 in 20 truly-equal pairs will show p < 0.05 by chance.

**Why not just compare accuracies with their confidence intervals?** All runs answer the same queries, and their errors are correlated (some queries are hard for every model). McNemar uses that pairing directly.

**Where to find the tests:**
- The tables test every run against the first one, then show all-pairs matrices for category and whole-query exact match. Each cell is "only row right : only column right", with the p-value, and is bold when p < 0.05.
- The JSON files list every pair in `paired_mcnemar`.
- To print them as readable lines, with the winner or "no significant difference":
  ```bash
  make mcnemar                          # dev runs (SPLIT=dev is the default)
  make mcnemar SPLIT=eval               # eval runs
  make mcnemar SPLIT=eval METRIC=full   # only whole-query exact match (METRIC=category for category)
  make mcnemar SPLIT=dev SIG=1          # only significant differences (p < 0.05)
  ```

## Reproducing the results
### What's shipped and what isn't
- **In the repo:**
  - the dataset (`data/eval_raw.jsonl`);
  - every run's predictions with the models' raw answers (`results/<split>_<label>_<scheme>.jsonl`);
  - the per-model decoding settings (`results/tuned_settings.json`);
  - the tables and tests.
- **Not in the repo:** models, servers and API keys. `make doctor` reports what this machine has and how to get the rest.
- **The dataset is frozen.** Regenerating it with qwen3 (`make data`) isn't deterministic, and it would lose the hand-fixed rows. Use `make data` only to build a *new* dataset.

### How re-running works
- **Runs resume.** `make dev` / `make eval` skip queries that already have stored predictions. With the shipped files complete, `make eval MODEL=tev1` only rebuilds that run's table and says "all rows already done".
- **To re-run a run from scratch, add `OVERWRITE=1`.** It deletes that run's stored predictions for the split and queries the model again; the tables then use the new answers.
  - With `LIMIT=N`, only N queries are re-run. The tables then cover just those N queries for that run until it's complete again.
- **Decoding is re-applied on every table build,** so a re-run is decoded with the shipped tuned settings, unless you re-tune with `make tune`.

### Steps
1. **Pick runs this machine can serve.** `make doctor` lists them; pull or start what's missing.
2. **Re-run a run:**
   ```bash
   make dev  BACKEND=ollama MODEL=tev1 OVERWRITE=1      # 100 dev queries
   make tune BACKEND=ollama MODEL=tev1                  # optional: re-tune decoding on the new dev answers
   make eval BACKEND=ollama MODEL=tev1 OVERWRITE=1      # 1,000 eval queries
   make results                                         # rebuild the tables, mcnemar_*.txt and the per-run reports
   ```
   Then compare with the numbers above.
   - Re-running tev1 0.8B's dev split this way reproduced its metrics exactly: category 0.750, whole query 0.100.
3. **Recalculate the significance tests:** `make results` rewrites `results/mcnemar_*.txt`. `make mcnemar SPLIT=eval [METRIC=full|category] [SIG=1]` prints them, optionally filtered.
4. **Add a new run:** `make dev` → `make tune` → `make eval` with the new `BACKEND`/`MODEL`, then add it to `EVAL_RUNS` / `DEV_RUNS` in `src/sidm/runs.py`, then `make results`.

**Time and space per run**, from the measured p50 latency on an M3 Max with one model loaded; dev takes a tenth of eval:

| Run | Model size | Eval (1,000 queries) |
|---|---|---|
| `embedded@ollama:nimble`, `router@ollama:nimble` | 9.5 GB | ~4.3 h each |
| `embedded@mlx:nimble` (dev only so far) | 10.5 GB after `make mlx-convert` (~40 GB peak) | ~4.8 h |
| `embedded@ollama:tev1` | 4.5 GB | ~3 h |
| `embedded@ollama:tev1:0.8b` | 0.8 GB | ~40 min |
| `embedded@ollaya:jeb:4b` | 4.5 GB | ~3 h |
| `embedded@ollaya:winnow:e4b` | 8 GB | ~1.7 h |
| Ollaya CPU models (dev only) | 0.8–3.6 GB | 20–47 s per query |

**Caveats:**
- **Latencies depend on the hardware.** Other Ollama or Ollaya versions, or other quantizations, can change answers slightly.
- **Run one model on the GPU at a time** (`make status` shows what's loaded).
- **Keep the Mac awake** during long runs.
- **Jev needs `TYPESAFE_API_KEY` and is a paid API.** Start with `LIMIT=5`.

## Latency analysis (nimble only)
This section analyzes **`nimble` only**: on Ollama, from its llama.cpp server log, and on MLX via nimble's own `ParallelScorer` timings. The other models were only timed end to end (p50 latencies in "Results").
- **tev1 should behave much the same:** Ollama renders it with the same shared prompt, split into 7–8 requests because of its 2k context.
- **Ollaya's models weren't analyzed;** their engines (ONNX, MLX, llama.cpp) score questions differently.

### Where the time goes: Ollama's server log (nimble)
Below is one nimble request from `~/.ollama/logs/server.log`: an embedded eval query (id 264, "storage furniture with glass doors chrome legs", 25 questions, 16.83 s wall time). Lines are trimmed, and `…` marks omitted lines.
The request ran as **26 server tasks: one prefill task plus one task per question**.

```
slot get_availabl: id  0 | task -1    |  - checking sim = 0.016 (86/5450) > 0.100
slot launch_slot_: id  0 | task -1    | sampler chain: logits -> ?penalties -> ?dry -> … -> top-p -> min-p -> ?xtc -> temp-ext -> dist     ← task 1: ordinary sampler
slot   operator(): id  0 | task 35089 | new prompt, n_ctx_slot = 8448, n_keep = 4, task.n_tokens = 5450
slot   operator(): id  0 | task 35089 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory …)
slot print_timing: id  0 | task 35089 | prompt processing, n_tokens =   3072, progress = 0.56, t =   3.98 s / 772.76 tokens per second
slot print_timing: id  0 | task 35089 | prompt processing, n_tokens =   4426, progress = 0.81, t =   8.02 s / 551.94 tokens per second
slot print_timing: id  0 | task 35089 | prompt processing, n_tokens =   5446, progress = 1.00, t =   8.74 s / 623.12 tokens per second
slot create_check: id  0 | task 35089 | created context checkpoint 2 of 32 (pos_min = 5445, pos_max = 5445, n_tokens = 5446, size = 50.251 MiB)
slot print_timing: id  0 | task 35089 | prompt eval time =   10843.61 ms /  5450 tokens (    1.99 ms per token,   502.60 tokens per second)
slot      release: id  0 | task 35089 | stop processing: n_tokens = 5450, truncated = 0
slot get_availabl: id  0 | task -1    |  - checking sim = 0.998 (5450/5459) > 0.100     ← question 1: task 1's prompt is a prefix of it
slot launch_slot_: id  0 | task -1    | sampler chain: logits -> logit-bias -> top-k -> ?temp-ext -> dist          ← tasks 2–26: constrained answer
slot   operator(): id  0 | task 35097 | new prompt, n_ctx_slot = 8448, n_keep = 4, task.n_tokens = 5459
slot print_timing: id  0 | task 35097 | prompt eval time =     147.10 ms /     9 tokens (   16.34 ms per token,    61.18 tokens per second)
slot      release: id  0 | task 35097 | stop processing: n_tokens = 5459, truncated = 0
slot get_availabl: id  0 | task -1    |  - checking sim = 0.998 (5447/5458) > 0.100     ← question 2: shares only 5447 tokens
slot   operator(): id  0 | task 35100 | new prompt, n_ctx_slot = 8448, n_keep = 4, task.n_tokens = 5458
slot   operator(): id  0 | task 35100 | restored context checkpoint (pos_min = 5445, pos_max = 5445, n_tokens = 5446, n_past = 5446, size = 50.251 MiB)
slot print_timing: id  0 | task 35100 | prompt eval time =     185.31 ms /    12 tokens (   15.44 ms per token,    64.76 tokens per second)
slot      release: id  0 | task 35100 | stop processing: n_tokens = 5458, truncated = 0
…                                       (23 more tasks like 35100, for questions 3–25)
[GIN] 2026/09/30 - 20:37:03 | 200 | 16.832918125s | 127.0.0.1 | POST "/v1/systemone"
```

| Step | Tasks | Time | Share of 16.83 s |
|---|---|---|---|
| Prefill task: the shared prompt + the first ~4 tokens of question 1's suffix; answers no question | 1 | 10.84 s (5,450 tokens at ~500 tok/s) | 64% |
| Question tasks: question 1 continues from the live state; questions 2–25 each restore the checkpoint; each processes ~9–13 tokens and reads one constrained answer token | 25 | 4.59 s (mean 184 ms each) | 27% |
| Rest: rendering, tokenization, scheduling, HTTP | n/a | ~1.4 s | 8% |

### How `/v1/systemone` is computed (Ollama, nimble)
This is read from the log. The prompt format is confirmed by nimble's source.
1. **One prompt holds everything shared:** a template preamble, then `state`, then the text of **every** question with its options.
   - That's ~5,446 tokens here, growing with total question text: ~5,000 + 66 per word for our question set.
2. **Each question is then a short suffix,** 12–14 tokens: its key plus the turn switch.
   - The answer is **one constrained token** (`logit-bias` sampler; one-letter codes A–Z).
   - Option probabilities come from a softmax over that token's logits for the codes.
3. **The shared part is prefilled once,** in a separate prefill task.
   - That task's prompt is the shared part plus the first ~4 tokens of question 1's suffix. Its one generated token is counted in usage (`output_tokens` = questions + 1) but answers nothing.
   - A context checkpoint is saved at the end of the **whole shared part: preamble, state and every question's text**.
4. **The questions are answered one after another,** not in parallel.
   - All tasks run on one slot. Each restores the 50 MB checkpoint, processes its suffix and reads the answer.
   - At ~16 ms per token, these tiny suffixes are ~8× less efficient than the bulk prefill (~2 ms per token).
5. **Nothing carries over between queries.**
   - The query comes right after the 86-token preamble, so consecutive queries share only 86 tokens.
   - nimble's hybrid/recurrent memory can only be restored from checkpoints, not cut back to an arbitrary prefix.
   - The reported `input_tokens` (~130k) charges every question the full prompt. The actual compute is one ~5.4k prefill plus 25 short suffixes.

### What is in each question's suffix (Ollama, nimble)
*Suffix* is the question's prompt length minus the 5,446-token shared part. *Shared with previous* is the number of suffix tokens in common with the previous question's suffix (from `checking sim = …`).

| Question(s) | Suffix tokens | Shared with previous |
|---|---|---|
| `cat_seating` (first) | 13 | 4 (with the prefill task) |
| `cat_tables`, `cat_beds`, `cat_mattresses`, `cat_storage`, `cat_desks` | 12 / 13 / 14 / 12 / 13 | 1 each |
| `filter_color` (first filter) | 12 | **0** |
| other `filter_*` | 12–14 | 1, but **2** for `filter_leg_material` → `filter_leg_color` |
| `word_1` (first word) | 13 | **0** |
| `word_2` … `word_7` | 13 each | **2** each |

- **Facts from the log:** consecutive suffixes share exactly as many leading tokens as the question keys share leading parts, and the longest keys have the longest suffixes. So each suffix starts with the question key.
- **Confirmed by nimble's `prepare_prompts`** (`nimble/scoring/parallel_schema.py`):
  - The user message is `safe_json({"context": context, "schema": fields}) + "\n\nRequested field: " + safe_json(name)`. `fields` is the **whole schema**, with lettered choices.
  - The chat template is applied with `enable_thinking=False`.
  - The shared prefix is the common token prefix of all the per-field prompts.
  - **So the model reads all questions once, and for each answer is told only "Requested field: `<key>`".** The key is the question's only per-question cue.
- **Not verified:**
  - Ollama's own wrapping.
  - The exact ~9-token turn switch.
  - Whether the prefill is exactly +4 tokens on purpose, which makes llama.cpp's end-minus-4 checkpoint land on the shared boundary. The open checks are listed in `backlog.md`.
- **Consequences:**
  - A question's own text affects only the shared prefill, not its per-question cost (~0.18 s).
  - Keys matter for accuracy: `word_1`…`word_7` differ only by a digit, so self-describing keys are worth trying.
  - Every question is answered with all the others in context.

### Batched scoring: nimble's `ParallelScorer` vs Ollama
- **What `ParallelScorer` does:** nimble's MLX scorer prefills the same shared prompt once. It then copies that cache across a batch (`broadcast_cache` → `mx.broadcast_to`) and scores all fields' suffixes in **one batched forward pass**.
- **What Ollama does:** it prefills once too, but answers the fields one at a time. Its System One implementation is sequential by design (PR #18606) and has no batching setting.
- **Measured on this M3 Max** (server-side timings, the same 25-question request, median of 3 queries):

| Backend / mode | Wall | Prefill | Field evaluation |
|---|---|---|---|
| Ollama (llama.cpp, questions sequential) | 15.6 s (p50 over 1,000 queries) | ~10.8 s (5.4k tokens) | ~4.6 s (25 × ~0.18 s) |
| MLX `parallel`, all 25 fields in one batch | 17.5 s | 12.7 s (6.5k tokens) | 3.4 s (one run spiked to 11.6 s; peak 16.9 GiB) |
| MLX `parallel`, field batch size 1 | 16.4 s | 12.4 s | 2.8 s |
| MLX `cached_serial` (one prefill, then fields one at a time) | 16.4 s | 11.7 s | 2.9 s |

- **On the 100 dev queries,** MLX takes 17.6 s per query vs Ollama's 16.2 s.
  - Scoring is ~1.4 s faster on MLX.
  - But nimble's own rendering is ~1k tokens longer (6.4k vs 5.4k; it adds a letter code to every option), and that costs ~2.3 s more prefill.
- **Batching gains little at this prompt length.** Each batched branch has to copy the long attention cache, so all fields cost ~3 s whether batched or not.
  - nimble's reported 7× speed-up is against re-prefilling the prompt for every field, which Ollama already avoids.
  - All modes return identical answers. The eval runs used `cached_serial`.
- **nimble was trained on prompts of up to 2,048 tokens.** Ours are ~5.4k (the checkpoint supports 8,192), which may also cost accuracy.

### What would reduce latency (nimble)
- **Prefill (~2/3 of the time)** scales with the **total text of all questions.**
  - Fewer or shorter questions cut it proportionally. Examples: shorter option descriptions; a two-stage request (category first, then only the filters that apply); dropping per-word questions, which saves ~0.3 s per word.
- **Two server-side changes would be needed for a big speed-up:**
  - Questions rendered *before* the state, so the static ~5k tokens become a reusable prefix.
  - Question suffixes evaluated as one batch.
- **`OLLAMA_NUM_PARALLEL`** raises throughput across concurrent queries, not the latency of one query.

## Usage
### Makefile
`make` (or `make help`) lists every workflow. Settings are variables: `BACKEND`, `MODEL`, `SCHEME`, `SPLIT`, `LIMIT`, `OVERWRITE`, `Q`, `V`.
```bash
make doctor                                           # what this machine has; which runs can be re-run (with fixes)
make runs                                             # status of every run: stored dev/eval predictions, decoding settings
make parse Q="blue velvet sofa for my mom"            # one query (default: nimble on Ollama, embedded scheme)
make dev   BACKEND=ollaya MODEL=jeb:4b                # 100 dev queries for a run (resumable), plus its table
make tune  BACKEND=ollaya MODEL=jeb:4b                # tune that run's decoding on dev -> results/tuned_settings.json
make eval  BACKEND=ollaya MODEL=jeb:4b                # 1,000 eval queries (resumes; OVERWRITE=1 re-runs from scratch)
make report BACKEND=ollama MODEL=tev1 SPLIT=eval      # detailed report: per-attribute scores, confusions, worst queries
make bench BACKEND=ollama MODEL=tev1                  # clean latency on 30 dev queries (nothing else running)
make results                                          # rebuild tables, mcnemar files and per-run reports from stored predictions (runs: src/sidm/runs.py)
make mcnemar SPLIT=eval [METRIC=full] [SIG=1]         # print the pairwise McNemar tests readably
make data                                             # generate / resume the dataset (needs qwen3:14b)
make status                                           # what's loaded in Ollama / Ollaya / MLX, and free disk
make ollama-pull | ollama-rm | ollama-unload | mlx-convert | mlx-serve | mlx-stop | ollaya-serve | ollaya-pull | ollaya-rm | ollaya-stop
```
- **Errors explain themselves:** a missing server, model or key prints `error:` and `fix:` lines, e.g. `fix: make ollama-pull MODEL=tev1`.
- **Re-decoding needs no model:** predictions store the raw answers, so `report`, `tune` and `results` re-decode them with the current code.
- **One model at a time:** don't run dataset generation and a GPU eval together; only one model on the GPU.

### Command line (`sidm`)
```bash
uv run sidm "cheap grey oak coffee table with storage"   # category, filters, residual, latency
uv run sidm -v   "..."   # + per-word roles with probabilities, top category candidates, each filter's probability and kept/dropped reason
uv run sidm -vv  "..."   # + top-3 options of every question, token usage
uv run sidm -vvv "..."   # + raw request and response JSON
uv run sidm --json "..." # decoded result as JSON
uv run sidm --untuned "..."                      # plain argmax decoding
uv run sidm --scheme router "..."                # the other category scheme
uv run sidm --backend ollaya --model jeb:4b "..."   # any backend / model
```

### Underlying commands
```bash
uv run python -m sidm.gen_dataset --n 1100 --out data/eval_raw.jsonl --workers 2                 # dataset (resumable)
uv run python -m sidm.results_table --runs embedded@ollama:tev1 --run --split dev                # run/resume + table
uv run python -m sidm.evaluate tune --apply --pred results/dev_tev1_embedded.jsonl               # per-model tuning
uv run python -m sidm.evaluate report --pred results/eval_tev1_embedded.jsonl --md results/report_eval_tev1_embedded.md
scripts/make_results.sh                                                                           # all result tables
```

## Layout
- **`src/sidm/`**, a Python 3.9 package with no dependencies beyond the standard library:
  - `schema.py`: ontology, attributes, residual vocabulary and consistency checks.
  - `gen_dataset.py`: synthetic dataset generation and validation.
  - `parser.py`: request building, category schemes, decoding and per-model tuned settings.
  - `ollama_client.py`: the backends, retries and automatic request splitting.
  - `evaluate.py`: `run`, `report`, `tune`, `compare`, and the prediction file paths (`pred_path`).
  - `results_table.py`: the results tables (Markdown + JSON) and McNemar tests.
  - `runs.py`: the registry of the study's runs (`EVAL_RUNS`, `DEV_RUNS`) and their status (`make runs`).
  - `doctor.py`: the machine check (`make doctor`).
  - `cli.py`: the `sidm` command.
  - `example_doc.py`: builds `docs/example_request.md`.
  - `preview.py`: the readable dataset dump.
- **`mlx_backend/`**, a separate Python 3.12 project:
  - `convert_model.py`: reusable adapter → MLX conversion.
  - `server.py`: `/v1/systemone` over `ParallelScorer`.
  - `serve.sh`: starts the server.
- **`scripts/`:** `make_results.sh` (all tables and McNemar files, from the presets in `runs.py`) and `ollaya_serve.sh`.
- **`Makefile`:** the typical workflows.
- **`data/`:**
  - `eval_raw.jsonl`: the dataset.
  - `schema.json`: the schema as JSON.
  - `REVIEW.md`: the manual review.
  - `preview.txt`: the readable dump.
- **`results/`:**
  - `results_{eval,dev}.{md,json}`: the main tables.
  - `mcnemar_{eval,dev}.txt`: readable pairwise significance tests.
  - `<split>_<label>_<scheme>.jsonl`: per-query predictions.
  - `tuned_settings.json`: per-model decoding settings.
  - `report_eval_<run>.md`: detailed per-run eval reports.
  - `bench/`: latency benchmarks.
  - Run logs and the helper scripts used for the long runs.
- **`docs/example_request.md`:** a real request and response.
- **`backlog.md`:** findings and next experiments.
- **`CLAUDE.md`:** notes for coding agents.
