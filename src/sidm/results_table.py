"""Results table: every metric with its exact definition, counts and 95% confidence interval, per run.

Writes <md> and a machine-readable <md stem>.json (per run: settings, every metric with counts and 95% intervals,
latency; plus exact McNemar tests for every pair of runs, on category and whole-query exact match).

A run is <scheme>@<backend>[:<model>], e.g. embedded@ollama:nimble, embedded@ollama:tev1, embedded@jev:jev-1.13.0
(the model defaults per backend: ollama_client.DEFAULT_MODELS). Labels in the output always name the model.

  python -m sidm.results_table                       # all default runs that have prediction files
  python -m sidm.results_table --runs embedded@ollama:nimble embedded@mlx:nimble
  python -m sidm.results_table --runs embedded@mlx:nimble --run   # first run/resume the parser on missing rows
  python -m sidm.results_table --preset eval --md results/results_eval.md   # the shipped eval table (runs in sidm/runs.py)
  scripts/make_results.sh                                         # regenerate all result tables
  python -m sidm.results_table --show-tests results/results_eval.json [--metric full] [--significant]

Prediction files are evaluate.pred_path(): results/<split>_<model>_<scheme>.jsonl for Ollama and
results/<split>_<backend>-<model>_<scheme>.jsonl otherwise. Running is resumable. All numbers use each
backend's dev-tuned decoding; the first run is the reference for the paired significance tests.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from sidm import evaluate as E
from sidm.ollama_client import DEFAULT_MODELS, SidmError, friendly_main
from sidm.parser import SCHEMES, tuned_settings
from sidm.runs import PRESETS

DEFAULT_RUNS = ["embedded@ollama:nimble", "router@ollama:nimble", "embedded@mlx:nimble", "embedded@jev:jev-latest"]
BACKEND_INFO = {
    "ollama": "Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context",
    "mlx": "nimble's MLX `ParallelScorer` (8-bit, `lm_head` in bf16) via `mlx_backend/server.py`; "
           "one prefill, then fields batched or one at a time (mode recorded per query)",
    "ollaya": "Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run",
    "decider": "AWS Strands Labs' Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v19`, MLX) via its own "
               "`/v1/systemone` server; the state is encoded once and only each question's suffix is added, batched",
    "jev": "TypeSafe's hosted Jev API (`https://api.typesafe.ai`); latency includes the network round trip",
}


def parse_run(run):
    """'<scheme>@<backend>[:<model>]' -> (scheme, backend, model)."""
    scheme, _, rest = run.partition("@")
    backend, _, model = rest.partition(":")
    if scheme not in SCHEMES or backend not in BACKEND_INFO:
        raise SystemExit("bad run %r: expected <scheme>@<backend>[:<model>] with scheme in %s, backend in %s"
                         % (run, sorted(SCHEMES), sorted(BACKEND_INFO)))
    return scheme, backend, model or DEFAULT_MODELS[backend]


def canonical_run(run):
    """Always name the model in labels: 'embedded@ollama' -> 'embedded@ollama:nimble'."""
    scheme, backend, model = parse_run(run)
    return "%s@%s:%s" % (scheme, backend, model)


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired booleans (same queries, two systems)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


# ---------------------------------------------------------------- metric specification
def prop(k, n):
    """A proportion with counts and CI: '0.945 (156/165) [0.90–0.97]'."""
    if not n:
        return "n/a (0/0)"
    lo, hi = wilson(k, n)
    return "%.3f (%d/%d) [%.2f–%.2f]" % (k / n, k, n, lo, hi)


def f1(c):
    p, r, f = E.prf(c["tp"], c["fp"], c["fn"])
    return "%.3f" % f


# (metric, measure, counted over, cell(metrics, latency))
SPEC = [
    ("Category, exact node", "accuracy", "all queries; the predicted node must equal the gold node "
     "(same depth; 'no category' is a value)", lambda m, L: prop(*m["category_exact_counts"])),
    ("Category correct at L1", "accuracy", "queries with a gold category; predicted path has the gold L1 node",
     lambda m, L: prop(*m["category_level_counts"]["L1"])),
    ("Category correct at L2", "accuracy", "queries whose gold category is at depth ≥ 2; predicted path has the gold L2 node",
     lambda m, L: prop(*m["category_level_counts"]["L2"])),
    ("Category correct at L3", "accuracy", "queries whose gold category is at depth 3; predicted node is the gold L3 node",
     lambda m, L: prop(*m["category_level_counts"]["L3"])),
    ("'No category' precision", "precision", "queries predicted as 'no category'",
     lambda m, L: prop(m["no_category_counts"]["tp"], m["no_category_counts"]["tp"] + m["no_category_counts"]["fp"])),
    ("'No category' recall", "recall", "queries whose gold is 'no category'",
     lambda m, L: prop(m["no_category_counts"]["tp"], m["no_category_counts"]["tp"] + m["no_category_counts"]["fn"])),
    ("Filters precision", "precision (micro)", "predicted attribute=value pairs, pooled over all queries",
     lambda m, L: prop(m["filters_counts"]["tp"], m["filters_counts"]["tp"] + m["filters_counts"]["fp"])),
    ("Filters recall", "recall (micro)", "gold attribute=value pairs, pooled over all queries",
     lambda m, L: prop(m["filters_counts"]["tp"], m["filters_counts"]["tp"] + m["filters_counts"]["fn"])),
    ("Filters F1", "F1 (micro)", "harmonic mean of the two rows above", lambda m, L: f1(m["filters_counts"])),
    ("Filter set exact match", "accuracy", "all queries; predicted filter set equals the gold set (incl. both empty)",
     lambda m, L: prop(*m["filters_exact_set_counts"])),
    ("Word-role accuracy", "accuracy", "all words of all queries; role category / filter / residual",
     lambda m, L: prop(*m["token_counts"])),
    ("Residual words precision", "precision", "words predicted 'residual'",
     lambda m, L: prop(m["residual_counts"]["tp"], m["residual_counts"]["tp"] + m["residual_counts"]["fp"])),
    ("Residual words recall", "recall", "words whose gold role is 'residual'",
     lambda m, L: prop(m["residual_counts"]["tp"], m["residual_counts"]["tp"] + m["residual_counts"]["fn"])),
    ("Residual words F1", "F1", "harmonic mean of the two rows above", lambda m, L: f1(m["residual_counts"])),
    ("**Whole query exactly right**", "accuracy", "all queries; category exact, filter set exact, and the set of "
     "residual word positions exact", lambda m, L: prop(*m["full_query_exact_counts"])),
    ("Latency p50 / p95", "seconds", "all queries; one request each, sequential, warm model",
     lambda m, L: "%.1f s / %.1f s" % (L["p50_s"], L["p95_s"])),
    ("Server prefill / field evaluation", "mean seconds", "mlx backend only (nimble's own timings)",
     lambda m, L: "%.2f s / %.2f s" % (L["mean_prefill_s"], L["mean_field_eval_s"]) if "mean_prefill_s" in L else "n/a"),
    ("Questions / input tokens", "mean per query", "server-reported input tokens (Ollama: every question charged the "
     "full prompt; mlx: tokens actually processed)",
     lambda m, L: "%.1f / %.0fk" % (L["mean_questions"], (L["mean_input_tokens"] or 0) / 1000)),
]


def _prop(k, n):
    lo, hi = wilson(k, n)
    return {"value": k / n if n else None, "correct": k, "counted": n,
            "ci95": [lo, hi] if n else None}


def _prf(c):
    p, r, f = E.prf(c["tp"], c["fp"], c["fn"])
    return {"precision": _prop(c["tp"], c["tp"] + c["fp"]), "recall": _prop(c["tp"], c["tp"] + c["fn"]),
            "f1": f, "counts": dict(c)}


def metrics_json(m, L):
    """Machine-readable form of the table rows: proportions carry counts and 95% Wilson intervals."""
    return {
        "category_exact": _prop(*m["category_exact_counts"]),
        "category_level": {lvl: _prop(*kn) for lvl, kn in m["category_level_counts"].items()},
        "no_category": _prf(m["no_category_counts"]),
        "filters_micro": _prf(m["filters_counts"]),
        "filter_set_exact": _prop(*m["filters_exact_set_counts"]),
        "word_role_accuracy": _prop(*m["token_counts"]),
        "word_role_confusion": m["token_confusion"],
        "residual_words": _prf(m["residual_counts"]),
        "whole_query_exact": _prop(*m["full_query_exact_counts"]),
        "per_attribute_filters": m["filters_per_attr"],
        "latency": {k: v for k, v in L.items() if k != "p50_by_query_length"},
    }


def build_table(split, runs, data, strict=False):
    runs = [canonical_run(r) for r in runs]
    gold = E.load_rows(data, "all")
    g = {r["id"]: r for r in gold}
    split_ids = {r["id"] for r in E.load_rows(data, split)}
    loaded = []
    for run in runs:
        scheme, backend, model = parse_run(run)
        path = E.pred_path(split, model, scheme, backend)
        preds = [p for p in E.load_preds(path, g) if p["id"] in split_ids] if path.exists() else []
        loaded.append((run, preds))
    missing = [run for run, preds in loaded if not preds]
    if missing and strict:
        raise SidmError("no %s predictions for %s" % (split, ", ".join(missing)),
                        "make %s BACKEND=<backend> MODEL=<model> for each, or drop --strict to skip them" % split)
    for run in missing:
        _, backend, model = parse_run(run)
        print("warning: skipping %s: no %s predictions (create with: make %s BACKEND=%s MODEL=%s)"
              % (run, split, split, backend, model), file=sys.stderr)
    loaded = [(run, preds) for run, preds in loaded if preds]
    if not loaded:
        raise SidmError("none of the requested runs has %s predictions" % split, "make runs  (shows what exists)")
    ids = set.intersection(*({p["id"] for p in preds} for _, preds in loaded))
    cols = []
    for run, preds in loaded:
        preds = [p for p in preds if p["id"] in ids]
        cols.append((run, E.compute(gold, preds), E.latency_stats(preds)))

    served = {run: sorted({p["pred"].get("served_model") or "not recorded" for p in preds}) for run, preds in loaded}

    def describe(run):
        scheme, backend, model = parse_run(run)
        cfg, sc = tuned_settings(backend, SCHEMES[scheme], model)
        return ("`%s`: %s; model `%s` (served as %s); category scheme %s, 'no category' threshold %.2f; "
                "filter p ≥ %.2f; word-role `category` weight ×%g" % (
                    run, BACKEND_INFO[backend], model, ", ".join("`%s`" % m for m in served[run]),
                    "top-level `cat_L1` + conditional group questions" if sc.router
                    else "group questions with an `other_product` escape, no `cat_L1`",
                    sc.none_thr, cfg["filter_min_prob"], cfg["word_weights"]["category"]))

    lines = ["# Results: %s split" % split, "",
             "- **Queries:** %d, evaluated by every run. The %s split has %d rows in `%s`." % (
                 len(ids), split, len(split_ids), data),
             "- **Decoding:** tuned per backend on the 100 dev rows (ids 0–99). Filters are masked by the "
             "predicted category's applicable attributes.",
             "- **Runs:**"] + ["  - " + describe(run) for run, _ in loaded] + [
             "- **Cells:** value (correct / counted) [95% Wilson interval]. F1 has no interval.", "",
             "| Metric | Measure | Counted over | " + " | ".join("`%s`" % c[0] for c in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for label, measure, over, cell in SPEC:
        lines.append("| %s | %s | %s | %s |" % (label, measure, over, " | ".join(cell(m, L) for _, m, L in cols)))

    result = {"split": split, "data": data, "n_queries": len(ids), "split_rows": len(split_ids),
              "metric_definitions": {label: {"measure": measure, "counted_over": over} for label, measure, over, _ in SPEC},
              "runs": [], "paired_mcnemar": []}
    for run, m, L in cols:
        scheme, backend, model = parse_run(run)
        cfg, sc = tuned_settings(backend, SCHEMES[scheme], model)
        result["runs"].append({
            "run": run, "scheme": scheme, "backend": backend, "model": model, "served_models": served[run],
            "backend_info": BACKEND_INFO[backend],
            "decoding": {"none_thr": sc.none_thr, "filter_min_prob": cfg["filter_min_prob"],
                         "word_weights": cfg["word_weights"]},
            "metrics": metrics_json(m, L)})

    if len(cols) > 1:
        order = sorted(ids)
        metrics = (("category", "Category, exact node"), ("full", "Whole query exactly right"))

        def test(m_a, m_b, key):
            return mcnemar_p([m_a["per_query"][i][key] for i in order], [m_b["per_query"][i][key] for i in order])

        # every pair (a listed before b), both metrics -> JSON
        tests = {}
        for i, (run_a, m_a, _) in enumerate(cols):
            for run_b, m_b, _ in cols[i + 1:]:
                for key, _ in metrics:
                    only_a, only_b, p = tests[run_a, run_b, key] = test(m_a, m_b, key)
                    result["paired_mcnemar"].append({"reference": run_a, "run": run_b, "metric": key,
                                                     "correct_only_reference": only_a, "correct_only_run": only_b,
                                                     "p_value": p})

        ref = cols[0][0]
        lines += ["", "## Paired comparison against `%s` (exact McNemar test, same queries)" % ref, "",
                  "| Run | Metric | correct only with `%s` | correct only with the other | p-value |" % ref,
                  "|---|---|---|---|---|"]
        for run, _, _ in cols[1:]:
            for key, label in metrics:
                only_ref, only_other, p = tests[ref, run, key]
                lines.append("| `%s` | %s | %d | %d | %.4f |" % (run, label, only_ref, only_other, p))

        names = [run for run, _, _ in cols]
        for key, label in metrics:
            lines += ["", "## All pairs: %s (exact McNemar test)" % label, "",
                      "Cell (row, column) = queries correct only with the row run : only with the column run, "
                      "and the p-value. **Bold** = significant at p < 0.05; the better run of the pair is the one "
                      "with the larger count.", "",
                      "| | " + " | ".join("`%s`" % n for n in names[1:]) + " |",
                      "|---|" + "---|" * (len(names) - 1)]
            for i, row in enumerate(names[:-1]):
                cells = []
                for j, col in enumerate(names[1:], 1):
                    if j <= i:
                        cells.append("")
                        continue
                    a, b, p = tests[row, col, key]
                    cell = "%d:%d p=%s" % (a, b, "<0.0001" if p < 1e-4 else "%.4f" % p)
                    cells.append("**%s**" % cell if p < 0.05 else cell)
                lines.append("| `%s` | %s |" % (row, " | ".join(cells)))
    return "\n".join(lines) + "\n", result


def show_tests(json_path, metric=None, significant_only=False, alpha=0.05):
    """Print the pairwise McNemar tests stored in a results JSON, one readable line each."""
    result = json.loads(Path(json_path).read_text())
    tests = [t for t in result["paired_mcnemar"] if metric in (None, t["metric"])]
    width = max(len(r["run"]) for r in result["runs"])
    shown = [t for t in tests if not significant_only or t["p_value"] < alpha]
    print("%s split, %d queries; %d tests%s" % (result["split"], result["n_queries"], len(tests),
                                              ", showing the %d with p < %g" % (len(shown), alpha) if significant_only else ""))
    for t in tests:
        a, b, p = t["correct_only_reference"], t["correct_only_run"], t["p_value"]
        if significant_only and p >= alpha:
            continue
        verdict = ("no significant difference" if p >= alpha
                   else "%s better" % (t["reference"] if a > b else t["run"]))
        print("%-8s %-*s vs %-*s %4d:%-4d p=%-8s %s" % (
            t["metric"], width, t["reference"], width, t["run"], a, b,
            "<0.0001" if p < 1e-4 else "%.4f" % p, verdict))


@friendly_main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="<scheme>@<backend>:<model> columns (default: those of %s with prediction files)" % DEFAULT_RUNS)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="use the registry's run list for the shipped tables (src/sidm/runs.py)")
    ap.add_argument("--strict", action="store_true", help="fail if a requested run has no predictions (default: skip it)")
    ap.add_argument("--overwrite", action="store_true",
                    help="with --run: delete each run's existing predictions for the split and run it from scratch")
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--data", default="data/eval_raw.jsonl")
    ap.add_argument("--run", action="store_true", help="run/resume the parser on missing rows first (~16 s per query on Ollama)")
    ap.add_argument("--limit", type=int, default=0, help="with --run: only the first N rows of the split (e.g. a paid-API trial)")
    ap.add_argument("--md", default=None, help="output file (default results/results_<split>.md)")
    ap.add_argument("--show-tests", metavar="JSON", default=None,
                    help="print the pairwise McNemar tests of an existing results JSON and exit")
    ap.add_argument("--metric", choices=["category", "full"], default=None, help="with --show-tests: one metric only")
    ap.add_argument("--significant", action="store_true", help="with --show-tests: only p < 0.05")
    args = ap.parse_args()
    if args.show_tests:
        return show_tests(args.show_tests, args.metric, args.significant)
    runs = args.runs or (PRESETS[args.preset] if args.preset else None) or [r for r in DEFAULT_RUNS
                         if E.pred_path(args.split, parse_run(r)[2], parse_run(r)[0], parse_run(r)[1]).exists()]
    if args.run:
        for run in runs:
            scheme, backend, model = parse_run(run)
            out = E.pred_path(args.split, model, scheme, backend)
            print("running %s -> %s" % (run, out), flush=True)
            E.run(SimpleNamespace(data=args.data, split=args.split, limit=args.limit, model=model, scheme=scheme,
                                  backend=backend, out=str(out), overwrite=args.overwrite))
    text, result = build_table(args.split, runs, args.data, strict=args.strict)
    out = Path(args.md or "results/results_%s.md" % args.split)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    print(text)
    print("written to %s and %s" % (out, out.with_suffix(".json")))


if __name__ == "__main__":
    main()
