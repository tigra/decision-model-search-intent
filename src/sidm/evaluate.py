"""Run the decision-model parser over the dataset and report accuracy + latency.

Usage:
  python -m sidm.evaluate run --split eval --scheme router --out results/eval_nimble_router.jsonl
  python -m sidm.evaluate report --pred results/eval_nimble_router.jsonl --md results/report_router.md
  python -m sidm.evaluate tune --pred results/dev_nimble_embedded.jsonl        # sweep the 'no category' threshold
  python -m sidm.evaluate compare --pred results/eval_nimble_router.jsonl results/eval_nimble_embedded.jsonl

Predictions keep the raw answers, so report/tune/compare re-decode them with the current decoding code.
"""
import dataclasses
import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from sidm import schema as S
from sidm.ollama_client import DEFAULT_MODELS, friendly_main
from sidm.parser import SCHEMES, DEFAULT_SCHEME, TUNED_FILE, decode, override_tuning, parse, tuned_settings
from sidm.score import query_parts, query_score, summarize

DEV_IDS = range(0, 100)


def load_rows(path, split):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    if split == "dev":
        return [r for r in rows if r["id"] in DEV_IDS]
    if split == "eval":
        return [r for r in rows if r["id"] not in DEV_IDS][:1000]
    return rows


def run(args):
    rows = load_rows(args.data, args.split)
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, "overwrite", False) and out_path.exists():
        n_old = sum(1 for line in open(out_path) if line.strip())
        out_path.unlink()
        print("%s: overwrite: deleted %d stored predictions, running from scratch" % (out_path, n_old), flush=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    if all(r["id"] in done for r in rows):
        print("%s: all %d rows already done (use OVERWRITE=1 / --overwrite to re-run from scratch)"
              % (out_path, len(rows)))
        return
    # warm-up (model load) - not timed
    parse("warm up sofa", model=args.model, scheme=args.scheme, backend=args.backend)
    parse("grey oak dining table for 6 cheap", model=args.model, scheme=args.scheme, backend=args.backend)
    t0 = time.time()
    with open(out_path, "a") as f:
        for k, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            p = parse(r["query"], model=args.model, scheme=args.scheme, backend=args.backend)
            f.write(json.dumps({"id": r["id"], "query": r["query"], "pred": p}) + "\n")
            f.flush()
            if k % 50 == 0:
                print("%d/%d  %.1fs elapsed, last latency %.3fs" % (k, len(rows), time.time() - t0, p["latency_s"]),
                      flush=True)


# ---------------------------------------------------------------- metrics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def path_of(n):
    return S.NODES[n].path() if n else []


def compute(gold_rows, preds, cat_key="category", filt_key="filters"):
    m = {}
    n = len(preds)
    g = {r["id"]: r for r in gold_rows}
    # category
    exact = lvl_hits = lvl_tot = 0
    hp = hr = 0.0
    per_level = defaultdict(lambda: [0, 0])
    confusions = Counter()
    for p in preds:
        gr = g[p["id"]]
        gc, pc = gr["category"], p["pred"][cat_key]
        exact += gc == pc
        gpath, ppath = path_of(gc), path_of(pc)
        for lvl in range(3):
            if lvl < len(gpath):
                per_level[lvl + 1][1] += 1
                per_level[lvl + 1][0] += len(ppath) > lvl and ppath[lvl] == gpath[lvl]
        inter = len(set(gpath) & set(ppath))
        if gpath or ppath:
            hp += inter / len(ppath) if ppath else (1.0 if not gpath else 0.0)
            hr += inter / len(gpath) if gpath else (1.0 if not ppath else 0.0)
        else:
            hp += 1
            hr += 1
        if gc != pc:
            confusions[(gc, pc)] += 1
    none_tp = sum(g[p["id"]]["category"] is None and p["pred"][cat_key] is None for p in preds)
    none_fp = sum(g[p["id"]]["category"] is not None and p["pred"][cat_key] is None for p in preds)
    none_fn = sum(g[p["id"]]["category"] is None and p["pred"][cat_key] is not None for p in preds)
    m["no_category_PRF"] = prf(none_tp, none_fp, none_fn)
    m["no_category_counts"] = {"tp": none_tp, "fp": none_fp, "fn": none_fn}
    m["n_queries"] = n
    m["category_exact_acc"] = exact / n
    m["category_exact_counts"] = [exact, n]
    m["category_level_acc"] = {"L%d" % k: v[0] / v[1] for k, v in sorted(per_level.items()) if v[1]}
    m["category_level_counts"] = {"L%d" % k: list(v) for k, v in sorted(per_level.items()) if v[1]}
    hP, hR = hp / n, hr / n
    m["category_hier_P"], m["category_hier_R"] = hP, hR
    m["category_hier_F1"] = 2 * hP * hR / (hP + hR) if hP + hR else 0
    m["category_top_confusions"] = [(str(a), str(b), c) for (a, b), c in confusions.most_common(15)]

    # filters
    tp = fp = fn = 0
    per_attr = defaultdict(lambda: [0, 0, 0])
    exact_f = 0
    for p in preds:
        gf = g[p["id"]]["filters"]
        pf = p["pred"][filt_key]
        gs, ps = set(gf.items()), set(pf.items())
        exact_f += gs == ps
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        for a in S.ATTRIBUTES:
            gv, pv = gf.get(a), pf.get(a)
            if gv is not None and pv == gv:
                per_attr[a][0] += 1
            else:
                if pv is not None:
                    per_attr[a][1] += 1
                if gv is not None:
                    per_attr[a][2] += 1
    m["filters_micro_PRF"] = prf(tp, fp, fn)
    m["filters_counts"] = {"tp": tp, "fp": fp, "fn": fn}
    m["filters_exact_set_acc"] = exact_f / n
    m["filters_exact_set_counts"] = [exact_f, n]
    m["filters_per_attr"] = {a: dict(zip(["P", "R", "F1"], prf(*v)), support=v[0] + v[2])
                             for a, v in per_attr.items()}

    # tokens
    classes = ["category", "filter", "residual"]
    conf = Counter()
    tok_ok = tok_n = 0
    res_exact = 0
    for p in preds:
        gl, pl = g[p["id"]]["token_labels"], p["pred"]["token_labels"]
        for a, b in zip(gl, pl):
            conf[(a, b)] += 1
            tok_ok += a == b
            tok_n += 1
        gres = [i for i, l in enumerate(gl) if l == "residual"]
        pres = [i for i, l in enumerate(pl) if l == "residual"]
        res_exact += gres == pres
    m["token_acc"] = tok_ok / tok_n
    m["token_counts"] = [tok_ok, tok_n]
    per_cls = {}
    for c in classes:
        t = conf[(c, c)]
        f_p = sum(v for (a, b), v in conf.items() if b == c and a != c)
        f_n = sum(v for (a, b), v in conf.items() if a == c and b != c)
        per_cls[c] = dict(zip(["P", "R", "F1"], prf(t, f_p, f_n)))
    m["token_per_class"] = per_cls
    m["token_macro_F1"] = sum(v["F1"] for v in per_cls.values()) / len(classes)
    m["token_confusion"] = {"%s->%s" % k: v for k, v in sorted(conf.items())}
    m["residual_word_PRF"] = (per_cls["residual"]["P"], per_cls["residual"]["R"], per_cls["residual"]["F1"])
    m["residual_counts"] = {"tp": conf[("residual", "residual")],
                            "fp": sum(v for (a, b), v in conf.items() if b == "residual" and a != "residual"),
                            "fn": sum(v for (a, b), v in conf.items() if a == "residual" and b != "residual")}
    m["residual_set_exact_acc"] = res_exact / n

    # whole query (+ per-query correctness, used for paired significance tests between runs)
    full = 0
    m["per_query"] = {}
    for p in preds:
        gr = g[p["id"]]
        gres = [i for i, l in enumerate(gr["token_labels"]) if l == "residual"]
        pres = [i for i, l in enumerate(p["pred"]["token_labels"]) if l == "residual"]
        cat_ok = gr["category"] == p["pred"][cat_key]
        ok = cat_ok and set(gr["filters"].items()) == set(p["pred"][filt_key].items()) and gres == pres
        full += ok
        parts = query_parts(gr, p["pred"], cat_key, filt_key)
        m["per_query"][p["id"]] = {"category": cat_ok, "full": ok, "score": query_score(parts), "score_parts": parts}
    m["full_query_exact_acc"] = full / n
    m["full_query_exact_counts"] = [full, n]
    # the forgiving whole-query metric (sidm/score.py)
    m["weighted_score"] = summarize([q["score_parts"] for q in m["per_query"].values()])
    return m


def server_timing_stats(preds):
    """Mean server-side prefill / batched field-evaluation seconds (mlx backend only)."""
    sm = [p["pred"]["server_metrics"] for p in preds if p["pred"].get("server_metrics")]
    if not sm:
        return {}
    return {"mean_prefill_s": statistics.mean(m["prefill_seconds"] for m in sm),
            "mean_field_eval_s": statistics.mean(m["field_evaluation_seconds"] for m in sm),
            "mean_prefix_tokens": statistics.mean(m["prefix_tokens"] for m in sm),
            "peak_gib": max(m.get("mlx_peak_active_gib", 0) for m in sm)}


def latency_stats(preds):
    lat = [p["pred"]["latency_s"] for p in preds]
    by_nq = defaultdict(list)
    for p in preds:
        nw = p["pred"].get("n_word_questions", p["pred"]["n_questions"] - 19)  # old router files: 7 cat + 12 filter
        by_nq["%d-%d words" % ((nw - 1) // 3 * 3 + 1, (nw - 1) // 3 * 3 + 3)].append(p["pred"]["latency_s"])
    toks = [p["pred"]["usage"]["input_tokens"] for p in preds if p["pred"].get("usage")]
    return {
        "n": len(lat), "mean_s": statistics.mean(lat), "p50_s": pct(lat, .5), "p90_s": pct(lat, .9),
        "p95_s": pct(lat, .95), "p99_s": pct(lat, .99), "min_s": min(lat), "max_s": max(lat),
        "queries_per_s": len(lat) / sum(lat),
        "mean_questions": statistics.mean(p["pred"]["n_questions"] for p in preds),
        "ms_per_question": 1000 * sum(lat) / sum(p["pred"]["n_questions"] for p in preds),
        "mean_input_tokens": statistics.mean(toks) if toks else None,
        "p50_by_query_length": {k: (len(v), pct(v, .5)) for k, v in sorted(by_nq.items(), key=lambda kv: int(kv[0].split("-")[0]))},
        "truncated_queries": sum(p["pred"]["truncated"] for p in preds),
        **server_timing_stats(preds),
    }


KEEP = ("scheme", "backend", "model", "served_model", "latency_s", "n_requests", "n_questions", "n_word_questions", "usage", "raw_answers", "server_metrics")


def pred_path(split, model, scheme, backend="ollama"):
    """results/<split>_<label>_<scheme>.jsonl; label = model for Ollama (original names), else <backend>-<model>
    (just the model when it already starts with the backend name, e.g. jev-latest). Colons in model tags become
    '-' so file names work on every OS: tev1:0.8b -> results/dev_tev1-0.8b_embedded.jsonl."""
    return Path("results/%s_%s_%s.jsonl" % (split, run_label(model, backend), scheme))


def run_label(model, backend="ollama"):
    """The model part of result file names (see pred_path): nimble, tev1-0.8b, ollaya-jeb-4b, jev-latest."""
    model = model or DEFAULT_MODELS[backend]
    label = model if backend == "ollama" or model.startswith(backend) else "%s-%s" % (backend, model)
    return label.replace(":", "-")


def load_preds(path, gold_by_id, scheme=None, tuned=True):
    """Read a prediction file and re-decode every row from its raw answers with its own scheme
    (files written before schemes existed are 'router'). `scheme` overrides it, e.g. with a changed none_thr."""
    preds = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for p in preds:
        keep = {k: p["pred"][k] for k in KEEP if k in p["pred"]}
        sc = scheme or SCHEMES[keep.setdefault("scheme", "router")]
        backend = keep.setdefault("backend", "ollama")
        p["pred"] = dict(decode({"answers": keep["raw_answers"]}, gold_by_id[p["id"]]["tokens"], sc, tuned, backend,
                                keep.get("model")), **keep)
    return preds


def report(args):
    gold = load_rows(args.data, "all")
    g = {r["id"]: r for r in gold}
    preds = load_preds(args.pred, g)
    untuned = load_preds(args.pred, g, tuned=False)
    scheme = preds[0]["pred"]["scheme"]
    res = {
        "scheme": scheme,
        "greedy_masked": compute(gold, preds),
        "greedy_raw_filters": compute(gold, preds, filt_key="filters_raw"),
        "untuned": compute(gold, untuned),
        "latency": latency_stats(preds),
    }
    for k in ("greedy_masked", "greedy_raw_filters", "untuned"):
        res[k].pop("per_query")
    Path(args.pred).with_suffix(".metrics.json").write_text(json.dumps(res, indent=1))
    m, L = res["greedy_masked"], res["latency"]
    first = preds[0]["pred"]
    label = "%s@%s:%s" % (scheme, first.get("backend", "ollama"), first.get("model") or args.model)
    md = ["# Search intent parsing: `%s` via /v1/systemone" % label, "",
          "%d queries, one request per query (%.1f questions on average)." % (len(preds), L["mean_questions"]), "",
          "## Accuracy", "",
          "| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |",
          "|---|---|---|---|"]

    def row(name, f):
        md.append("| %s | %s | %s | %s |" % (name, f(res["greedy_masked"]), f(res["greedy_raw_filters"]), f(res["untuned"])))

    row("Category exact node acc", lambda x: "%.3f" % x["category_exact_acc"])
    for lvl in ["L1", "L2", "L3"]:
        row("Category %s acc (given gold depth ≥ %s)" % (lvl, lvl[1]), lambda x, l=lvl: "%.3f" % x["category_level_acc"].get(l, float("nan")))
    row("Category hierarchical F1", lambda x: "%.3f" % x["category_hier_F1"])
    row("'No category' P / R (tp/fp/fn)", lambda x: "%.2f / %.2f (%d/%d/%d)" % (
        x["no_category_PRF"][0], x["no_category_PRF"][1], *[x["no_category_counts"][k] for k in ("tp", "fp", "fn")]))
    row("Filters micro P / R / F1", lambda x: "%.3f / %.3f / %.3f" % tuple(x["filters_micro_PRF"]))
    row("Filter set exact match", lambda x: "%.3f" % x["filters_exact_set_acc"])
    row("Word-role accuracy", lambda x: "%.3f" % x["token_acc"])
    row("Word-role macro F1", lambda x: "%.3f" % x["token_macro_F1"])
    row("Residual words P / R / F1", lambda x: "%.3f / %.3f / %.3f" % tuple(x["residual_word_PRF"]))
    row("Residual set exact match", lambda x: "%.3f" % x["residual_set_exact_acc"])
    row("**Full query exact match**", lambda x: "**%.3f**" % x["full_query_exact_acc"])

    md += ["", "### Per-attribute filter scores (masked)", "", "| attribute | support | P | R | F1 |", "|---|---|---|---|---|"]
    for a, v in sorted(m["filters_per_attr"].items(), key=lambda kv: -kv[1]["support"]):
        md.append("| %s | %d | %.3f | %.3f | %.3f |" % (a, v["support"], v["P"], v["R"], v["F1"]))
    md += ["", "### Word roles (gold → predicted)", "", "| gold \\ pred | category | filter | residual |", "|---|---|---|---|"]
    for a in ["category", "filter", "residual"]:
        md.append("| %s | %s |" % (a, " | ".join(str(m["token_confusion"].get("%s->%s" % (a, b), 0))
                                               for b in ["category", "filter", "residual"])))
    md += ["", "### Top category confusions (gold → predicted)", ""]
    md += ["- %s → %s: %d" % c for c in m["category_top_confusions"]]
    md += ["", "## Latency (sequential, warm model)", "",
           "| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |",
           "|---|---|---|---|---|---|---|---|---|",
           "| %.3fs | %.3fs | %.3fs | %.3fs | %.3fs | %.3fs | %.2f | %.1f | %s |" % (
               L["mean_s"], L["p50_s"], L["p90_s"], L["p95_s"], L["p99_s"], L["max_s"], L["queries_per_s"],
               L["ms_per_question"], "%.0f" % L["mean_input_tokens"] if L["mean_input_tokens"] else "n/a"),
           "", "p50 latency by query length: " + ", ".join("%s: %.3fs (n=%d)" % (k, v[1], v[0]) for k, v in L["p50_by_query_length"].items()),
           "", "Truncated queries (more than the word-question budget): %d" % L["truncated_queries"], ""]

    # worst examples
    def nerr(p):
        gr = g[p["id"]]
        e = (gr["category"] != p["pred"]["category"]) * 2
        e += len(set(gr["filters"].items()) ^ set(p["pred"]["filters"].items()))
        e += sum(a != b for a, b in zip(gr["token_labels"], p["pred"]["token_labels"]))
        return e
    worst = sorted(preds, key=nerr, reverse=True)[:20]
    md += ["## 20 worst examples", ""]
    for p in worst:
        gr = g[p["id"]]
        md.append("- `%s`  \n  gold: cat=%s filters=%s roles=%s  \n  pred: cat=%s filters=%s roles=%s" % (
            gr["query"], gr["category"], gr["filters"], "".join(l[0] for l in gr["token_labels"]),
            p["pred"]["category"], p["pred"]["filters"], "".join(l[0] for l in p["pred"]["token_labels"])))
    Path(args.md).write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))


def tune(args):
    """Sweep decoding settings over saved answers, one at a time with the others fixed (use dev, not eval).

    Settings: the scheme's 'no category' threshold, the filter probability threshold, and the word-role
    `category` weight. With --apply, coordinate ascent on whole-query exact match (ties keep the current
    value) and the result is saved under "<backend>:<model>" in results/tuned_settings.json.
    """
    gold = load_rows(args.data, "all")
    g = {r["id"]: r for r in gold}
    first = load_preds(args.pred, g)[0]["pred"]
    backend, model, base = first["backend"], first.get("model"), SCHEMES[first["scheme"]]
    key = "%s:%s" % (backend, model) if model else backend
    cfg, sc = tuned_settings(backend, base, model)
    current = {"none_thr": sc.none_thr, "filter_min_prob": cfg["filter_min_prob"],
               "category_weight": cfg["word_weights"]["category"]}
    print("%s, scheme %s; current: %s" % (key, base.name, current))

    def settings(v):
        return {"none_thr": {base.name: v["none_thr"]}, "filter_min_prob": v["filter_min_prob"],
                "word_weights": dict(cfg["word_weights"], category=v["category_weight"])}

    def metrics(v):
        with override_tuning(key, **settings(v)):
            return compute(gold, load_preds(args.pred, g))

    grids = {"none_thr": args.thresholds, "filter_min_prob": args.filter_thresholds,
             "category_weight": args.category_weights}
    best = dict(current)
    for _ in range(2 if args.apply else 1):  # two coordinate-ascent passes when applying
        for name, grid in grids.items():
            print("\n%-16s category  filters F1  word_acc  full_exact" % name)
            scores = []
            for x in grid:
                m = metrics(dict(best, **{name: x}))
                scores.append((m["full_query_exact_acc"], x))
                print("  %-14g %.3f     %.3f       %.3f     %.3f" % (x, m["category_exact_acc"],
                      m["filters_micro_PRF"][2], m["token_acc"], m["full_query_exact_acc"]))
            if args.apply:
                top = max(s for s, _ in scores)
                tied = [x for s, x in scores if s == top]
                best[name] = best[name] if best[name] in tied else tied[0]
    if args.apply:
        saved = json.loads(TUNED_FILE.read_text()) if TUNED_FILE.exists() else {}
        saved[key] = settings(best)
        TUNED_FILE.write_text(json.dumps(saved, indent=1, sort_keys=True) + "\n")
        m = metrics(best)
        print("\napplied %s -> %s (dev full exact %.3f, category %.3f)" % (key, best, m["full_query_exact_acc"],
                                                                         m["category_exact_acc"]))


COMPARE_ROWS = [
    ("Category exact node acc", lambda m, L: "%.3f" % m["category_exact_acc"]),
    ("Category L1 / L2 / L3 acc", lambda m, L: " / ".join("%.3f" % m["category_level_acc"].get(l, float("nan")) for l in ("L1", "L2", "L3"))),
    ("'No category' P / R", lambda m, L: "%.2f / %.2f" % m["no_category_PRF"][:2]),
    ("Filters micro F1", lambda m, L: "%.3f" % m["filters_micro_PRF"][2]),
    ("Filter set exact match", lambda m, L: "%.3f" % m["filters_exact_set_acc"]),
    ("Word-role accuracy", lambda m, L: "%.3f" % m["token_acc"]),
    ("Residual words F1", lambda m, L: "%.3f" % m["residual_word_PRF"][2]),
    ("**Full query exact match**", lambda m, L: "**%.3f**" % m["full_query_exact_acc"]),
    ("Latency p50 / p95", lambda m, L: "%.1fs / %.1fs" % (L["p50_s"], L["p95_s"])),
    ("Questions / input tokens per query", lambda m, L: "%.1f / %.0fk" % (L["mean_questions"], (L["mean_input_tokens"] or 0) / 1000)),
]


def compare(args):
    """Side-by-side metrics (tuned decoding) for several prediction files over the same rows."""
    gold = load_rows(args.data, "all")
    g = {r["id"]: r for r in gold}
    runs = [(Path(f).stem, load_preds(f, g)) for f in args.pred]
    ids = set.intersection(*({p["id"] for p in preds} for _, preds in runs))
    cols = []
    for name, preds in runs:
        preds = [p for p in preds if p["id"] in ids]
        cols.append(("%s (%s)" % (name, preds[0]["pred"]["scheme"]), compute(gold, preds), latency_stats(preds)))
    md = ["%d common queries" % len(ids), "", "| Metric | " + " | ".join(c[0] for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    md += ["| %s | %s |" % (label, " | ".join(f(m, L) for _, m, L in cols)) for label, f in COMPARE_ROWS]
    text = "\n".join(md)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


@friendly_main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    data = "data/eval_raw.jsonl"
    r = sub.add_parser("run")
    r.add_argument("--data", default=data)
    r.add_argument("--split", default="eval", choices=["dev", "eval", "all"])
    r.add_argument("--model", default=None, help="default per backend: %s" % DEFAULT_MODELS)
    r.add_argument("--scheme", default=DEFAULT_SCHEME, choices=sorted(SCHEMES))
    r.add_argument("--backend", default="ollama", choices=sorted(DEFAULT_MODELS))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--overwrite", action="store_true", help="delete existing predictions in --out and run from scratch")
    r.set_defaults(fn=run)
    rp = sub.add_parser("report")
    rp.add_argument("--data", default=data)
    rp.add_argument("--pred", required=True)
    rp.add_argument("--model", default="nimble")
    rp.add_argument("--md", required=True)
    rp.set_defaults(fn=report)
    t = sub.add_parser("tune")
    t.add_argument("--data", default=data)
    t.add_argument("--pred", required=True)
    t.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    t.add_argument("--filter-thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    t.add_argument("--category-weights", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    t.add_argument("--apply", action="store_true", help="pick the best values and save them for this backend:model")
    t.set_defaults(fn=tune)
    c = sub.add_parser("compare")
    c.add_argument("--data", default=data)
    c.add_argument("--pred", nargs="+", required=True)
    c.add_argument("--md", default=None)
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
