"""Figures for the README, written as deterministic SVGs to docs/figures/ (make figures).

Every figure is computed from shipped files only (predictions, result JSONs, benchmark files, the dataset);
no model is called. Needs the optional `figures` dependency group: uv run --group figures python -m sidm.figures

Style: explicit white background; one fixed color per eval run (a validated categorical order) used in every
figure, plus marker shapes / direct labels as the secondary channel; hairline recessive axes.
"""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from sidm import evaluate as E  # noqa: E402
from sidm import schema as S  # noqa: E402
from sidm.ollama_client import BACKENDS, friendly_main  # noqa: E402
from sidm.results_table import wilson  # noqa: E402
from sidm.runs import DEV_RUNS, EVAL_RUNS, split_run  # noqa: E402

OUT = Path("docs/figures")
DATA = "data/eval_raw.jsonl"

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#f0efec"
BAND = "#f5f4f0"  # alternating row background, lighter than the grid
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # validated on #ffffff
BLUE_RAMP = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED_POLE, BLUE_POLE = "#e34948", "#2a78d6"

NAMES = {
    "embedded@ollama:nimble": "nimble (embedded)",
    "router@ollama:nimble": "nimble (router)",
    "embedded@ollama:tev1": "tev1 4B",
    "embedded@ollama:tev1:0.8b": "tev1 0.8B",
    "embedded@ollaya:jeb:4b": "jeb:4b",
    "embedded@ollaya:winnow:e4b": "winnow:e4b",
    "embedded@decider:strands-decider-2b": "Strands Decider 2B",
    "embedded@mlx:nimble": "nimble (MLX)",
    "embedded@jev:jev-latest": "Jev (hosted)",
    "embedded@openai:gpt-6-luna": "OpenAI (hosted)",
}
# Fixed per run, never by rank (validated palette slots on white). nimble router, an ablation of nimble, is the dark
# neutral. Jev (red) and OpenAI (violet) are the pair compared most, so they get two slots that separate well
# (red/orange fail the normal-vision floor, ΔE 7); red, violet and nimble's blue pass all-pairs. Strands Decider takes
# the orange. Runs not listed (e.g. MLX nimble in the sweep) are dark neutral too.
COLOR = {
    "embedded@ollama:nimble": PALETTE[0],
    "embedded@openai:gpt-6-luna": PALETTE[6],
    "embedded@ollama:tev1": PALETTE[2],
    "embedded@ollama:tev1:0.8b": PALETTE[3],
    "embedded@ollaya:jeb:4b": PALETTE[4],
    "embedded@ollaya:winnow:e4b": PALETTE[5],
    "embedded@decider:strands-decider-2b": PALETTE[1],
    "embedded@jev:jev-latest": PALETTE[7],
    "router@ollama:nimble": INK2,
}
COLOR = defaultdict(lambda: INK2, COLOR)


def _best_first(runs):
    """Display order for figures: hosted APIs first (Jev, OpenAI), next to the best local run (nimble embedded),
    then the registry order. Colors don't depend on it."""
    return sorted(runs, key=lambda r: (not BACKENDS[split_run(r)[1]].remote, runs.index(r)))


FIG_RUNS = _best_first(EVAL_RUNS)
FIG_DEV_RUNS = _best_first(DEV_RUNS)
MARKER = {"ollama": "o", "ollaya": "s", "decider": "D", "mlx": "^", "jev": "P", "openai": "X"}

# Latency comes from the 1,000-query eval runs, all measured with nothing else running (tev1 4B's first eval
# overlapped CPU jobs and was re-run on 2026-10-07: identical answers, 9.9 s instead of 13.0 s). A run whose
# latency is known to be distorted goes here with a short note; figures then mark it.
INFLATED = {}


def name(run):
    return NAMES.get(run) or "%s (%s)" % (split_run(run)[2], split_run(run)[1])


def style():
    plt.rcParams.update({
        "svg.fonttype": "none", "svg.hashsalt": "sidm-figures",
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2,
    })


def save(fig, filename, note=None):
    if note:  # below everything already drawn (tick labels, axis labels), so it never collides
        fig.draw_without_rendering()
        bb = fig.get_tightbbox()
        fig.text(bb.x0 / fig.get_figwidth(), (bb.y0 - 0.06) / fig.get_figheight(), note, ha="left", va="top",
                 fontsize=7.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
    plt.close(fig)
    print("written to %s" % path)


# ---------------------------------------------------------------- data
_GOLD = None


def gold():
    global _GOLD
    if _GOLD is None:
        rows = E.load_rows(DATA, "all")
        _GOLD = (rows, {r["id"]: r for r in rows})
    return _GOLD


def preds(run, split="eval", path=None):
    scheme, backend, model = split_run(run)
    path = path or E.pred_path(split, model, scheme, backend)
    return E.load_preds(path, gold()[1]) if Path(path).exists() else []


_METRICS = {}


def metrics(run, split="eval", tuned=True):
    key = (run, split, tuned)
    if key not in _METRICS:
        scheme, backend, model = split_run(run)
        P = E.load_preds(E.pred_path(split, model, scheme, backend), gold()[1], tuned=tuned)
        _METRICS[key] = E.compute(gold()[0], P)
    return _METRICS[key]


def eval_latencies(run):
    P = preds(run, "eval")
    return [p["pred"]["latency_s"] for p in P], [p["pred"]["n_questions"] for p in P]


def prop(k_n):
    k, n = k_n
    lo, hi = wilson(k, n)
    return k / n, lo, hi


# ---------------------------------------------------------------- 0. eval results table
def fig_eval_table():
    """The README's eval table as a figure: cell shade = position between the row's worst and best run."""
    result = json.loads(Path("results/results_eval.json").read_text())
    by_run = {r["run"]: r["metrics"] for r in result["runs"]}
    runs = [r for r in FIG_RUNS if r in by_run]
    rows = [  # (label, value with optional interval, higher is better)
        ("Category, exact node", lambda m: (m["category_exact"]["value"], m["category_exact"]["ci95"]), True),
        ("Category correct at L1", lambda m: (m["category_level"]["L1"]["value"], m["category_level"]["L1"]["ci95"]), True),
        ("Filters F1 (micro)", lambda m: (m["filters_micro"]["f1"], None), True),
        ("Filter set exact match", lambda m: (m["filter_set_exact"]["value"], m["filter_set_exact"]["ci95"]), True),
        ("Word-role accuracy", lambda m: (m["word_role_accuracy"]["value"], m["word_role_accuracy"]["ci95"]), True),
        ("Residual words F1", lambda m: (m["residual_words"]["f1"], None), True),
        ("Whole query exactly right", lambda m: (m["whole_query_exact"]["value"], m["whole_query_exact"]["ci95"]), True),
    ]
    cells = [[fn(by_run[r]) for r in runs] for _, fn, _ in rows]
    lat = []
    for r in runs:
        lat.append((statistics.median(eval_latencies(r)[0]), INFLATED.get(r)))
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP[1:6])  # worst run still gets a visible cell
    n_rows, n_cols = len(rows) + 1, len(runs)
    fig, ax = plt.subplots(figsize=(1.75 * len(runs) + 2, 6.2))

    def cell(i, j, shade, text, sub, bold):
        ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, color=cmap(shade), linewidth=0))
        r, g, b, _ = cmap(shade)  # white text where the cell is darker than mid-grey (relative luminance)
        dark = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2 < 0.3
        ax.text(j + 0.5, i + (0.42 if sub else 0.5), text, ha="center", va="center", fontsize=9.5,
                fontweight="bold" if bold else "normal", color="white" if dark else INK)
        if sub:
            ax.text(j + 0.5, i + 0.72, sub, ha="center", va="center", fontsize=7, color="white" if dark else INK)

    for i, ((label, _, _), row) in enumerate(zip(rows, cells)):
        vals = [v for v, _ in row]
        lo_v, hi_v = min(vals), max(vals)
        for j, (v, ci) in enumerate(row):
            cell(i, j, (v - lo_v) / ((hi_v - lo_v) or 1), "%.3f" % v,
                 "%.2f–%.2f" % tuple(ci) if ci else None, v == hi_v)
    i = len(rows)  # latency: lower is better, so the fastest run gets the darkest shade
    lv = [v for v, _ in lat]
    for j, (v, note) in enumerate(lat):
        cell(i, j, (max(lv) - v) / ((max(lv) - min(lv)) or 1), "%.1f s" % v, note, v == min(lv))
    labels = [r[0] for r in rows] + ["Latency p50"]
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, -0.55)
    ax.set_yticks([k + 0.5 for k in range(n_rows)])
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    for j, r in enumerate(runs):  # column header: run color chip (same color as every other figure) + name
        ax.add_patch(plt.Rectangle((j + 0.08, -0.42), 0.12, 0.22, color=COLOR[r], linewidth=0, clip_on=False))
        ax.text(j + 0.26, -0.31, name(r), ha="left", va="center", fontsize=8.5, color=INK)
        ax.text(j + 0.26, -0.1, split_run(r)[1], ha="left", va="center", fontsize=7.5, color=MUTED)
    ax.axhline(len(rows) + 0.0, color=AXIS, linewidth=0.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Eval results, 1,000 queries (decoding tuned per model on dev)", pad=6)
    save(fig, "eval_results.svg", "Bold: best run in the row. Shade: position between the row's worst (light) and "
                                  "best (dark) run; for latency, faster is darker. Small text: 95% Wilson interval. "
                                  "Hosted APIs' latency includes the network.")


# ---------------------------------------------------------------- weighted query score: pairwise tests, breakdown
def fig_score_tests():
    result = json.loads(Path("results/results_eval.json").read_text())
    runs = [r for r in FIG_RUNS if r in {x["run"] for x in result["runs"]}]
    tests = {(t["reference"], t["run"]): t for t in result["paired_score_tests"]}
    cmap = LinearSegmentedColormap.from_list("div", [RED_POLE, "#f4b3b2", NEUTRAL, "#a9c9f0", BLUE_POLE])
    n = len(runs)
    fig, ax = plt.subplots(figsize=(0.95 * n + 2.2, 0.75 * n + 0.6))
    for i, a in enumerate(runs):
        for j, b in enumerate(runs):
            if i == j:
                continue
            t = tests.get((a, b)) or tests.get((b, a))
            d = t["mean_diff_reference_minus_run"] * (1 if t["reference"] == a else -1)  # row minus column
            sig = t["p_value"] < 0.05
            share = max(-1.0, min(1.0, d / 0.15))  # color saturates at a 0.15 score difference
            color = cmap(0.5 + 0.5 * share) if sig else NEUTRAL
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=color, linewidth=0))
            strong = sig and abs(share) > 0.55
            ax.text(j + 0.5, i + 0.42, "%+.3f" % d, ha="center", va="center", fontsize=8,
                    color="white" if strong else INK, fontweight="bold" if sig else "normal")
            p = t["p_value"]
            ax.text(j + 0.5, i + 0.72, "p<.001" if p < 1e-3 else "p=%.3f" % p, ha="center", va="center",
                    fontsize=6.5, color="white" if strong else INK2)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([k + 0.5 for k in range(n)])
    ax.set_yticks([k + 0.5 for k in range(n)])
    ax.set_xticklabels([name(r) for r in runs], rotation=35, ha="right")
    ax.set_yticklabels([name(r) for r in runs])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Weighted query score, pairwise: row run's mean score minus the column run's (1,000 eval queries)")
    save(fig, "score_tests_eval.svg", "Paired sign-flip test on the per-query scores. Blue: row run significantly "
                                      "better; red: column run better (p < 0.05); grey: no significant difference.")


def fig_score_breakdown():
    """Where each run's score comes from: the points each part contributes (they add up to the score)."""
    from sidm.score import PARTS, WEIGHTS
    result = json.loads(Path("results/results_eval.json").read_text())
    ranges = result.get("score_rank_ranges", {})
    runs = [r for r in FIG_RUNS if r in {x["run"] for x in result["runs"]}]
    labels = {"category": "category", "filters": "filters (F0.5)", "residual": "residual words (F2)",
              "other_roles": "other word roles"}
    shades = [BLUE_RAMP[6], BLUE_RAMP[4], BLUE_RAMP[3], BLUE_RAMP[2]]  # importance order: darkest = category
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(runs) + 1.6))
    for i, run in enumerate(runs):
        contrib = dict.fromkeys(PARTS, 0.0)
        per_q = metrics(run)["per_query"].values()
        for q in per_q:  # each part's share of the query's score: w_k * v_k / sum of present weights
            parts = q["score_parts"]
            den = sum(WEIGHTS[k] for k in PARTS if parts[k] is not None)
            for k in PARTS:
                if parts[k] is not None:
                    contrib[k] += WEIGHTS[k] * parts[k] / den / len(per_q)
        left = 0.0
        for k, c in zip(PARTS, shades):
            ax.barh(i, contrib[k], left=left, height=0.62, color=c, edgecolor="white", linewidth=1,
                    label=labels[k] if i == 0 else None)
            left += contrib[k]
        ax.barh(i, 1 - left, left=left, height=0.62, color="white", edgecolor=AXIS, linewidth=0.8, hatch="////",
                label="points lost" if i == 0 else None)
        rr = ranges.get(run)
        rank = ("rank %d" % rr["min"] if rr["min"] == rr["max"] else "rank %d–%d" % (rr["min"], rr["max"])) if rr else ""
        ax.text(1.01, i, "%.3f · %s" % (left, rank) if rank else "%.3f" % left, va="center",
                fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([name(r) for r in runs])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("weighted query score: points earned per part (they add up to the score)")
    ax.set_title("Where each run's weighted query score comes from (1,000 eval queries)", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=8, handlelength=1.4, borderaxespad=0.2)
    save(fig, "score_breakdown.svg", "Right: score and its rank range over 300 random weight vectors that keep the "
                                     "order category ≥ filters ≥ residual words ≥ other word roles.")


# ---------------------------------------------------------------- 1. accuracy vs latency
def _latency_scatter(value, ylabel, title, filename, note, ylim):
    """Shared by accuracy_vs_latency and score_vs_latency: value(metrics) -> (value, ci_low, ci_high)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pts = []
    for run in FIG_RUNS:
        m = metrics(run)
        acc, lo, hi = value(m)
        p50 = statistics.median(eval_latencies(run)[0])
        pts.append((p50, acc, run))
        ax.errorbar(p50, acc, yerr=[[acc - lo], [hi - acc]], fmt="none", ecolor=COLOR[run], elinewidth=1.4, capsize=0)
        ax.scatter(p50, acc, s=70, color=COLOR[run], marker=MARKER[split_run(run)[1]], edgecolor="white",
                   linewidth=2, zorder=3, label=name(run))
        pts[-1] = pts[-1] + (name(run) + (" (latency %s)" % INFLATED[run] if run in INFLATED else ""),)
    # direct labels; a label that would sit on top of an earlier one (close in log-x and y) goes below its point
    lo_y, hi_y = ylim([p[1] for p in pts])
    placed = []
    for x, y, run, label in sorted(pts, key=lambda p: p[0]):
        clash = any(abs(math.log10(x) - math.log10(px)) < 0.25 and abs(y - py) < 0.045 * (hi_y - lo_y)
                    for px, py in placed)
        ax.annotate(label, (x, y), xytext=(8, -12 if clash else 4), textcoords="offset points", fontsize=8.5,
                    color=INK2)
        if not clash:
            placed.append((x, y))
    pts = [p[:3] for p in pts]
    # Pareto frontier of the local runs (a hosted API's latency is another machine plus the network): no other
    # local run is both faster and more accurate
    local = [p for p in pts if not BACKENDS[split_run(p[2])[1]].remote]
    front = sorted(p for p in local if not any(q[0] <= p[0] and q[1] > p[1] for q in local if q is not p))
    ax.plot([p[0] for p in front], [p[1] for p in front], color=AXIS, linewidth=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g s" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(p[0] for p in pts) / 1.5, 26)
    ax.set_ylim(*ylim([p[1] for p in pts]))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("p50 latency per query: M3 Max, or the hosted API (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [plt.Line2D([], [], marker=MARKER[b], linestyle="", color=MUTED, markersize=7, label=lbl)
               for b, lbl in (("ollama", "Ollama"), ("ollaya", "Ollaya"), ("decider", "Strands Decider server"),
                              ("jev", "TypeSafe API (hosted, incl. network)"),
                              ("openai", "OpenAI API (hosted, incl. network)"))
               if any(split_run(r)[1] == b for r in EVAL_RUNS)]
    ax.legend(handles=handles, title="backend (marker)", loc="center left", bbox_to_anchor=(0.01, 0.42),
              title_fontsize=8.5)
    save(fig, filename, note)


def fig_accuracy_latency():
    _latency_scatter(lambda m: prop(m["full_query_exact_counts"]), "whole query exactly right",
                     "Accuracy vs latency, 1,000 eval queries", "accuracy_vs_latency.svg",
                     "Vertical lines: 95% Wilson intervals of the accuracy. Grey line: Pareto frontier of the local runs "
                     "(M3 Max). Latency: p50 over the same 1,000 queries.",
                     lambda ys: (0, math.ceil((max(ys) + 0.06) * 10) / 10))


def fig_score_latency():
    _latency_scatter(lambda m: (m["weighted_score"]["mean"], *m["weighted_score"]["ci95"]), "weighted query score",
                     "Weighted query score vs latency, 1,000 eval queries", "score_vs_latency.svg",
                     "Vertical lines: 95% intervals of the mean score. Grey line: Pareto frontier of the local runs "
                     "(M3 Max). Latency: p50 over the same 1,000 queries. Score: see sidm/score.py.",
                     lambda ys: (math.floor((min(ys) - 0.05) * 10) / 10, 1.0))


# ---------------------------------------------------------------- 2. accuracy by part
PARTS = [
    ("Category, exact node", lambda m: prop(m["category_exact_counts"])),
    ("Filters F1", lambda m: (m["filters_micro_PRF"][2], None, None)),
    ("Word-role accuracy", lambda m: prop(m["token_counts"])),
    ("Residual words F1", lambda m: (m["residual_word_PRF"][2], None, None)),
    ("Whole query exactly right", lambda m: prop(m["full_query_exact_counts"])),
    ("Weighted query score", lambda m: (m["weighted_score"]["mean"], *m["weighted_score"]["ci95"])),
]


def fig_accuracy_by_part():
    # accuracy rows share a 0-1 axis; latency gets its own panel and log axis below (never a second y-axis)
    fig, (ax, lat_ax) = plt.subplots(2, 1, figsize=(7.6, 6.8), gridspec_kw={"height_ratios": [len(PARTS), 1.25],
                                                                            "hspace": 0.32})
    runs = FIG_RUNS
    for i, (label, fn) in enumerate(PARTS):
        y0 = len(PARTS) - 1 - i
        if i % 2 == 0:  # alternate row bands, so each metric's group of dots reads as one block
            ax.axhspan(y0 - 0.5, y0 + 0.5, color=BAND, linewidth=0, zorder=0)
        for j, run in enumerate(runs):
            v, lo, hi = fn(metrics(run))
            y = y0 + ((len(runs) - 1) / 2 - j) * 0.1  # first run on top, as in the legend
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=COLOR[run], linewidth=1.4, solid_capstyle="round")
            ax.scatter(v, y, s=42, color=COLOR[run], marker=MARKER[split_run(run)[1]], edgecolor="white",
                       linewidth=1.5, zorder=3, label=name(run) if i == 0 else None)
    ax.set_ylim(-0.5, len(PARTS) - 0.5)
    ax.set_yticks(range(len(PARTS)))
    ax.set_yticklabels([p[0] for p in reversed(PARTS)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("Who wins which part, 1,000 eval queries")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)

    lat_ax.axhspan(-0.5, 0.5, color=BAND, linewidth=0, zorder=0)  # continues the row banding
    for j, run in enumerate(runs):
        p50 = statistics.median(eval_latencies(run)[0])
        y = ((len(runs) - 1) / 2 - j) * 0.1
        lat_ax.scatter(p50, y, s=42, color=COLOR[run], marker=MARKER[split_run(run)[1]],
                       facecolor="white" if run in INFLATED else COLOR[run], edgecolor=COLOR[run] if run in INFLATED
                       else "white", linewidth=1.5, zorder=3)
    lat_ax.set_xscale("log")
    lat_ax.set_xticks([0.3, 0.5, 1, 2, 5, 10, 20])
    lat_ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g s" % v))
    lat_ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lat_ax.set_xlim(0.2, 26)
    lat_ax.set_ylim(-0.5, 0.5)
    lat_ax.set_yticks([0])
    lat_ax.set_yticklabels(["Latency p50\n(lower is better)"])
    lat_ax.tick_params(axis="y", length=0)
    lat_ax.xaxis.grid(True)
    lat_ax.set_axisbelow(True)
    lat_ax.spines["left"].set_visible(False)
    save(fig, "accuracy_by_part.svg", "Lines: 95% intervals (Wilson for proportions, normal for the mean score; F1 "
                                       "has none). Weighted query score: see sidm/score.py. Decoding tuned "
                                       "per model on dev.\nLatency: log scale, own axis; hosted APIs (Jev, OpenAI) include the network.")


# ---------------------------------------------------------------- 4. McNemar heatmaps
def fig_mcnemar():
    result = json.loads(Path("results/results_eval.json").read_text())
    runs = [r for r in FIG_RUNS if r in {x["run"] for x in result["runs"]}]
    tests = {(t["reference"], t["run"], t["metric"]): t for t in result["paired_mcnemar"]}
    cmap = LinearSegmentedColormap.from_list("div", [RED_POLE, "#f4b3b2", NEUTRAL, "#a9c9f0", BLUE_POLE])
    fig, axes = plt.subplots(1, 2, figsize=(1.6 * len(runs) + 1.5, 0.75 * len(runs) + 0.6))
    for ax, (metric, title) in zip(axes, (("full", "Whole query exactly right"), ("category", "Category, exact node"))):
        n = len(runs)
        for i, a in enumerate(runs):
            for j, b in enumerate(runs):
                if i == j:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color="white"))
                    continue
                t = tests.get((a, b, metric)) or tests.get((b, a, metric))
                ra, rb = (t["correct_only_reference"], t["correct_only_run"]) if t["reference"] == a else \
                    (t["correct_only_run"], t["correct_only_reference"])
                sig = t["p_value"] < 0.05
                share = (ra - rb) / max(1, ra + rb)  # >0: row better
                color = cmap(0.5 + 0.5 * share) if sig else NEUTRAL
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=color, linewidth=0))
                txt = "%d:%d" % (ra, rb)
                strong = sig and abs(share) > 0.55
                ax.text(j + 0.5, i + 0.42, txt, ha="center", va="center", fontsize=8,
                        color="white" if strong else INK, fontweight="bold" if sig else "normal")
                ax.text(j + 0.5, i + 0.72, "p<.001" if t["p_value"] < 1e-3 else "p=%.3f" % t["p_value"],
                        ha="center", va="center", fontsize=6.5, color="white" if strong else INK2)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_xticks([k + 0.5 for k in range(n)])
        ax.set_yticks([k + 0.5 for k in range(n)])
        ax.set_xticklabels([name(r) for r in runs], rotation=35, ha="right")
        ax.set_yticklabels([name(r) for r in runs] if ax is axes[0] else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title)
    fig.subplots_adjust(wspace=0.06)
    fig.suptitle("Pairwise McNemar tests, 1,000 eval queries: queries correct only with the row run : only with "
                 "the column run", x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "mcnemar_eval.svg", "Blue: row run significantly better; red: column run better (p < 0.05); "
                                  "grey: no significant difference. Color depth = share of the disagreements won.")


# ---------------------------------------------------------------- 5. calibration
def _reliability(P, kind):
    g = gold()[1]
    conf, ok = [], []
    for p in P:
        ans, gr = p["pred"]["raw_answers"], g[p["id"]]
        if kind == "filters":
            for attr in S.ATTRIBUTES:
                a = ans.get("filter_" + attr)
                if not a:
                    continue
                pr = a["probabilities"]
                c = max(pr, key=pr.get)
                conf.append(pr[c])
                ok.append(c == gr["filters"].get(attr, "not_specified"))
        else:
            for i, lab in enumerate(gr["token_labels"], 1):
                a = ans.get("word_%d" % i)
                if not a:
                    continue
                pr = a["probabilities"]
                c = max(pr, key=pr.get)
                conf.append(pr[c])
                ok.append(c == lab)
    bins = [[] for _ in range(10)]
    for c, o in zip(conf, ok):
        bins[min(9, int(c * 10))].append((c, o))
    xs = [statistics.mean(c for c, _ in b) for b in bins if len(b) >= 20]
    ys = [statistics.mean(o for _, o in b) for b in bins if len(b) >= 20]
    ece = sum(len(b) * abs(statistics.mean(c for c, _ in b) - statistics.mean(o for _, o in b))
              for b in bins if b) / max(1, len(conf))
    return xs, ys, ece


def fig_calibration():
    runs = FIG_RUNS
    cells = len(runs) + 1  # + the legend
    ncol = min((3, 4, 5), key=lambda c: (math.ceil(cells / c) * c - cells, -c))  # fewest empty cells
    nrow = math.ceil((len(runs) + 1) / ncol)  # +1 cell for the legend
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.1 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    series = (("filters", "filter questions", PALETTE[0]), ("words", "word-role questions", PALETTE[1]))
    for ax, run in zip(axes, runs):
        P = preds(run)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        labels = []
        for kind, label, color in series:
            xs, ys, ece = _reliability(P, kind)
            ax.plot(xs, ys, color=color, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1)
            labels.append("%s ECE %.2f" % ("filters" if kind == "filters" else "words", ece))
        ax.text(0.04, 0.96, "\n".join(labels), transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
        ax.set_title(name(run), fontsize=9.5)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True)
        ax.set_axisbelow(True)
    for ax in axes[len(runs):]:
        ax.axis("off")
    axes[-1].legend(handles=[plt.Line2D([], [], color=c, marker="o", label=l) for _, l, c in series] +
                    [plt.Line2D([], [], color=AXIS, linewidth=1, label="perfect calibration")],
                    loc="center", fontsize=8.5)
    fig.supxlabel("probability of the chosen option (raw model output)", fontsize=9, color=INK2)
    fig.supylabel("share of answers that are correct", fontsize=9, color=INK2)
    fig.suptitle("Calibration, 1,000 eval queries: how much a model's probability can be trusted",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "calibration_eval.svg", "Bins of 0.1 with ≥ 20 answers. ECE: expected calibration error (lower is "
                                      "better). Why decoding thresholds are tuned per model.")


# ---------------------------------------------------------------- 6. accuracy by difficulty (heatmap)
def _strata(gr):
    n_f = len(gr["filters"])
    depth = len(gr["category_path"])
    nw = len(gr["tokens"])
    return {
        "filters": "%d" % n_f,
        "typo": "typo" if gr["gold_intent"].get("typo") else "no typo",
        "category depth": {0: "none", 1: "L1", 2: "L2", 3: "L3"}[depth],
        "query words": "1–3" if nw <= 3 else "4–6" if nw <= 6 else "7+",
    }


def _difficulty(key, vrange, title, filename, note):
    """Shared by accuracy_by_difficulty (key 'full') and score_by_difficulty (key 'score')."""
    rows, g = gold()
    eval_ids = {r["id"] for r in E.load_rows(DATA, "eval")}
    order = {"filters": ["0", "1", "2", "3"], "category depth": ["none", "L1", "L2", "L3"],
             "query words": ["1–3", "4–6", "7+"], "typo": ["no typo", "typo"]}
    cols = [(f, v) for f, vals in order.items() for v in vals]
    counts = Counter((f, _strata(g[i])[f]) for i in eval_ids for f in order)
    runs = FIG_RUNS
    grid = []
    for run in runs:
        per_q = metrics(run)["per_query"]
        acc = defaultdict(lambda: [0, 0])
        for i, r in per_q.items():
            for f, v in _strata(g[int(i) if isinstance(i, str) else i]).items():
                acc[(f, v)][0] += r[key]
                acc[(f, v)][1] += 1
        grid.append([acc[c][0] / acc[c][1] if acc[c][1] else float("nan") for c in cols])
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    vmin, vmax, dark_from = vrange
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(runs)):
        for j in range(len(cols)):
            v = grid[i][j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8, color="white" if v > dark_from else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["%s\nn=%d" % (v, counts[(f, v)]) for f, v in cols], fontsize=8)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([name(r) for r in runs])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    x = 0
    for f, vals in order.items():  # group labels and separators
        ax.text(x + (len(vals) - 1) / 2, -0.85, f, ha="center", va="bottom", fontsize=9, color=INK2,
                fontweight="bold")
        if x:
            ax.axvline(x - 0.5, color="white", linewidth=3)
        x += len(vals)
    ax.set_title(title, pad=26)
    save(fig, filename, note)


def fig_difficulty():
    _difficulty("full", (0, 0.8, 0.42), "Whole query exactly right, by query difficulty (1,000 eval queries)",
                "accuracy_by_difficulty.svg", "Cells: share of the stratum's queries parsed exactly right; n: queries "
                "per stratum. 'none' = queries without a product type (e.g. 'modern plastic "
                "furniture'): every model tags the word 'furniture' as category, gold says residual.")


def fig_score_difficulty():
    _difficulty("score", (0.4, 1.0, 0.78), "Weighted query score, by query difficulty (1,000 eval queries)",
                "score_by_difficulty.svg", "Cells: mean weighted query score of the stratum's queries; n: queries per "
                "stratum. Compare with accuracy_by_difficulty.svg (whole query exactly right).")


# ---------------------------------------------------------------- 7. latency vs questions
def fig_latency_questions():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for run in FIG_RUNS:
        lat, nq = eval_latencies(run)
        by = defaultdict(list)
        for l, q in zip(lat, nq):
            by[q].append(l)
        xs = sorted(q for q in by if len(by[q]) >= 10)
        stats = [median_range(by[q]) for q in xs]
        med = [m for m, _, _ in stats]
        ax.scatter(nq, lat, s=6, color=COLOR[run], alpha=0.18, linewidth=0, rasterized=True)
        ax.fill_between(xs, [lo for _, lo, _ in stats], [hi for _, _, hi in stats], color=COLOR[run], alpha=0.2,
                        linewidth=0)
        ax.plot(xs, med, color=COLOR[run], marker=MARKER[split_run(run)[1]], markersize=5,
                linestyle="--" if run in INFLATED else "-", markeredgecolor="white", markeredgewidth=1,
                label=name(run) + (" (%s)" % INFLATED[run] if run in INFLATED else ""))
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g s" % v))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("questions in the request (18 fixed + 1 per query word)")
    ax.set_ylabel("latency per query (log scale)")
    ax.set_title("Latency vs request size, 1,000 eval queries, M3 Max")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    save(fig, "latency_vs_eval_request_size.svg", "Dots: single queries; lines: median per question count (counts "
                                                  "with ≥ 10 queries); bands: where 95% of the queries land (2.5th–97.5th percentile). "
                                                  "Question counts vary only with query length.")


# ---------------------------------------------------------------- 7b. controlled sweep (make sweep)
SWEEP_RUNS = [r for r in FIG_RUNS if not r.startswith("router@")] + ["embedded@mlx:nimble"]  # router: same model


def fig_latency_sweep():
    from sidm.bench import QUERY, sweep_path
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    found = 0
    for run in SWEEP_RUNS:
        _, backend, model = split_run(run)
        path = sweep_path(backend, model)
        if not path.exists():
            continue
        found += 1
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        by = defaultdict(list)
        for r in rows:
            by[r["n_questions"]].append(r["latency_s"])
        xs = sorted(by)
        med = [statistics.median(by[n]) for n in xs]
        color = COLOR.get(run, INK2)
        ax.scatter([r["n_questions"] for r in rows], [r["latency_s"] for r in rows], s=7, color=color, alpha=0.3,
                   linewidth=0)
        ax.plot(xs, med, color=color, marker=MARKER[backend], markersize=4, markeredgecolor="white",
                markeredgewidth=0.8, linewidth=1.8, label="%s: %.1f s at N=%d" % (name(run), med[-1], xs[-1]))
    if not found:
        plt.close(fig)
        print("skipped latency_sweep.svg: no results/bench/sweep_*.jsonl (make sweep)")
        return
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, None)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g s" % v))
    blocks = ((1, 6, "category questions"), (7, 18, "filter questions"), (19, 29, "word questions"))
    for k, (lo, hi, lbl) in enumerate(blocks):
        if k % 2 == 0:  # alternate column bands mark the three question blocks
            ax.axvspan(lo - 0.5, hi + 0.5, color=BAND, linewidth=0, zorder=0)
        ax.text((lo + hi) / 2, 1.01, lbl, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.set_xlim(0.5, 29.5)
    ax.set_xlabel("questions sent: the first N of the full 29-question request")
    ax.set_ylabel("latency per request")
    ax.set_title("Latency vs number of questions, controlled sweep, M3 Max (Jev, OpenAI: hosted APIs)", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    save(fig, "latency_sweep.svg", 'One request ("%s") sent with its first N questions, 3 times per N in shuffled '
                                   "order.\nEvery request starts with a new first word, so no prefill is reused. "
                                   "Dots: single requests; lines: median." % QUERY)


# ---------------------------------------------------------------- hosted APIs: latency up close
HOSTED_RUNS = [r for r in FIG_RUNS if BACKENDS[split_run(r)[1]].remote]


def median_range(values, share=0.95):
    """Median and the range holding the middle `share` of the values (2.5th-97.5th percentile for 95%),
    by linear interpolation between order statistics: where most points land, not the median's uncertainty."""
    xs = sorted(values)

    def pct(q):
        pos = q * (len(xs) - 1)
        i = int(pos)
        return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (pos - i)

    tail = (1 - share) / 2
    return statistics.median(xs), pct(tail), pct(1 - tail)


def _hosted_axes(ax, ylabel):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g s" % v))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8.5)


def fig_hosted_latency_eval():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    top = 0
    for k, run in enumerate(HOSTED_RUNS):
        lat, nq = eval_latencies(run)
        by = defaultdict(list)
        for l, q in zip(lat, nq):
            by[q].append(l)
        xs = sorted(q for q in by if len(by[q]) >= 10)
        stats = [median_range(by[q]) for q in xs]
        off = (k - (len(HOSTED_RUNS) - 1) / 2) * 0.12  # keep the two runs' dots apart
        ax.scatter([q + off for q in nq], lat, s=6, color=COLOR[run], alpha=0.2, linewidth=0, rasterized=True)
        ax.fill_between(xs, [s_[1] for s_ in stats], [s_[2] for s_ in stats], color=COLOR[run], alpha=0.18,
                        linewidth=0)
        ax.plot(xs, [s_[0] for s_ in stats], color=COLOR[run], marker=MARKER[split_run(run)[1]], markersize=5,
                markeredgecolor="white", markeredgewidth=1, label="%s: p50 %.2f s" % (name(run), statistics.median(lat)))
        top = max(top, sorted(lat)[int(0.99 * len(lat))])
    ax.set_ylim(0, top * 1.1)
    ax.set_xlabel("questions in the request (18 fixed + 1 per query word)")
    ax.set_title("Hosted APIs: latency vs request size, 1,000 eval queries")
    _hosted_axes(ax, "latency per query (network included)")
    save(fig, "hosted_latency_eval.svg", "Lines: median per question count (counts with ≥ 10 queries); bands: where "
                                         "95% of the queries land (2.5th–97.5th percentile). Dots: single queries; "
                                         "the slowest 1% are above the plot.")


def fig_hosted_latency_sweep():
    from sidm.bench import sweep_path
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    top = 0
    for run in HOSTED_RUNS:
        _, backend, model = split_run(run)
        path = sweep_path(backend, model)
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        by = defaultdict(list)
        for r in rows:
            by[r["n_questions"]].append(r["latency_s"])
        xs = sorted(by)
        ax.scatter([r["n_questions"] for r in rows], [r["latency_s"] for r in rows], s=10, color=COLOR[run],
                   alpha=0.35, linewidth=0)
        ax.plot(xs, [statistics.median(by[n]) for n in xs], color=COLOR[run], marker=MARKER[backend], markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=name(run))
        top = max(top, sorted(r["latency_s"] for r in rows)[int(0.97 * len(rows))])
    for k, (lo, hi) in enumerate(((1, 6), (7, 18), (19, 29))):
        if k % 2 == 0:
            ax.axvspan(lo - 0.5, hi + 0.5, color=BAND, linewidth=0, zorder=0)
    for lo, hi, lbl in ((1, 6, "category questions"), (7, 18, "filter questions"), (19, 29, "word questions")):
        ax.text((lo + hi) / 2, 1.01, lbl, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color=INK2)
    ax.set_xlim(0.5, 29.5)
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel("questions sent: the first N of the full 29-question request")
    ax.set_title("Hosted APIs: latency vs number of questions, controlled sweep", pad=20)
    _hosted_axes(ax, "latency per request (network included)")
    save(fig, "hosted_latency_sweep.svg", "3 requests per N in shuffled order, each with a new first word. Lines: "
                                          "median. The slowest 3% of requests are above the plot.")


# ---------------------------------------------------------------- 8. tuning effect
def fig_tuning():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    runs = FIG_RUNS
    for i, run in enumerate(runs):
        u = metrics(run, tuned=False)["full_query_exact_acc"]
        t = metrics(run)["full_query_exact_acc"]
        y = len(runs) - 1 - i
        ax1.plot([u, t], [y, y], color=AXIS, linewidth=2, zorder=1)
        ax1.scatter([u], [y], s=46, color="white", edgecolor=COLOR[run], linewidth=1.8, zorder=3)
        ax1.scatter([t], [y], s=54, color=COLOR[run], edgecolor="white", linewidth=1.5, zorder=3)
        ax1.text(max(u, t) + 0.012, y, "±0.00" if abs(t - u) < 0.005 else "%+.2f" % (t - u), va="center", fontsize=8, color=INK2)
    ax1.set_yticks(range(len(runs)))
    ax1.set_yticklabels([name(r) for r in reversed(runs)])
    ax1.tick_params(axis="y", length=0)
    ax1.spines["left"].set_visible(False)
    ax1.set_xlim(0, math.ceil((max(metrics(r)["full_query_exact_acc"] for r in runs) + 0.07) * 10) / 10)
    ax1.xaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("whole query exactly right (eval)")
    ax1.set_title("Plain argmax (hollow) → tuned decoding (filled)")
    # filter-threshold sweep for nimble on dev
    from sidm.parser import override_tuning, tuning_key
    rows, g = gold()
    path = E.pred_path("dev", "nimble", "embedded", "ollama")
    xs = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    vals = {"precision": [], "recall": [], "F1": []}
    for x in xs:
        with override_tuning(tuning_key("ollama", "nimble"), filter_min_prob=x):
            m = E.compute(rows, E.load_preds(path, g))
        p, r, f = m["filters_micro_PRF"]
        vals["precision"].append(p)
        vals["recall"].append(r)
        vals["F1"].append(f)
    for (k, v), c in zip(vals.items(), PALETTE):
        ax2.plot(xs, v, color=c, label=k, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1)
    ax2.axvline(0.95, color=AXIS, linewidth=1)
    ax2.text(0.94, 0.705, "chosen 0.95", ha="right", fontsize=8, color=INK2)
    ax2.set_ylim(0.7, 1.0)
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("filter probability threshold")
    ax2.set_title("nimble filters vs threshold (dev)")
    ax2.legend(loc="lower left")
    save(fig, "tuning_effect.svg", "Tuning sets three decoder parameters per model on the 100 dev queries; "
                                   "no model weights change.")


# ---------------------------------------------------------------- 9. per-attribute filter F1
def fig_filter_attributes():
    runs = FIG_RUNS
    per = [metrics(run)["filters_per_attr"] for run in runs]
    attrs = sorted(per[0], key=lambda a: -per[0][a]["support"])
    grid = [[per[i][a]["F1"] if per[i][a]["support"] else float("nan") for i in range(len(runs))] for a in attrs]
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.imshow(grid, cmap=cmap, vmin=0.3, vmax=1, aspect="auto")
    for i in range(len(attrs)):
        for j in range(len(runs)):
            v = grid[i][j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8, color="white" if v > 0.72 else INK)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([name(r) for r in runs], rotation=30, ha="right")
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels(["%s  (n=%d)" % (a.replace("_", " "), per[0][a]["support"]) for a in attrs])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Filter F1 per attribute, 1,000 eval queries")
    save(fig, "filter_f1_by_attribute.svg", "n: gold occurrences of the attribute in eval. Color scale starts at "
                                            "0.3. Width (inches → bucket) is the hardest attribute for most models.")


# ---------------------------------------------------------------- 10. word-role confusion (dev, all runs)
def fig_word_roles():
    roles = ["category", "filter", "residual"]
    runs = FIG_DEV_RUNS
    ncol = 5
    nrow = math.ceil(len(runs) / ncol)
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.75 * nrow))
    for ax, run in zip(axes.ravel(), runs):
        conf = metrics(run, split="dev")["token_confusion"]
        mat = []
        for a in roles:
            tot = sum(conf.get("%s->%s" % (a, b), 0) for b in roles) or 1
            mat.append([conf.get("%s->%s" % (a, b), 0) / tot for b in roles])
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "%.2f" % mat[i][j], ha="center", va="center", fontsize=7.5,
                        color="white" if mat[i][j] > 0.6 else INK)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["cat", "filt", "res"], fontsize=7.5)
        ax.set_yticklabels(["cat", "filt", "res"], fontsize=7.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(name(run), fontsize=8.5)
    for ax in axes.ravel()[len(runs):]:
        ax.axis("off")
    fig.suptitle("Word roles, 100 dev queries: rows = gold role, columns = predicted (row-normalized)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "word_role_confusion_dev.svg", "Encoders and small models put nearly every word in one column: "
                                             "they cannot resolve 'role of word N' against the numbered word list.")


# ---------------------------------------------------------------- 11. dataset statistics
def fig_dataset():
    rows, _ = gold()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.4, 1.6]})
    c = PALETTE[0]

    def bars(ax, labels, values, title, horizontal=False):
        if horizontal:
            ax.barh(range(len(values)), values, height=0.6, color=c)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.xaxis.grid(True)
            for i, v in enumerate(values):
                ax.annotate("%d" % v, (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        else:
            ax.bar(range(len(values)), values, width=0.6, color=c)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.yaxis.grid(True)
            for i, v in enumerate(values):
                ax.annotate("%d" % v, (i, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=9.5)

    depth = Counter(len(r["category_path"]) for r in rows)
    bars(axes[0], ["none", "L1", "L2", "L3"], [depth[k] for k in range(4)], "Category depth")
    nf = Counter(len(r["filters"]) for r in rows)
    bars(axes[1], ["0", "1", "2", "3"], [nf[k] for k in range(4)], "Filters per query")
    nw = Counter(min(len(r["tokens"]), 11) for r in rows)
    ks = list(range(1, 12))
    bars(axes[2], [str(k) if k < 11 else "11+" for k in ks], [nw[k] for k in ks], "Words per query")
    att = Counter(a for r in rows for a in r["filters"])
    order = sorted(att, key=lambda a: -att[a])
    bars(axes[3], [a.replace("_", " ") for a in order], [att[a] for a in order], "Filter attributes", horizontal=True)
    fig.suptitle("Synthetic dataset: 1,100 furniture queries (dev 100 + eval 1,000)", x=0.01, y=1.04, ha="left",
                 fontsize=10.5, fontweight="bold")
    save(fig, "dataset_stats.svg")


# ---------------------------------------------------------------- 12. category scheme ablation
def fig_scheme_ablation():
    rows, g = gold()
    kinds = ["false 'no category'", "missed 'no category'", "wrong group", "right group, wrong node/depth"]
    runs = ["router@ollama:nimble", "embedded@ollama:nimble"]
    counts = {}
    for run in runs:
        c = Counter()
        for p in preds(run):
            gc, pc = g[p["id"]]["category"], p["pred"]["category"]
            if gc == pc:
                continue
            if pc is None:
                c[kinds[0]] += 1
            elif gc is None:
                c[kinds[1]] += 1
            elif S.NODES[pc].path()[0] != S.NODES[gc].path()[0]:
                c[kinds[2]] += 1
            else:
                c[kinds[3]] += 1
        counts[run] = c
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    h = 0.34
    for k, run in enumerate(runs):
        ys = [i + (k - 0.5) * (h + 0.04) for i in range(len(kinds))]
        vals = [counts[run][kd] for kd in kinds]
        ax.barh(ys, vals, height=h, color=COLOR[run], label="%s: %d errors" % (name(run), sum(vals)))
        for y, v in zip(ys, vals):
            ax.annotate("%d" % v, (v, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("category errors out of 1,000 eval queries")
    ax.set_title("Category schemes: where the errors come from")
    ax.legend(loc="lower right")
    save(fig, "category_scheme_errors.svg", "router asks a top-level 'which group?' question; embedded folds the "
                                            "group decision into the group questions (other_product option).")


FIGURES = [fig_eval_table, fig_accuracy_latency, fig_accuracy_by_part, fig_mcnemar, fig_calibration, fig_difficulty,
           fig_latency_questions, fig_latency_sweep, fig_tuning, fig_filter_attributes, fig_word_roles, fig_dataset, fig_scheme_ablation,
           fig_score_latency, fig_score_difficulty, fig_score_tests, fig_score_breakdown,
           fig_hosted_latency_eval, fig_hosted_latency_sweep]


@friendly_main
def main():
    style()
    for f in FIGURES:
        f()


if __name__ == "__main__":
    main()
