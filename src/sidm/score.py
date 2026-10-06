"""The weighted query score: a forgiving whole-query metric next to "whole query exactly right".

Per query, four parts, each in [0, 1]:
  C  category: 1 if exact; else the shared path prefix / the deeper of the two paths (gold seating>sofas>
     chesterfield, predicted sofas -> 2/3; same group only -> 1/3; another group -> 0). 'No category' is
     right only against 'no category'.
  F  filters: F0.5 over attribute=value pairs (precision weighted 4x recall: a wrong filter hides correct
     results, a missing one only adds some).
  R  residual words: F2 over residual word positions (recall weighted 4x precision).
  T  the other word roles: accuracy over the words whose gold role is category or filter.

A part is *present* when the gold or the prediction has it (C always; F when either side has a filter; R when
either side has a residual word; T when the query has category or filter words). The score is the weighted
mean over the present parts only, so an absent part gives no free points, and a hallucinated filter or
residual word still costs (it makes the part present, with value 0):

  score = sum(w_k * v_k for present k) / sum(w_k for present k)
"""
import math
import random
import statistics

from sidm import schema as S

PARTS = ("category", "filters", "residual", "other_roles")
WEIGHTS = {"category": 0.45, "filters": 0.30, "residual": 0.17, "other_roles": 0.08}  # category > filters > words
BETA = {"filters": 0.5, "residual": 2.0}


def _path(node):
    return S.NODES[node].path() if node else []


def category_credit(gold, pred):
    if gold == pred:
        return 1.0
    if gold is None or pred is None:
        return 0.0
    gp, pp = _path(gold), _path(pred)
    shared = 0
    while shared < min(len(gp), len(pp)) and gp[shared] == pp[shared]:
        shared += 1
    return shared / max(len(gp), len(pp))


def fbeta(gold, pred, beta):
    """F-beta of two sets (beta < 1 favors precision, > 1 recall). Callers only use it when a set is non-empty."""
    tp, fn, fp = len(gold & pred), len(gold - pred), len(pred - gold)
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def query_parts(gold_row, pred, cat_key="category", filt_key="filters"):
    """{part: value in [0, 1], or None when the part is absent from both gold and prediction}."""
    gl, pl = gold_row["token_labels"], pred["token_labels"]
    gf, pf = set(gold_row["filters"].items()), set(pred[filt_key].items())
    gres = {i for i, label in enumerate(gl) if label == "residual"}
    pres = {i for i, label in enumerate(pl) if label == "residual"}
    other = [i for i, label in enumerate(gl) if label != "residual"]
    return {
        "category": category_credit(gold_row["category"], pred[cat_key]),
        "filters": fbeta(gf, pf, BETA["filters"]) if gf or pf else None,
        "residual": fbeta(gres, pres, BETA["residual"]) if gres or pres else None,
        "other_roles": sum(gl[i] == pl[i] for i in other) / len(other) if other else None,
    }


def query_score(parts, weights=WEIGHTS):
    present = [k for k in PARTS if parts[k] is not None]
    return sum(weights[k] * parts[k] for k in present) / sum(weights[k] for k in present)


def summarize(parts_list):
    """Mean score with a 95% interval, plus each part's mean over the queries where it is present."""
    scores = [query_score(p) for p in parts_list]
    mean = statistics.mean(scores)
    half = 1.96 * statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else float("nan")
    part_means = {}
    for k in PARTS:
        vals = [p[k] for p in parts_list if p[k] is not None]
        part_means[k] = {"mean": statistics.mean(vals) if vals else None, "present_in": len(vals)}
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": len(scores), "parts": part_means,
            "weights": dict(WEIGHTS), "beta": dict(BETA)}


def paired_permutation_p(a, b, n=10000, seed=0):
    """Two-sided sign-flip permutation test on paired per-query scores: P(|sum of diffs| >= observed) if the two
    runs were equally good (each difference equally likely to have either sign).

    With more than 50 differing queries the flipped sum is close to normal with variance sum(d^2), so the
    p-value comes from that (instant); below that, `n` random sign flips."""
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if not diffs:
        return 1.0
    observed = abs(sum(diffs))
    if len(diffs) > 50:
        return math.erfc(observed / math.sqrt(2 * sum(d * d for d in diffs)))
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def rank_ranges(parts_by_run, n=300, seed=0):
    """Rank range of each run over random weight vectors that keep the importance order
    (category >= filters >= residual >= other_roles): how much the ranking depends on the chosen weights."""
    rng = random.Random(seed)
    ranks = {run: [] for run in parts_by_run}
    for _ in range(n):
        w = dict(zip(PARTS, sorted((rng.random() for _ in PARTS), reverse=True)))
        means = {run: statistics.mean(query_score(p, w) for p in pl) for run, pl in parts_by_run.items()}
        for rank, run in enumerate(sorted(means, key=means.get, reverse=True), 1):
            ranks[run].append(rank)
    return {run: {"min": min(r), "max": max(r), "mode": max(set(r), key=r.count)} for run, r in ranks.items()}
