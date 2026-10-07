"""Search-intent parser: one /v1/systemone request answers category, filters and per-word roles.

Category is asked in one of two ablatable schemes (see CategoryScheme); filters and word roles are shared.
"""
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from sidm import schema as S
from sidm.ollama_client import DEFAULT_MODELS, system_one, system_one_split

MAX_QUESTIONS = 64
NONE, UNSPEC, NOT_SPEC, OTHER = "none", "unspecified", "not_specified", "other_product"


@dataclass(frozen=True)
class CategoryScheme:
    """How the category is asked and decoded. The schemes differ only in these fields.

    router:   explicit top-level question cat_L1 ('which group?' + 'none'), then one *conditional* question
              per group ("if tables is searched, which type?") that must pick a type of that group.
    embedded: no cat_L1; each group question may answer OTHER ("not this group"), so the group decision
              is embedded in the group questions and 'no category' = no group claims the query.
    """
    name: str
    router: bool     # ask cat_L1
    escape: bool     # group questions have the OTHER option
    none_thr: float  # tuned on dev: predict 'no category' when none_signal() > none_thr


SCHEMES = {
    "router": CategoryScheme("router", router=True, escape=False, none_thr=0.5),
    "embedded": CategoryScheme("embedded", router=False, escape=True, none_thr=0.7),  # dev-tuned: P 1.0, R 0.4 on 5 none rows
}
DEFAULT_SCHEME = "embedded"  # eval: category 0.945 vs 0.879 for router (results/compare_schemes_eval.md)


def get_scheme(scheme):
    return SCHEMES[scheme] if isinstance(scheme, str) else scheme

ROLE_GUIDE = (
    "Each word of the query has one role. category: part of the product type name (sofa, coffee table, "
    "chest of drawers). filter: part of a requested attribute value - color, material, leg material/color, style, "
    "shape, bed size, width, seats, firmness, feature or room - including connecting words inside that phrase "
    "('with storage', 'for 6', 'oak legs', 'under 30 inch'). residual: every other word - price, deals, quality, "
    "brand, recipient, year, shopping words and connecting words outside those phrases ('for my mom', 'cheap')."
)
WORD_CRITERIA = {"category": None, "filter": None, "residual": None}


def _syn(node, k=2):
    return node.name + " (" + ", ".join(node.synonyms[:k]) + ")"


def router_question():
    return {
        "type": "choice",
        "instructions": "Which furniture product group is searched? 'none' if no product type is named.",
        "criteria": dict({l1: _syn(S.NODES[l1]) + "; e.g. " + ", ".join(S.NODES[c].name.lower() for c in S.NODES[l1].children[:5])
                          for l1 in S.L1_IDS}, **{NONE: None}),
    }


def group_question(l1, scheme):
    """Type question for one L1 group: its L2 and L3 nodes flattened, plus UNSPEC (and OTHER if escape)."""
    group = S.NODES[l1]
    crit = {}
    if scheme.escape:
        crit[OTHER] = "a different kind of product, or no product named"
        crit[UNSPEC] = "only a general word for %s (%s)" % (group.name.lower(), ", ".join(group.synonyms[:3]))
        instructions = ("Is a kind of %s searched? Which exact product type? "
                        "'%s' if the query is about another kind of product or names none." % (group.name.lower(), OTHER))
    else:
        crit[UNSPEC] = "only the general term, no specific type"
        instructions = "If %s is searched, which exact product type is named?" % group.name.lower()
    for l2 in group.children:
        n2 = S.NODES[l2]
        crit[l2] = _syn(n2) + (", type not specified" if n2.children else "")
        for l3 in n2.children:
            crit[l3] = _syn(S.NODES[l3])
    assert len(crit) <= 26, l1
    return {"type": "choice", "instructions": instructions, "criteria": crit}


def category_questions(scheme):
    qs = {"cat_L1": router_question()} if scheme.router else {}
    qs.update({"cat_" + l1: group_question(l1, scheme) for l1 in S.L1_IDS})
    return qs


def _value_desc(attr, v, forms):
    if attr == "width":
        lo, hi = S.WIDTH_BUCKETS[v]
        if v == "under_30in":
            return "under 30 in"
        if v == "over_90in":
            return "90 in or more"
        return "%d-%d in" % (lo, hi - 1)
    extra = [f for f in forms if f != v.replace("_", " ")][:2]
    return ", ".join(extra) if extra else None


def filter_questions():
    qs = {}
    for attr, (desc, vals) in S.ATTRIBUTES.items():
        crit = {NOT_SPEC: None}
        crit.update({v: _value_desc(attr, v, forms) for v, forms in vals.items()})
        qs["filter_" + attr] = {
            "type": "choice",
            "instructions": "%s requested in the query? 'not_specified' if not mentioned." % desc,
            "criteria": crit,
        }
    return qs


FILTER_QS = filter_questions()
CAT_QS = {name: category_questions(sc) for name, sc in SCHEMES.items()}


def max_words(scheme):
    return MAX_QUESTIONS - len(CAT_QS[scheme.name]) - len(FILTER_QS)


def word_questions(tokens, limit):
    qs = {}
    for i, w in enumerate(tokens[:limit], 1):
        qs["word_%d" % i] = {
            "type": "choice",
            "instructions": 'Role of word %d "%s"?' % (i, w),
            "criteria": WORD_CRITERIA,
        }
    return qs


def build_request(query, scheme=DEFAULT_SCHEME):
    scheme = get_scheme(scheme)
    tokens = query.split()
    state = {"query": query, "words": ["%d: %s" % (i, w) for i, w in enumerate(tokens, 1)], "word_roles": ROLE_GUIDE}
    questions = dict(CAT_QS[scheme.name])
    questions.update(FILTER_QS)
    questions.update(word_questions(tokens, max_words(scheme)))
    return state, questions, tokens


# ---------------------------------------------------------------- decoding
def category_candidates(ans):
    """All (node, prob, question) answers of the group questions, best first; UNSPEC -> the L1 node."""
    cands = []
    for l1 in S.L1_IDS:
        for c, p in ans["cat_" + l1]["probabilities"].items():
            if c != OTHER:
                cands.append((l1 if c == UNSPEC else c, p, "cat_" + l1))
    return sorted(cands, key=lambda x: -x[1])


def none_signal(ans, scheme):
    """Evidence for 'no category' in [0, 1]: P(cat_L1 = none) for router, 1 - best node prob for embedded."""
    if scheme.router:
        return ans["cat_L1"]["probabilities"].get(NONE, 0.0)
    return 1.0 - category_candidates(ans)[0][1]


def decode_category(ans, scheme, tuned=True):
    """Tuned: none if none_signal > scheme.none_thr, else the most confident group answer (group-max).
    Untuned (plain argmax): router follows cat_L1 top-down; embedded is none only if every group says OTHER."""
    if tuned:
        return None if none_signal(ans, scheme) > scheme.none_thr else category_candidates(ans)[0][0]
    if scheme.router:
        l1 = ans["cat_L1"]["choice"]
        if l1 == NONE:
            return None
        c = ans["cat_" + l1]["choice"]
        return l1 if c == UNSPEC else c
    if all(ans["cat_" + l1]["choice"] == OTHER for l1 in S.L1_IDS):
        return None
    return category_candidates(ans)[0][0]


# Decoding settings tuned on the 100 dev rows (see results/dev_report_router.md); untuned = plain argmax everywhere.
TUNED = {"filter_min_prob": 0.95, "word_weights": {"category": 8.0, "filter": 1.0, "residual": 1.0}}
UNTUNED = {"filter_min_prob": 0.0, "word_weights": {"category": 1.0, "filter": 1.0, "residual": 1.0}}
# Backends are calibrated differently, so each may override TUNED and the schemes' none_thr (tuned on its dev run).
TUNED_BY_BACKEND = {
    "ollama": {},
    # mlx dev sweep (results/dev_mlx-nimble_embedded.jsonl): the Ollama-tuned values are already at/near the
    # optimum (filter 0.95 best F1; category weight 8 vs 16 within one query; none_thr flat), so no overrides.
    "mlx": {},
    "ollaya": {},
    "jev": {},
}
# Per-model settings found by `evaluate tune --apply` on that model's dev run, keyed "<backend>:<model>".
# They take precedence over the backend entry above (models are calibrated very differently).
TUNED_FILE = Path(__file__).resolve().parents[2] / "results" / "tuned_settings.json"
if TUNED_FILE.exists():
    TUNED_BY_BACKEND.update(json.loads(TUNED_FILE.read_text()))


def tuning_key(backend, model=None):
    key = "%s:%s" % (backend, model)
    return key if model and key in TUNED_BY_BACKEND else backend


@contextmanager
def override_tuning(key, **settings):
    """Temporarily change the tuned settings stored under `key` (used by `evaluate tune` sweeps)."""
    saved = TUNED_BY_BACKEND.get(key)
    TUNED_BY_BACKEND[key] = dict(saved or {}, **settings)
    try:
        yield
    finally:
        if saved is None:
            TUNED_BY_BACKEND.pop(key, None)
        else:
            TUNED_BY_BACKEND[key] = saved


def tuned_settings(backend, scheme, model=None):
    """(decoding cfg, scheme with the tuned none_thr) for tuned decoding: model entry, else backend entry."""
    over = TUNED_BY_BACKEND.get(tuning_key(backend, model), {})
    cfg = dict(TUNED, **{k: v for k, v in over.items() if k in TUNED})
    thr = over.get("none_thr", {}).get(scheme.name)
    return cfg, (replace(scheme, none_thr=thr) if thr is not None else scheme)


def decode(resp, tokens, scheme=DEFAULT_SCHEME, tuned=True, backend="ollama", model=None):
    ans = resp["answers"]
    scheme = get_scheme(scheme)
    cfg, scheme = tuned_settings(backend, scheme, model) if tuned else (UNTUNED, scheme)
    cat = decode_category(ans, scheme, tuned)
    filters_raw = {}
    for attr in S.ATTRIBUTES:
        q = ans["filter_" + attr]
        c = q["choice"]
        if c != NOT_SPEC and q["probabilities"][c] >= cfg["filter_min_prob"]:
            filters_raw[attr] = c
    allowed = set(S.applicable_attrs(cat))
    filters_masked = {a: v for a, v in filters_raw.items() if a in allowed}
    labels = []
    w = cfg["word_weights"]
    for i in range(1, len(tokens) + 1):
        a = ans.get("word_%d" % i)
        if a is None:
            labels.append("residual")  # truncated words default to residual
            continue
        pr = a["probabilities"]
        labels.append(max(w, key=lambda k: pr.get(k, 0.0) * w[k]))
    return {
        "category": cat,
        "filters_raw": filters_raw,
        "filters": filters_masked,
        "token_labels": labels,
        "residual_tokens": [t for t, l in zip(tokens, labels) if l == "residual"],
        "truncated": len(tokens) > max_words(scheme),
    }


def parse(query, model=None, scheme=DEFAULT_SCHEME, backend="ollama"):
    scheme = get_scheme(scheme)
    model = model or DEFAULT_MODELS[backend]
    state, questions, tokens = build_request(query, scheme)
    resp, latency, n_requests = system_one_split(model, state, questions, backend=backend)
    out = decode(resp, tokens, scheme, backend=backend, model=model)
    out.update(scheme=scheme.name, backend=backend, model=model, served_model=resp.get("model"),
               latency_s=latency, n_requests=n_requests, n_questions=len(questions),
               n_word_questions=sum(q.startswith("word_") for q in questions),
               usage=resp.get("usage"), raw_answers=resp["answers"])
    if "refusals" in resp:  # openai backend: questions the model refused to answer
        out["refusals"] = resp["refusals"]
    if "metrics" in resp:  # mlx backend: prefill vs batched field evaluation timings
        m = resp["metrics"]
        out["server_metrics"] = {k: m[k] for k in ("prefill_seconds", "field_evaluation_seconds", "total_seconds",
                                                   "prefix_tokens", "branch_batches", "mlx_peak_active_gib") if k in m}
    return out


if __name__ == "__main__":
    import json
    import sys
    q = " ".join(sys.argv[1:]) or "grey mid century sectional with oak legs cheap free shipping"
    r = parse(q)
    r.pop("raw_answers")
    print(json.dumps(r, indent=1))
