"""Generate a synthetic search-query dataset: sample gold labels, let an LLM verbalize them, validate.

Usage: python -m sidm.gen_dataset --n 1100 --out data/eval_raw.jsonl [--workers 4]
"""
import argparse
import difflib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sidm import schema as S
from sidm.ollama_client import chat

GEN_MODEL = "qwen3:14b"

ROLE_RULE = """Word labeling rule - EVERY word of the query must be inside exactly one span:
- role "category": the words naming the product type (e.g. "l shaped couch", "chest of drawers" - connecting words inside it like "of" belong to it).
- role "filter": the words expressing one requested attribute value; connecting words that belong to that phrase are included ("with storage", "under 30 inch", "for 6", "oak legs").
- role "residual": every other word: price/quality/shopping words, brands, recipients, and connecting words that are not part of a filter or category phrase ("for my mom" -> residual, "sofa for kids" -> "for kids" is the room filter).
"""

SYSTEM = """You write realistic e-commerce search queries for a furniture store, the way real shoppers type them
(lowercase mostly, short, no punctuation needed, free word order).

Constraints:
- Mention the category using one of the given wordings or a very close natural variant.
- Express each filter using a natural wording (examples given); do not add any other attribute (no extra colors, materials, sizes, styles, features, rooms).
- Include each residual phrase (verbatim or with a tiny natural variation). Do not add other descriptive words.
{rule}
Return JSON: {{"query": "...", "spans": [{{"text": "<exact substring of query>", "role": "category|filter|residual", "attribute": "<attribute name for filters else null>", "value": "<value id for filters else null>"}}]}}
Spans must be listed in query order and together cover every word of the query.""".format(rule=ROLE_RULE)

PROMPT = """Write ONE query that expresses EXACTLY the following intent, nothing more:
{intent}
{cat_note}{typo_note}"""

OUT_FORMAT = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "spans": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "role": {"type": "string", "enum": ["category", "filter", "residual"]},
            "attribute": {"type": ["string", "null"]},
            "value": {"type": ["string", "null"]},
        }, "required": ["text", "role", "attribute", "value"]}},
    },
    "required": ["query", "spans"],
}


# ---------------------------------------------------------------- sampling
def sample_gold(rng):
    r = rng.random()
    if r < 0.05:
        node = None
    elif r < 0.13:
        node = rng.choice(S.L1_IDS)
    elif r < 0.50:
        node = rng.choice([n for n in S.NODES.values() if n.level == 2]).id
    else:
        node = rng.choice([n for n in S.NODES.values() if n.level == 3]).id

    l1 = S.NODES[node].path()[0] if node else None
    attrs = S.applicable_attrs(node) if node else ["color", "material", "style", "leg_material", "leg_color", "shape"]
    if l1 and not S.ROOMS_L1[l1]:
        attrs = [a for a in attrs if a != "room"]
    if node == "loveseat":
        attrs = [a for a in attrs if a != "seat_count"]
    n_f = rng.choices([0, 1, 2, 3], weights=[15, 35, 32, 18])[0]
    if node is None:
        n_f = max(n_f, 1)
    chosen = rng.sample(attrs, min(n_f, len(attrs)))
    if "material" in chosen and "leg_material" in chosen and rng.random() < 0.5:
        chosen.remove("leg_material")
    filters = {}
    for a in chosen:
        vals = list(S.ATTRIBUTES[a][1])
        if a == "feature" and l1:
            vals = S.FEATURES_L1[l1]
        if a == "room" and l1:
            vals = S.ROOMS_L1[l1]
        filters[a] = rng.choice(vals)
    n_r = rng.choices([0, 1, 2], weights=[45, 40, 15])[0]
    residual = rng.sample(S.RESIDUAL_PHRASES, n_r)
    typo = rng.random() < 0.12
    g = {"category": node, "filters": filters, "residual": residual, "typo": typo}
    if "width" in filters:
        lo, hi = S.WIDTH_BUCKETS[filters["width"]]
        lo, hi = max(lo, 12), min(hi, 130)
        inch = rng.randrange(lo + (1 if filters["width"] == "over_90in" else 0), hi)
        fmt = rng.choice(["%d inch", "%d in", '%d"', "%d inches", "%d cm", "%s ft"])
        if fmt == "%d cm":
            cm = round(inch * 2.54)
            txt = fmt % cm if lo <= cm / 2.54 < hi else "%d inch" % inch
        elif fmt == "%s ft":
            ft = round(inch / 12.0 * 2) / 2
            txt = (fmt % ("%g" % ft)) if lo <= ft * 12 < hi else "%d inch" % inch
        else:
            txt = fmt % inch
        g["width_text"] = txt
    return g


def _width_hint(bucket):
    lo, hi = S.WIDTH_BUCKETS[bucket]
    if bucket == "under_30in":
        return 'a width under 30 inches (ONE number), e.g. "24 inch", "under 30in", "20\\""'
    if bucket == "over_90in":
        return 'a width over 90 inches (ONE number), e.g. "96 inch", "100\\"", "8 ft"'
    return 'a width between %d and %d inches written as ONE number, e.g. "%d inch", "%d in", "%d\\""' % (lo, hi - 1, (lo + hi) // 2, lo + 2, hi - 2)


def render_intent(g):
    lines = []
    if g["category"]:
        n = S.NODES[g["category"]]
        lines.append('- category: %s (wordings: %s)' % (n.name, ", ".join('"%s"' % s for s in n.synonyms)))
    else:
        lines.append("- category: NONE (do not name any product type such as table, chair, sofa, bed; the query is only "
                     "attributes/residual words, optionally with the generic word 'furniture')")
    for a, v in g["filters"].items():
        if a == "width":
            lines.append('- filter attribute="width" value="%s": write the width as "%s"' % (v, g["width_text"]))
        else:
            forms = S.ATTRIBUTES[a][1][v]
            lines.append('- filter attribute="%s" value="%s" (%s; wordings: %s)' % (
                a, v, S.ATTRIBUTES[a][0], ", ".join('"%s"' % f for f in forms)))
    for p in g["residual"]:
        lines.append('- residual phrase: "%s"' % p)
    if not g["filters"] and not g["residual"]:
        lines.append("- nothing else: the query is just the category wording (1-3 words)")
    if not g["filters"]:
        lines.append("- no filters")
    return "\n".join(lines)


def build_prompt(g):
    cat_note = ""
    if g["category"] and S.NODES[g["category"]].level < 3:
        cat_note = "Keep the category generic - do not name a more specific sub-type.\n"
    typo_note = "- Include exactly one small realistic typo in one word (e.g. a missing or swapped letter)." if g["typo"] else "- No typos."
    return PROMPT.format(intent=render_intent(g), cat_note=cat_note, typo_note=typo_note)


# ---------------------------------------------------------------- validation
def norm(tok):
    return re.sub(r"[^\w$\"'.-]", "", tok.lower()).strip(".,;:!?")


def ratio(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


GENERIC_WORDS = {"furniture", "piece", "pieces", "items", "item", "stuff", "decor", "furnishings"}

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(inch|inches|in\b|\"|''|ft|feet|foot|'|cm)?", re.I)


def width_inches(text):
    m = _NUM_RE.search(text)
    if not m:
        return None
    x, unit = float(m.group(1)), (m.group(2) or "").lower()
    before = text[:m.start()].lower()
    if re.search(r"(under|below|less than|max|up to)\s*$", before):
        x -= 0.5
    elif re.search(r"(over|above|more than|at least|min)\s*$", before):
        x += 0.5
    if unit in ("ft", "feet", "foot", "'"):
        x *= 12
    elif unit == "cm":
        x /= 2.54
    return x


def _term_index():
    terms = []
    for n in S.NODES.values():
        terms += [("category", f) for f in n.synonyms]
    for a, (_, vals) in S.ATTRIBUTES.items():
        for forms in vals.values():
            terms += [("attribute", f) for f in forms]
    return [(kind, f.lower()) for kind, f in terms]


TERMS = _term_index()


def check_residual_leaks(tokens, labels):
    """Residual words must not contain a category or attribute term (catches mislabeling)."""
    i = 0
    while i < len(tokens):
        if labels[i] != "residual":
            i += 1
            continue
        j = i
        while j < len(tokens) and labels[j] == "residual":
            j += 1
        text = " " + " ".join(norm(t) for t in tokens[i:j]) + " "
        for kind, f in TERMS:
            if " " + f + " " in text:
                raise ValueError("residual words %r contain %s term %r - label it correctly or remove it"
                                 % (text.strip(), kind, f))
        i = j


def validate(g, out):
    """Return (row, flags) or raise ValueError."""
    query = out["query"].strip()
    toks = query.split()
    if not toks:
        raise ValueError("empty query")
    ntoks = [norm(t) for t in toks]
    labels = [None] * len(toks)
    spans_out = []
    pos = 0
    flags = []
    for sp in out["spans"]:
        stoks = [norm(t) for t in sp["text"].split()]
        if not stoks:
            continue
        # spans are in query order; search from current position
        found = -1
        for i in range(pos, len(toks) - len(stoks) + 1):
            if ntoks[i:i + len(stoks)] == stoks:
                found = i
                break
        if found < 0:
            raise ValueError("span not found in order: %r" % sp["text"])
        if found != pos:
            raise ValueError("gap before span %r (unlabeled words %r)" % (sp["text"], toks[pos:found]))
        role = sp["role"]
        if role == "category" and g["category"] is None and all(t in GENERIC_WORDS for t in stoks):
            role = "residual"  # "minimalist furniture": generic word, not a category node
            flags.append("relabeled_generic:%s" % sp["text"])
        if role == "filter" and sp.get("attribute") not in g["filters"]:
            # generator often tags words of a requested residual phrase as a filter -> relabel
            res_words = set(w for p_ in g["residual"] for w in re.findall(r"[a-z0-9$]+", p_.lower()))
            if stoks and all(re.sub(r"[^a-z0-9$]", "", t) in res_words for t in stoks):
                role = "residual"
                flags.append("relabeled_residual:%s" % sp["text"])
        for i in range(found, found + len(stoks)):
            labels[i] = role
        pos = found + len(stoks)
        spans_out.append({"start": found, "end": pos, "role": role,
                          "attribute": sp.get("attribute") if role == "filter" else None,
                          "value": sp.get("value") if role == "filter" else None,
                          "text": " ".join(toks[found:pos])})
    if pos != len(toks):
        raise ValueError("unlabeled trailing words %r" % toks[pos:])

    check_residual_leaks(toks, labels)

    # category
    cat_spans = [s for s in spans_out if s["role"] == "category"]
    if g["category"] is None:
        if cat_spans:
            raise ValueError("category span but gold is none")
    else:
        if not cat_spans:
            raise ValueError("missing category span")
        text = " ".join(s["text"] for s in cat_spans)
        syns = S.NODES[g["category"]].synonyms + [S.NODES[g["category"]].name]
        sc = max(ratio(text, s) for s in syns)
        if sc < 0.5:
            raise ValueError("category text %r far from %s" % (text, g["category"]))
        if sc < 0.8:
            flags.append("cat_fuzzy:%.2f" % sc)
        # must not name a different, more specific node (e.g. L2 gold but L3 wording)
        for nid, n in S.NODES.items():
            if nid != g["category"] and any(ratio(text, s) > 0.92 for s in n.synonyms) and sc < 0.92:
                raise ValueError("category text %r matches other node %s" % (text, nid))

    # filters
    got = {}
    for s in spans_out:
        if s["role"] != "filter":
            continue
        a, v = s["attribute"], s["value"]
        if a not in g["filters"]:
            raise ValueError("extra filter attribute %r (%r)" % (a, s["text"]))
        if v != g["filters"][a]:
            raise ValueError("filter %s value %r != gold %r" % (a, v, g["filters"][a]))
        got.setdefault(a, []).append(s["text"])
    for a, v in g["filters"].items():
        if a not in got:
            raise ValueError("missing filter %s" % a)
        text = " ".join(got[a])
        if a == "width":
            w = width_inches(text)
            lo, hi = S.WIDTH_BUCKETS[v]
            if w is None or not (lo <= w < hi):
                raise ValueError("width %r not in bucket %s" % (text, v))
        else:
            forms = S.ATTRIBUTES[a][1][v] + [v.replace("_", " ")]
            sc = max(max(ratio(text, f) for f in forms),
                     max(1.0 if f.lower() in text.lower() else 0 for f in forms))
            if sc < 0.5:
                raise ValueError("filter %s text %r far from %s" % (a, text, v))
            if sc < 0.8:
                flags.append("filter_fuzzy:%s:%.2f" % (a, sc))

    # residual phrases: soft check
    res_text = " ".join(s["text"] for s in spans_out if s["role"] == "residual").lower()
    for p in g["residual"]:
        if not any(w in res_text for w in re.findall(r"[a-z0-9$]{3,}", p.lower())):
            flags.append("residual_missing:%s" % p)
    if not g["residual"] and res_text:
        flags.append("residual_extra:%s" % res_text)

    row = {
        "query": query,
        "tokens": toks,
        "token_labels": labels,
        "spans": spans_out,
        "category": g["category"],
        "category_path": S.NODES[g["category"]].path() if g["category"] else [],
        "filters": g["filters"],
        "residual_tokens": [toks[i] for i, l in enumerate(labels) if l == "residual"],
        "gold_intent": g,
        "flags": flags,
    }
    return row


def generate_one(idx, seed, max_tries=4):
    rng = random.Random(seed * 1000003 + idx)
    g = sample_gold(rng)
    errors = []
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(g)}]
    for attempt in range(max_tries):
        raw = None
        try:
            raw = chat(GEN_MODEL, messages, fmt=OUT_FORMAT,
                       temperature=0.9, seed=seed * 7919 + idx * 31 + attempt)
            row = validate(g, json.loads(raw))
            row["id"] = idx
            row["gen_attempts"] = attempt + 1
            return row, errors
        except Exception as e:  # noqa: BLE001 - collect and retry with feedback
            errors.append(str(e)[:200])
            if raw is not None:
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That answer is invalid: %s. Write a new query that expresses "
                                                "exactly the requested intent (no other attributes) and label "
                                                "every word following the rule." % e}]
    return None, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/eval_raw.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [i for i in range(args.start, args.start + args.n) if i not in done]
    lock = threading.Lock()
    n_ok = n_fail = n_retry = 0
    err_log = open(str(out_path) + ".errors.log", "a")
    with open(out_path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(generate_one, i, args.seed): i for i in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            row, errors = fut.result()
            with lock:
                for e in errors:
                    err_log.write("%d\t%s\n" % (futs[fut], e))
                if row:
                    n_ok += 1
                    n_retry += row["gen_attempts"] > 1
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                else:
                    n_fail += 1
                if k % 25 == 0 or k == len(todo):
                    print("%d/%d ok=%d failed=%d retried=%d" % (k, len(todo), n_ok, n_fail, n_retry), flush=True)
    err_log.close()
    print("done ok=%d failed=%d retried=%d" % (n_ok, n_fail, n_retry))


if __name__ == "__main__":
    sys.exit(main())
