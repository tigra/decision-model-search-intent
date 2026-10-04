"""Parse one search query with a decision model and print the result.

  python -m sidm.cli "grey oak coffee table with storage cheap"          # result only
  python -m sidm.cli -v   "..."   # + per-word roles, category candidates, filter confidences / dropped filters
  python -m sidm.cli -vv  "..."   # + top options of every question, token usage
  python -m sidm.cli -vvv "..."   # + raw request and response JSON
  python -m sidm.cli --json "..." # machine-readable decoded result
  python -m sidm.cli --scheme router "..."     # category scheme with the top-level question (see parser.py)
  python -m sidm.cli --backend mlx "..."       # nimble's MLX ParallelScorer (start mlx_backend/serve.sh first)
  python -m sidm.cli --backend jev "..."       # TypeSafe's hosted Jev (export TYPESAFE_API_KEY)
"""
import argparse
import json
import sys

from sidm import schema as S
from sidm.ollama_client import BACKENDS, DEFAULT_MODELS, friendly_main, system_one_split
from sidm.parser import (DEFAULT_SCHEME, NOT_SPEC, OTHER, SCHEMES, build_request, tuned_settings,
                         category_candidates, decode, none_signal)


def _path(node_id):
    return " > ".join(S.NODES[n].name for n in S.NODES[node_id].path()) if node_id else "(none)"


def _top(probs, k=3):
    return sorted(probs.items(), key=lambda kv: -kv[1])[:k]


def _fmt_top(probs, k=3):
    return "  ".join("%s %.2f" % (c, p) for c, p in _top(probs, k))


@friendly_main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Parse a furniture search query into category, filters and residual words.")
    ap.add_argument("query", nargs="+", help="search query (quotes optional)")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v, -vv, -vvv for more detail")
    ap.add_argument("--model", default=None, help="default per backend: %s" % DEFAULT_MODELS)
    ap.add_argument("--scheme", default=DEFAULT_SCHEME, choices=sorted(SCHEMES), help="category question scheme")
    ap.add_argument("--backend", default="ollama", choices=sorted(BACKENDS),
                    help="ollama; mlx = nimble's ParallelScorer server (mlx_backend/serve.sh); "
                         "jev = TypeSafe's hosted API (TYPESAFE_API_KEY)")
    ap.add_argument("--untuned", action="store_true", help="plain argmax decoding instead of the dev-tuned one")
    ap.add_argument("--json", action="store_true", help="print the decoded result as JSON")
    args = ap.parse_args(argv)

    query = " ".join(args.query)
    scheme = SCHEMES[args.scheme]
    args.model = args.model or DEFAULT_MODELS[args.backend]
    state, questions, tokens = build_request(query, scheme)
    request = {"model": args.model, "state": state, "questions": questions}
    resp, latency, n_requests = system_one_split(args.model, state, questions, backend=args.backend)
    ans = resp["answers"]
    dec = decode(resp, tokens, scheme, tuned=not args.untuned, backend=args.backend, model=args.model)
    cfg, tuned_scheme = tuned_settings(args.backend, scheme, args.model)

    if args.json:
        out = {k: dec[k] for k in ("category", "filters", "filters_raw", "token_labels", "residual_tokens")}
        out.update(query=query, scheme=scheme.name, backend=args.backend, model=args.model,
                   served_model=resp.get("model"), server_metrics=resp.get("metrics"),
                   category_path=S.NODES[dec["category"]].path() if dec["category"] else [],
                   latency_s=round(latency, 3), usage=resp.get("usage"))
        if args.verbose >= 3:
            out.update(request=request, response=resp)
        print(json.dumps(out, indent=1))
        return 0

    v = args.verbose
    print("query     %s" % query)
    print("category  %s" % _path(dec["category"]))
    print("filters   %s" % (", ".join("%s=%s" % kv for kv in dec["filters"].items()) or "-"))
    print("residual  %s" % (" ".join(dec["residual_tokens"]) or "-"))
    print("latency   %.2fs  (%d questions in %d request%s, scheme %s, backend %s, model %s%s)" % (
        latency, len(questions), n_requests, "s" if n_requests > 1 else "", scheme.name, args.backend,
        resp.get("model") or args.model,
        ", untuned decoding" if args.untuned else ""))
    if v >= 1 and resp.get("metrics"):
        m = resp["metrics"]
        print("server    prefill %.2fs (%d tokens), batched field evaluation %.2fs (%d batch), peak %.1f GiB" % (
            m["prefill_seconds"], m["prefix_tokens"], m["field_evaluation_seconds"], m["branch_batches"],
            m["mlx_peak_active_gib"]))

    if v >= 1:
        print("\nwords")
        for i, (w, lab) in enumerate(zip(tokens, dec["token_labels"]), 1):
            a = ans.get("word_%d" % i)
            probs = "  ".join("%s %.2f" % (c[:3], a["probabilities"][c]) for c in ("category", "filter", "residual")) if a else "(truncated)"
            print("  %2d %-16s %-9s %s" % (i, w, lab, probs))

        print("\ncategory candidates (group-max)")
        print("  'no category' signal %.2f (threshold %.2f): %s" % (
            none_signal(ans, tuned_scheme), tuned_scheme.none_thr,
            "P(cat_L1 = none)" if scheme.router else "1 - best candidate probability"))
        for node, p, q in category_candidates(ans)[:3]:
            print("  %.2f  %-24s %s  [%s]" % (p, node, _path(node), q))
        if scheme.router:
            print("  cat_L1 (top-level question, not used for the group): %s" % _fmt_top(ans["cat_L1"]["probabilities"]))
        else:
            print("  P(%s) per group: %s" % (OTHER, "  ".join(
                "%s %.2f" % (q[4:], a["probabilities"].get(OTHER, 0.0)) for q, a in ans.items() if q.startswith("cat_"))))

        print("\nfilters (threshold %.2f%s)" % (0.0 if args.untuned else cfg["filter_min_prob"],
                                                "" if args.untuned else ", masked by category applicability"))
        for attr in S.ATTRIBUTES:
            a = ans["filter_" + attr]
            c = a["choice"]
            if c == NOT_SPEC:
                if v >= 2:
                    print("  %-12s -           (not_specified %.2f)" % (attr, a["probabilities"][NOT_SPEC]))
                continue
            p = a["probabilities"][c]
            if attr in dec["filters"]:
                status = "kept"
            elif attr in dec["filters_raw"]:
                status = "dropped: not applicable to category"
            else:
                status = "dropped: below threshold"
            print("  %-12s %-12s %.2f  %s" % (attr, c, p, status))

    if v >= 2:
        print("\nall questions (top 3 options)")
        for name, a in ans.items():
            print("  %-20s %-22s conf %.2f | %s" % (name, a.get("choice"), a.get("confidence", 0.0), _fmt_top(a["probabilities"])))
        print("\nusage     %s" % resp.get("usage"))

    if v >= 3:
        print("\nrequest\n" + json.dumps(request, indent=1))
        print("\nresponse\n" + json.dumps(resp, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
