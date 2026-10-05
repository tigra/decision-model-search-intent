"""Are a decision model's answers independent of the other questions in the request?

  python -m sidm.independence --backend jev            (make independence BACKEND=jev)

Same state (the sweep's QUERY) in every request. The full 29-question request is sent twice (determinism
baseline); then the K least confident questions of that answer are re-asked
  - alone,
  - with 5 other questions,
and once each in variants of the full request: question order reversed, one other question reworded, and
10 unrelated extra questions added.

If the model evaluates every question on its own (shared state, one branch per question), each target's
probabilities are identical in every variant. If questions share one context (nimble's layout: all question
texts in one prompt), they shift. Writes results/bench/independence_<label>.json.
"""
import argparse
import json
from pathlib import Path

from sidm.bench import QUERY, SCHEME
from sidm.evaluate import run_label
from sidm.ollama_client import DEFAULT_MODELS, friendly_main, system_one
from sidm.parser import build_request

EXTRA = {  # unrelated questions, short, two or three options
    "extra_%d" % i: {"type": "choice", "instructions": text, "criteria": {"yes": None, "no": None}}
    for i, text in enumerate([
        "Is the query written in English?", "Does the query mention a price?", "Does the query mention a brand?",
        "Is the query a question?", "Does the query mention delivery?", "Is the query longer than five words?",
        "Does the query mention a material?", "Does the query mention a room?", "Is a color mentioned?",
        "Does the query contain a number?"])
}


def out_path(backend, model):
    return Path("results/bench/independence_%s.json" % run_label(model, backend))


def max_diff(a, b):
    """Largest absolute difference between two probability dicts over the same options."""
    return max(abs(a["probabilities"][k] - b["probabilities"].get(k, 0.0)) for k in a["probabilities"])


@friendly_main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="jev", choices=sorted(DEFAULT_MODELS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--targets", type=int, default=3, help="how many of the least confident questions to test")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]
    state, questions, _ = build_request(QUERY, SCHEME)

    def ask(qs):
        return system_one(model, state, qs, backend=args.backend)[0]["answers"]

    runs = {"full": ask(questions), "full_again": ask(questions)}
    conf = sorted(questions, key=lambda k: runs["full"][k]["confidence"])
    targets = conf[:args.targets]
    others = [k for k in questions if k not in targets]
    reworded_key = next(k for k in others if k.startswith("filter_"))
    reworded = dict(questions)
    reworded[reworded_key] = dict(questions[reworded_key],
                                  instructions="Which value of this attribute does the shopper ask for, if any?")
    runs["reversed"] = ask(dict(reversed(list(questions.items()))))
    runs["reworded_%s" % reworded_key] = ask(reworded)
    runs["plus_10_extra"] = ask(dict(questions, **EXTRA))
    for t in targets:
        runs["alone_%s" % t] = ask({t: questions[t]})
        runs["with_5_%s" % t] = ask({t: questions[t], **{k: questions[k] for k in others[:5]}})

    base = runs["full"]
    summary = {}
    for t in targets:
        rows = {}
        for name, ans in runs.items():
            if name == "full" or t not in ans or (name.startswith(("alone_", "with_5_")) and not name.endswith(t)):
                continue
            rows[name] = {"max_prob_diff": round(max_diff(base[t], ans[t]), 4),
                          "choice_changed": ans[t]["choice"] != base[t]["choice"]}
        summary[t] = {"full_answer": base[t], "vs_full": rows}
    out = out_path(args.backend, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "model": model, "query": QUERY, "targets": targets,
                               "summary": summary, "answers": runs}, indent=1))
    for t, s in summary.items():
        print("%s  (full: %s, confidence %.2f)" % (t, s["full_answer"]["choice"], s["full_answer"]["confidence"]))
        for name, r in s["vs_full"].items():
            print("  %-34s max |Δp| %.3f%s" % (name, r["max_prob_diff"], "  choice changed" if r["choice_changed"] else ""))
    print("written to %s" % out)


if __name__ == "__main__":
    main()
