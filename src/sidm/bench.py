"""Latency vs number of questions: one realistic request, sent with its first N questions (make sweep).

  python -m sidm.bench --backend ollama --model nimble [--repeats 3] [--overwrite]

The request is the embedded-scheme request for QUERY (category, then filter, then word questions, in that
order); N runs from 1 to all of them, each N `--repeats` times, in a shuffled (seeded) order so thermal or
memory drift spreads over all N.

Every request starts with a first word no other request uses (a nonce: random per process + request number).
The query sits right after the template preamble, so this cuts the prefix shared with *any* earlier prompt to
the preamble. Changing the first word only between consecutive requests is not enough: Ollama's llama.cpp
runner keeps a RAM cache of earlier prompts (`cache state: N prompts` in server.log, up to ~8 GB) and resumes
from the best match, so a repeated first word lets a request skip most of its prefill.

Writes results/bench/sweep_<label>.jsonl (one row per request; resumable). Run with nothing else on the machine.
"""
import argparse
import json
import random
import time
from pathlib import Path

from sidm.evaluate import run_label
from sidm.ollama_client import DEFAULT_MODELS, friendly_main, system_one_split
from sidm.parser import build_request

QUERY = "grey mid century modern sectional with oak legs cheap free shipping"  # 11 words -> 29 questions
NONCE = "%04x" % random.SystemRandom().randrange(16 ** 4)  # new per process: no reuse across runs either
SCHEME = "embedded"


def sweep_path(backend, model):
    return Path("results/bench/sweep_%s.jsonl" % run_label(model, backend))


def plan(n_max, repeats, seed=0):
    jobs = [(n, rep) for n in range(1, n_max + 1) for rep in range(repeats)]
    random.Random(seed).shuffle(jobs)
    return jobs


def request(k, n):
    """Request number k (in run order) with the first n questions; no other request has its first word."""
    query = " ".join(["x%s%03d" % (NONCE, k)] + QUERY.split()[1:])
    state, questions, _ = build_request(query, SCHEME)
    return query, state, dict(list(questions.items())[:n])


@friendly_main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="ollama", choices=sorted(DEFAULT_MODELS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true", help="delete this model's stored sweep and start over")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]
    out = sweep_path(args.backend, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out.exists():
        out.unlink()
    done = {(r["n_questions"], r["repeat"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    n_max = len(request(0, 64)[2])
    jobs = plan(n_max, args.repeats)
    todo = [(k, n, rep) for k, (n, rep) in enumerate(jobs) if (n, rep) not in done]
    if not todo:
        print("%s: all %d requests already done (--overwrite / OVERWRITE=1 to re-run)" % (out, len(jobs)))
        return
    for warm in ("warm up sofa", "blue oak dining table"):  # model load, not timed
        state, questions, _ = build_request(warm, SCHEME)
        system_one_split(model, state, questions, backend=args.backend)
    t0 = time.time()
    with open(out, "a") as f:
        for i, (k, n, rep) in enumerate(todo, 1):
            query, state, questions = request(k, n)
            resp, latency, n_requests = system_one_split(model, state, questions, backend=args.backend)
            row = {"backend": args.backend, "model": model, "served_model": resp.get("model"), "query": query,
                   "n_questions": n, "repeat": rep, "order": k, "latency_s": latency, "n_requests": n_requests,
                   "usage": resp.get("usage"), "server_metrics": resp.get("metrics")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print("%d/%d  %.0fs elapsed, N=%d took %.2fs" % (i, len(todo), time.time() - t0, n, latency), flush=True)
    print("written to %s" % out)


if __name__ == "__main__":
    main()
