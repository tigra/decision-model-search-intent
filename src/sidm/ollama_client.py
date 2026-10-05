"""Minimal stdlib client for Ollama chat and the /v1/systemone decision endpoint.

/v1/systemone is served by ablatable backends with the same contract:
  ollama: Ollama's llama.cpp runner (questions scored sequentially)
  mlx:    nimble's MLX ParallelScorer via mlx_backend/server.py
  ollaya: Ollaya's local server (encoder classifiers and decoders; scripts/ollaya_serve.sh)
  decider: AWS Strands Labs' Strands Decider server (decider_backend/serve.sh)
  jev:    TypeSafe's hosted Jev API (needs TYPESAFE_API_KEY)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")


class SidmError(RuntimeError):
    """A user-facing problem with a known fix; entry points print it without a traceback."""

    def __init__(self, message, fix=None):
        super().__init__(message)
        self.fix = fix


def friendly_main(main):
    """Decorator for CLI entry points: print SidmError / missing-file errors as `error:` + `fix:`, exit 1."""
    import functools

    @functools.wraps(main)
    def wrapper(*args, **kwargs):
        try:
            return main(*args, **kwargs)
        except SidmError as e:
            print("error: %s" % e, file=sys.stderr)
            if e.fix:
                print("fix:   %s" % e.fix, file=sys.stderr)
        except FileNotFoundError as e:
            print("error: file not found: %s" % (e.filename or e), file=sys.stderr)
            if str(e.filename or "").endswith(".jsonl"):
                print("fix:   this run has no predictions yet; create them with "
                      "`make dev BACKEND=… MODEL=…` (or `make eval`). `make runs` lists what exists.", file=sys.stderr)
        except KeyboardInterrupt:
            print("interrupted (runs are resumable: re-run the same command to continue)", file=sys.stderr)
        sys.exit(1)
    return wrapper


class Backend:
    def __init__(self, url, default_model, api_key_env=None, remote=False):
        self.url, self.default_model, self.api_key_env, self.remote = url, default_model, api_key_env, remote

    def headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise SidmError("%s is not set (needed for this backend)" % self.api_key_env,
                                "export %s=<your key>" % self.api_key_env)
            h["Authorization"] = "Bearer " + key
        return h


BACKENDS = {
    "ollama": Backend(HOST, "nimble"),
    "mlx": Backend(os.environ.get("SIDM_MLX_URL", "http://localhost:11500"), "nimble"),
    "ollaya": Backend(os.environ.get("SIDM_OLLAYA_URL", "http://localhost:11435"), "jeb:4b"),
    # serves one checkpoint (StrandsAgents/strands-decider-2B-hobson-v19); the model field is a label
    "decider": Backend(os.environ.get("SIDM_DECIDER_URL", "http://localhost:11600"), "strands-decider-2b"),
    # jev-latest moves with releases; the versioned model that answered is recorded from the response.
    "jev": Backend(os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
                   os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest"), api_key_env="TYPESAFE_API_KEY",
                   remote=True),
}
DEFAULT_MODELS = {name: b.default_model for name, b in BACKENDS.items()}


def _post(path, payload, timeout=600, host=HOST, headers=None):
    req = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode(),
        headers=headers or {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:  # keep the server's error message (e.g. prompt too long, bad key)
        body = e.read().decode(errors="replace")[:500]
        raise urllib.error.HTTPError(e.url, e.code, "%s: %s" % (e.reason, body), e.headers, None) from None


def chat(model, messages, fmt=None, temperature=0.8, seed=None, think=False, num_ctx=4096):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    if fmt is not None:
        payload["format"] = fmt
    return _post("/api/chat", payload)["message"]["content"]


RETRY_CODES = {429, 500, 502, 503, 504}
START_FIX = {
    "ollama": "start the Ollama app (or `ollama serve`)",
    "mlx": "make mlx-serve   (first time: make mlx-convert)",
    "ollaya": "make ollaya-serve",
    "decider": "make decider-serve   (first time: make decider-setup)",
    "jev": "check the network connection and TYPESAFE_BASE_URL",
}
PULL_FIX = {
    "ollama": "make ollama-pull MODEL=%s",
    "ollaya": "make ollaya-pull MODEL=%s",
    "mlx": "make mlx-convert   (the MLX server serves one converted nimble model)",
    "decider": "make decider-serve   (it serves one checkpoint; weights download on first start)",
    "jev": "check the model name (e.g. jev-latest, jev-1.13.0)",
}


def _friendly(e, backend, model):
    """Map transport errors to SidmError with a fix; return None to re-raise the original."""
    b = BACKENDS[backend]
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404 and ("not found" in str(e.reason).lower() or "MODEL_NOT_FOUND" in str(e.reason)):
            return SidmError("model '%s' isn't available on the %s backend" % (model, backend),
                             PULL_FIX[backend] % model if "%s" in PULL_FIX[backend] else PULL_FIX[backend])
        if e.code in (401, 403):
            return SidmError("%s rejected the request (HTTP %d): %s" % (backend, e.code, e.reason),
                             "check %s" % (b.api_key_env or "the server's API key settings"))
        return None
    if isinstance(e, urllib.error.URLError):
        return SidmError("%s server not reachable at %s (%s)" % (backend, b.url, e.reason), START_FIX[backend])
    return None


def system_one(model, state, questions, keep_alive="30m", backend="ollama", max_retries=5):
    """Returns (response_json, wall_latency_seconds of the successful attempt).

    Remote backends are retried with exponential backoff on rate limits and server errors.
    """
    b = BACKENDS[backend]
    payload = {"model": model, "state": state, "questions": questions}
    if backend == "ollama":
        payload["keep_alive"] = keep_alive  # Ollama-specific; not part of the TypeSafe contract
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = _post("/v1/systemone", payload, host=b.url, headers=b.headers())
            return resp, time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if not b.remote or e.code not in RETRY_CODES or attempt == max_retries:
                friendly = _friendly(e, backend, model)
                if friendly:
                    raise friendly from None
                raise
            wait = min(60, 2 ** attempt)
            print("%s: HTTP %d, retrying in %ds" % (backend, e.code, wait), file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        except (urllib.error.URLError, ConnectionError) as e:
            if b.remote and attempt < max_retries:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise (_friendly(urllib.error.URLError(e) if isinstance(e, ConnectionError) else e, backend, model)
                   or e) from None


# ---------------------------------------------------------------- splitting oversized requests
import math
import re

_TOO_LONG = re.compile(r"has (\d+) tokens; expected 1[^0-9]+(\d+)")
_LEARNED_CHUNKS = {}  # (backend, model) -> number of requests needed for a full question set


def _chunks(names, k):
    size = math.ceil(len(names) / k)
    return [names[i:i + size] for i in range(0, len(names), size)]


def system_one_split(model, state, questions, backend="ollama", learned=True, start=None, **kw):
    """Like system_one, but splits the questions over several requests when the rendered prompt exceeds the
    model's context (e.g. tev1's 2k window in Ollama; the server reports `has N tokens; expected 1–M`).

    Questions are answered independently anyway; a split only removes the other questions from the context.
    The split that worked is remembered per (backend, model) and tried first next time. `learned=False` starts
    from one request instead (the fewest requests for this question set); `start` forces the first try.
    Returns (merged_response, summed_latency_of_successful_requests, n_requests actually sent).
    """
    names = list(questions)
    k = start or (_LEARNED_CHUNKS.get((backend, model), 1) if learned else 1)
    while True:
        answers, usage, latency, models = {}, {"input_tokens": 0, "output_tokens": 0}, 0.0, set()
        parts = _chunks(names, k)
        try:
            for part in parts:
                resp, lat = system_one(model, state, {n: questions[n] for n in part}, backend=backend, **kw)
                answers.update(resp["answers"])
                for key in usage:
                    usage[key] += (resp.get("usage") or {}).get(key, 0)
                latency += lat
                models.add(resp.get("model"))
        except urllib.error.HTTPError as e:
            m = _TOO_LONG.search(str(e.reason)) if e.code == 400 else None
            if not m or k >= len(names):
                raise
            have, limit = int(m.group(1)), int(m.group(2))
            # tokens shrink less than proportionally (state + template are repeated), so leave headroom
            k = min(len(names), max(k + 1, math.ceil(k * have / (limit * 0.8))))
            _LEARNED_CHUNKS[(backend, model)] = k
            continue
        merged = {"model": "/".join(sorted(m for m in models if m)), "answers": answers, "usage": usage}
        if len(parts) == 1 and "metrics" in resp:  # server-side timings (mlx backend) are per request
            merged["metrics"] = resp["metrics"]
        resp = merged
        return resp, latency, len(parts)
