"""Check what this machine has for the study: servers, models, keys, data, and which runs can be re-run.

  python -m sidm.doctor        (make doctor)

Read-only. Every problem line comes with the command that fixes it.
"""
import glob
import json
import os
import shutil
import urllib.request
from pathlib import Path

from sidm.ollama_client import BACKENDS, ENV_FILE, ENV_LOADED, friendly_main
from sidm.runs import DEV_RUNS, EVAL_RUNS, run_status, split_run

MODELS_DIR = Path(os.environ.get("SIDM_MODELS_DIR", Path.home() / "sidm-models"))
OLLAYA_BIN = MODELS_DIR / "ollaya" / "bin" / "ollaya"
OK, BAD, NA = "✓", "✗", "–"


def _get(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return None


def _names(tags):
    """Model names from an Ollama/Ollaya /api/tags response, with ':latest' dropped."""
    names = set()
    for m in (tags or {}).get("models", []):
        n = m.get("name") or m.get("model") or ""
        names.add(n[:-len(":latest")] if n.endswith(":latest") else n)
    return names


def _line(mark, text, fix=None):
    print("  %s %s" % (mark, text) + ("\n      fix: %s" % fix if fix else ""))


def _version_ok(v, minimum=(0, 35)):
    try:
        return tuple(int(x) for x in v.split(".")[:2]) >= minimum
    except ValueError:
        return False


@friendly_main
def main():
    state = {}
    print("Environment")
    free = shutil.disk_usage(Path.home()).free / 1e9
    _line(OK if free > 20 else BAD, "free disk: %.0f GB" % free,
          None if free > 20 else "free space: model downloads need 5-40 GB (make ollama-rm / ollaya-rm MODEL=...)")
    rows = sum(1 for _ in open("data/eval_raw.jsonl")) if Path("data/eval_raw.jsonl").exists() else 0
    _line(OK if rows >= 1100 else BAD, "dataset data/eval_raw.jsonl: %d rows (dev 100 + eval 1000)" % rows,
          None if rows >= 1100 else "the dataset ships with the repo; restore data/eval_raw.jsonl")

    print("\nOllama (backend `ollama`, %s)" % BACKENDS["ollama"].url)
    v = _get(BACKENDS["ollama"].url + "/api/version")
    if not v:
        _line(BAD, "not running", "start the Ollama app (or `ollama serve`); /v1/systemone needs Ollama >= 0.35")
        state["ollama"] = None
    else:
        _line(OK if _version_ok(v["version"]) else BAD, "running, version %s" % v["version"],
              None if _version_ok(v["version"]) else "update Ollama to >= 0.35 (decision models / /v1/systemone)")
        state["ollama"] = _names(_get(BACKENDS["ollama"].url + "/api/tags"))
        _line(OK, "models: %s" % (", ".join(sorted(state["ollama"])) or "none"))

    print("\nOllaya (backend `ollaya`, %s)" % BACKENDS["ollaya"].url)
    if not OLLAYA_BIN.exists():
        _line(BAD, "not installed at %s" % OLLAYA_BIN,
              "OLLAYA_INSTALL_DIR=%s/ollaya OLLAYA_NO_SERVICE=1 sh <(curl -fsSL https://ollaya.dev/install.sh)" % MODELS_DIR)
        state["ollaya"] = None
    else:
        tags = _get(BACKENDS["ollaya"].url + "/api/tags")
        if tags is None:
            _line(BAD, "installed, server not running", "make ollaya-serve")
            state["ollaya"] = None
        else:
            state["ollaya"] = _names(tags)
            _line(OK, "running; models: %s" % (", ".join(sorted(state["ollaya"])) or "none"))

    print("\nMLX ParallelScorer (backend `mlx`, %s)" % BACKENDS["mlx"].url)
    configs = glob.glob(str(MODELS_DIR / "*" / "nimble-model.json"))
    if not configs:
        _line(BAD, "no converted nimble model under %s" % MODELS_DIR, "make mlx-convert   (~40 GB peak disk, ~10.5 GB after)")
    else:
        _line(OK, "converted model: %s" % Path(configs[0]).parent.name)
    health = _get(BACKENDS["mlx"].url + "/health")
    state["mlx"] = bool(configs) and health is not None
    _line(OK if health else BAD, "server running" if health else "server not running",
          None if health else ("make mlx-serve" if configs else "make mlx-convert, then make mlx-serve"))

    print("\nStrands Decider (backend `decider`, %s)" % BACKENDS["decider"].url)
    if not Path("decider_backend/.venv").exists():
        _line(BAD, "environment not set up", "make decider-setup")
    up = _get(BACKENDS["decider"].url + "/openapi.json") is not None
    state["decider"] = up
    _line(OK if up else BAD, "server running" if up else "server not running",
          None if up else "make decider-serve   (first start downloads ~4 GB)")

    print("\nJev (backend `jev`, %s)" % BACKENDS["jev"].url)
    key = os.environ.get("TYPESAFE_API_KEY")
    state["jev"] = bool(key)
    source = "from .env" if "TYPESAFE_API_KEY" in ENV_LOADED else "from the environment"
    _line(OK if key else BAD, "TYPESAFE_API_KEY %s" % ("set (%s)" % source if key else "not set"),
          None if key else "put TYPESAFE_API_KEY=<key> in %s (cp .env.example .env), or export it   "
                           "(only needed for jev runs; paid API)" % ENV_FILE.name)

    print("\nRuns of the study (src/sidm/runs.py): can they be re-run here?")
    runnable = 0
    runs = list(dict.fromkeys(DEV_RUNS + EVAL_RUNS + ["embedded@jev:jev-latest"]))
    width = max(map(len, runs))
    for run in runs:
        scheme, backend, model = split_run(run)
        st = run_status(run)["splits"]
        stored = "stored dev %d/100, eval %d/1000" % (st["dev"]["rows"], st["eval"]["rows"])
        avail, fix = True, None
        if backend in ("ollama", "ollaya"):
            names = state[backend]
            if names is None:
                avail, fix = False, "start the %s server first (see above)" % backend
            elif model not in names:
                avail, fix = False, "make %s-pull MODEL=%s" % (backend, model)
        elif backend == "mlx":
            avail = state["mlx"]
            fix = None if avail else ("make mlx-serve" if configs else "make mlx-convert, then make mlx-serve")
        elif backend == "decider":
            avail, fix = state["decider"], None if state["decider"] else "make decider-serve"
        elif backend == "jev":
            avail, fix = state["jev"], None if state["jev"] else "put TYPESAFE_API_KEY=<key> in .env"
        runnable += avail
        _line(OK if avail else BAD, "%-*s  %s" % (width, run, stored), fix)
    print("\n%d of %d runs can be re-run on this machine right now (one model on the GPU at a time)." % (runnable, len(runs)))
    print("Results ship with the repo: results/results_eval.md, results/mcnemar_eval.txt; `make runs` lists them.")


if __name__ == "__main__":
    main()
