"""Registry of the study's runs and their status on this machine.

A run is <scheme>@<backend>:<model>. EVAL_RUNS / DEV_RUNS are the runs in the shipped result tables
(`make results` builds them via `results_table --preset eval|dev`); the first run is the reference for
the "vs first run" significance tests.

  python -m sidm.runs                 # status table: predictions done per split, tuned settings, served model
  python -m sidm.runs --reports eval  # detailed per-run reports for the registry's eval runs (part of make results)
"""
import json
import time
from pathlib import Path

from sidm.ollama_client import DEFAULT_MODELS, friendly_main
from sidm.parser import TUNED_BY_BACKEND

EVAL_RUNS = [
    "embedded@ollama:nimble",
    "router@ollama:nimble",
    "embedded@ollama:tev1",
    "embedded@ollama:tev1:0.8b",
    "embedded@ollaya:jeb:4b",
    "embedded@ollaya:winnow:e4b",
]
DEV_RUNS = [
    "embedded@ollama:nimble",
    "router@ollama:nimble",
    "embedded@mlx:nimble",
    "embedded@ollama:tev1",
    "embedded@ollama:tev1:0.8b",
    "embedded@ollaya:jeb:4b",
    "embedded@ollaya:winnow:e4b",
    "embedded@ollaya:decider:2b",
    "embedded@ollaya:decision:eos",
    "embedded@ollaya:kev:0.8b",
    "embedded@ollaya:laya:en",
    "embedded@ollaya:laya:typed-decisions",
    "embedded@ollaya:von",
    "embedded@ollaya:nli:modernbert-large",
]
PRESETS = {"eval": EVAL_RUNS, "dev": DEV_RUNS}
SPLIT_SIZES = {"dev": 100, "eval": 1000}


def split_run(run):
    """'<scheme>@<backend>[:<model>]' -> (scheme, backend, model) with the backend's default model."""
    scheme, _, rest = run.partition("@")
    backend, _, model = rest.partition(":")
    return scheme, backend, model or DEFAULT_MODELS[backend]


def _first_pred(path):
    with open(path) as f:
        line = f.readline()
    return json.loads(line)["pred"] if line.strip() else {}


def known_runs():
    """Registry runs first, then any other run that has prediction files in results/."""
    runs = list(dict.fromkeys(DEV_RUNS + EVAL_RUNS))
    for path in sorted(Path("results").glob("*_*_*.jsonl")):
        if not path.name.startswith(("dev_", "eval_")) or path.stat().st_size == 0:
            continue
        p = _first_pred(path)
        backend = p.get("backend", "ollama")
        run = "%s@%s:%s" % (p.get("scheme", "router"), backend, p.get("model") or DEFAULT_MODELS[backend])
        if run not in runs:
            runs.append(run)
    return runs


def run_status(run):
    """Per split: rows done / split size, last update, served model; plus whether tuned settings exist."""
    from sidm.evaluate import pred_path
    scheme, backend, model = split_run(run)
    status = {"run": run, "backend": backend, "model": model,
              "tuned": "%s:%s" % (backend, model) in TUNED_BY_BACKEND, "splits": {}}
    for split, size in SPLIT_SIZES.items():
        path = pred_path(split, model, scheme, backend)
        if path.exists() and path.stat().st_size:
            with open(path) as f:
                rows = sum(1 for line in f if line.strip())
            status["splits"][split] = {"rows": rows, "size": size, "path": str(path),
                                       "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                                       "served_model": _first_pred(path).get("served_model")}
        else:
            status["splits"][split] = {"rows": 0, "size": size, "path": str(path)}
    return status


def _cell(s):
    if not s["rows"]:
        return "–"
    mark = "" if s["rows"] >= s["size"] else " partial"
    return "%d/%d%s" % (s["rows"], s["size"], mark)


def write_reports(split="eval", data="data/eval_raw.jsonl"):
    """Detailed per-run reports (per-attribute filter scores, category confusions, 20 worst queries) for the
    registry runs of `split`: results/report_<split>_<label>_<scheme>.md, the same name `make report` uses."""
    import contextlib
    import io
    from types import SimpleNamespace
    from sidm.evaluate import pred_path, report
    for run in PRESETS[split]:
        scheme, backend, model = split_run(run)
        pred = pred_path(split, model, scheme, backend)
        if not pred.exists():
            print("skipping report for %s: no %s predictions" % (run, split))
            continue
        md = Path("results/report_%s.md" % pred.stem)
        with contextlib.redirect_stdout(io.StringIO()):  # report() echoes the table; keep this quiet
            report(SimpleNamespace(data=data, pred=str(pred), model=model, md=str(md)))
        print("written to %s" % md)


@friendly_main
def main():
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--reports":
        return write_reports(sys.argv[2])
    rows = [run_status(r) for r in known_runs()]
    width = max(len(s["run"]) for s in rows)
    print("%-*s  %-14s %-15s %-10s %s" % (width, "run", "dev", "eval", "decoding", "last update"))
    for s in rows:
        d, e = s["splits"]["dev"], s["splits"]["eval"]
        updated = max((x.get("updated", "") for x in (d, e)), default="")
        print("%-*s  %-14s %-15s %-10s %s" % (width, s["run"], _cell(d), _cell(e),
                                               "per-model" if s["tuned"] else "default", updated))
    missing = [s for s in rows if not s["splits"]["dev"]["rows"]]
    print("\nlegend: n/size = predictions stored for the split ('–' = none).")
    print("        decoding: per-model = settings tuned on this model's dev run (results/tuned_settings.json);")
    print("                  default   = the backend defaults, which were tuned on nimble's dev run.")
    if missing:
        print("create missing predictions with: make dev BACKEND=<backend> MODEL=<model>   (then make tune, make eval)")
    print("re-run a complete run from scratch with OVERWRITE=1, e.g. make eval BACKEND=ollama MODEL=tev1 OVERWRITE=1")


if __name__ == "__main__":
    main()
