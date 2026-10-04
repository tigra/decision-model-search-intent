"""Turn a Hugging Face PEFT LoRA adapter (or a merged checkpoint) into an MLX model for nimble's ParallelScorer.

  uv run --project mlx_backend python mlx_backend/convert_model.py all                    # Bespoke-Nimble-9B, 8-bit
  uv run --project mlx_backend python mlx_backend/convert_model.py all --q-bits none      # keep bf16
  uv run --project mlx_backend python mlx_backend/convert_model.py all --repo bespokelabs/Bespoke-Nimble-9B-v2
  uv run --project mlx_backend python mlx_backend/convert_model.py quantize --q-bits 4 --keep-intermediate

Stages (run `all`, or one at a time): download -> merge -> quantize -> verify -> config.
Each stage records itself in <models-dir>/<name>/state.json and is skipped when already done, so an
interrupted run resumes. Only files this script downloaded or produced under <models-dir> are deleted.

ParallelScorer computes candidate logits from the output head, which must stay unquantized:
quantization therefore skips `lm_head` (and `embed_tokens` if the weights are tied).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

NIMBLE_GIT = "https://github.com/bespokelabsai/nimble.git"
NIMBLE_COMMIT = "62076b4f2d365b5879dafcf7f6dd072a1fe76df7"
# Packages for the one-off merge (nimble's requirements/training.txt); kept out of the server env.
MERGE_PACKAGES = ["torch==2.8.0", "peft==0.21.0", "accelerate==1.15.0", "sentencepiece==0.2.2"]
STAGES = ["download", "merge", "quantize", "verify", "config"]
GB = 1e9


# ---------------------------------------------------------------- paths and state
def models_dir(args):
    return Path(args.models_dir or os.environ.get("SIDM_MODELS_DIR") or Path.home() / "sidm-models").expanduser()


def setup_env(root):
    """Keep every Hugging Face byte under the models dir; no xet chunk cache on the system disk."""
    os.environ["HF_HOME"] = str(root / "hf-home")
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")


def nimble_repo(root):
    repo = root / "nimble-repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", NIMBLE_GIT, str(repo)], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != NIMBLE_COMMIT:
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", NIMBLE_COMMIT], check=True)
    return repo


class State:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def done(self, stage):
        return stage in self.data

    def save(self, stage, **info):
        self.data[stage] = info
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / GB


def dir_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and not f.is_symlink()) / GB


def remove_owned(path, root, what):
    path = Path(path).resolve()
    if root.resolve() not in path.parents:
        raise RuntimeError("refusing to delete %s outside %s" % (path, root))
    if path.exists():
        size = dir_gb(path)
        shutil.rmtree(path)
        print("deleted %s (%.1f GB): %s" % (what, size, path))


def remove_cached_repo(hub, repo, root):
    """Delete one repo from our Hugging Face cache, including content-store blobs only it references.

    Recent huggingface_hub keeps file contents in a shared <hub>/blobs/<xx>/<sha256> store with a `.refs`
    file listing the repos using each blob; deleting models--<repo>/ alone would leave those bytes behind.
    """
    repo_dir = "models--" + repo.replace("/", "--")
    remove_owned(hub / repo_dir, root, "cached %s" % repo)
    store, freed = hub / "blobs", 0
    for refs in store.glob("*/*.refs") if store.is_dir() else []:
        users = [l for l in refs.read_text().splitlines() if l.strip()]
        if users and all(u.startswith(repo_dir + "/") for u in users):
            blob = refs.with_suffix("")
            freed += blob.stat().st_size if blob.exists() else 0
            for f in (blob, refs, blob.with_name(blob.name + ".lock")):
                f.unlink(missing_ok=True)
    if freed:
        print("deleted %.1f GB of %s content-store blobs" % (freed / GB, repo))


# ---------------------------------------------------------------- stages
def stage_download(args, root, out, state):
    from huggingface_hub import HfApi, snapshot_download
    hub = root / "hf"
    info = HfApi().model_info(args.repo, revision=args.revision or "main")
    revision = info.sha
    adapter = Path(snapshot_download(args.repo, revision=revision, cache_dir=hub))
    contract_file = adapter / "schema_config.json"
    contract = json.loads(contract_file.read_text()) if contract_file.exists() else {}
    is_adapter = (adapter / "adapter_config.json").exists()
    base, base_revision = args.base or contract.get("model"), args.base_revision or contract.get("revision")
    base_path = None
    if is_adapter:
        if not base:
            raise SystemExit("adapter has no schema_config.json model; pass --base")
        size = sum(s.size or 0 for s in HfApi().model_info(base, revision=base_revision, files_metadata=True).siblings
                   if s.rfilename.endswith(".safetensors")) / GB
        quant = size / 2 if args.q_bits == 8 else size / 4 if args.q_bits == 4 else 0
        # base + merge coexist; the base is deleted before quantizing, so merge + quantized come after
        need = (size * 3 + quant if args.keep_intermediate else max(size * 2, size + quant)) + 2
        have = free_gb(root)
        print("base %s: %.1f GB; peak need ~%.0f GB, free %.0f GB" % (base, size, need, have))
        if need > have and not args.force:
            raise SystemExit("not enough free disk under %s (need ~%.0f GB, have %.0f GB); "
                             "free space, use --models-dir on another volume, or --force" % (root, need, have))
        base_path = Path(snapshot_download(base, revision=base_revision, cache_dir=hub,
                                           allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]))
    state.save("download", repo=args.repo, revision=revision, adapter_path=str(adapter), is_adapter=is_adapter,
               base=base, base_revision=base_revision, base_path=str(base_path) if base_path else None)


def stage_merge(args, root, out, state):
    d = state.data["download"]
    if not d["is_adapter"]:
        state.save("merge", merged_path=d["adapter_path"], skipped="repo is already a merged checkpoint")
        return
    merged = out / "merged-bf16"
    repo = nimble_repo(root)
    cmd = ["uv", "run", "--project", str(Path(__file__).parent)] + sum((["--with", p] for p in MERGE_PACKAGES), []) + [
        "python", str(repo / "nimble/scoring/merge_local_adapter.py"),
        "--adapter", d["adapter_path"], "--base", d["base_path"], "--output", str(merged)]
    print("merging (CPU, bf16; needs ~2x the base size in RAM):", " ".join(cmd))
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH=str(repo)))
    state.save("merge", merged_path=str(merged))


def stage_quantize(args, root, out, state):
    merged = Path(state.data["merge"]["merged_path"])
    d = state.data["download"]
    if args.q_bits == "none":
        target = merged
    else:
        if d["is_adapter"] and d["base_path"] and not args.keep_intermediate:
            remove_cached_repo(root / "hf", d["base"], root)
        target = out / ("mlx-q%d" % args.q_bits)
        if target.exists():
            remove_owned(target, root, "incomplete quantized output")
        from mlx_lm import convert
        cfg = json.loads((merged / "config.json").read_text())
        tied = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings", False))
        keep = ("lm_head",) + (("embed_tokens",) if tied else ())

        def predicate(path, module, config=None):
            return not any(k in path for k in keep)

        convert(hf_path=str(merged), mlx_path=str(target), quantize=True, q_bits=args.q_bits,
                q_group_size=args.q_group_size, quant_predicate=predicate)
        for name in ("schema_config.json", "READY.json"):  # prompt contract + adapter identity for the temperature
            if (merged / name).exists():
                shutil.copyfile(merged / name, target / name)
    state.save("quantize", model_path=str(target), q_bits=args.q_bits, size_gb=round(dir_gb(target), 2),
               unquantized=None if args.q_bits == "none" else ["lm_head"])


def stage_verify(args, root, out, state):
    repo = nimble_repo(root)
    sys.path.insert(0, str(repo))
    from nimble.scoring.parallel_scorer import ParallelScorer
    q, d = state.data["quantize"], state.data["download"]
    scorer = ParallelScorer(model_path=q["model_path"], max_input_tokens=args.max_input_tokens,
                            model_id=d["repo"], revision=d["revision"])
    schema = json.loads((repo / "examples/parallel_schema.json").read_text())
    result = scorer.score("The payment service is down for all customers since 9am.", schema)
    print("verify output:", result["output"])
    print("verify metrics:", {k: result["metrics"][k] for k in ("prefix_tokens", "prefill_seconds",
                                                                "field_evaluation_seconds", "mlx_peak_active_gib")})
    merged = Path(state.data["merge"]["merged_path"])
    if q["q_bits"] != "none" and d["is_adapter"] and not args.keep_intermediate:
        remove_owned(merged, root, "bf16 merge")
    state.save("verify", output=result["output"], temperature=result["temperature"],
               temperature_fitted=result["temperature_fitted"], metrics=result["metrics"])


def stage_config(args, root, out, state):
    q, d = state.data["quantize"], state.data["download"]
    config = {"model_path": q["model_path"], "model_id": d["repo"], "revision": d["revision"],
              "max_input_tokens": args.max_input_tokens, "q_bits": q["q_bits"], "nimble_repo": str(nimble_repo(root))}
    path = out / "nimble-model.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    state.save("config", path=str(path))
    print("ready: %s  (%.1f GB)\nstart the server with: mlx_backend/serve.sh --config %s" % (q["model_path"], q["size_gb"], path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all"] + STAGES)
    ap.add_argument("--repo", default="bespokelabs/Bespoke-Nimble-9B")
    ap.add_argument("--revision", default=None, help="hub revision (default: latest main, resolved to the full hash)")
    ap.add_argument("--base", default=None, help="base model repo (default: from the adapter's schema_config.json)")
    ap.add_argument("--base-revision", default=None)
    ap.add_argument("--q-bits", default="8", choices=["4", "8", "none"])
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--models-dir", default=None, help="default: $SIDM_MODELS_DIR or ~/sidm-models")
    ap.add_argument("--name", default=None, help="output folder name (default: from repo and bits)")
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--keep-intermediate", action="store_true", help="keep the base snapshot and the bf16 merge")
    ap.add_argument("--redo", action="store_true", help="re-run the selected stage even if recorded as done")
    ap.add_argument("--force", action="store_true", help="skip the free-disk check")
    args = ap.parse_args()
    args.q_bits = args.q_bits if args.q_bits == "none" else int(args.q_bits)

    root = models_dir(args)
    setup_env(root)
    name = args.name or "%s-%s" % (args.repo.split("/")[-1].lower(), "bf16" if args.q_bits == "none" else "q%d" % args.q_bits)
    out = root / name
    state = State(out / "state.json")
    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        if state.done(stage) and not (args.redo and args.stage != "all"):
            print("[%s] already done, skipping" % stage)
            continue
        missing = [s for s in STAGES[:STAGES.index(stage)] if not state.done(s)]
        if missing:
            raise SystemExit("[%s] needs earlier stages first: %s" % (stage, missing))
        print("[%s] ..." % stage, flush=True)
        globals()["stage_" + stage](args, root, out, state)


if __name__ == "__main__":
    main()
