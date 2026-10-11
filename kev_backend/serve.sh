#!/bin/zsh
# Serve Kev (jaredpalmer/kev) on :11700 (its own TypeSafe-compatible /v1/systemone server, MLX) for the `kev` backend.
# Unloads Ollama's models and stops Ollaya first: run only one model on the GPU at a time.
# Weights (LoRA + pointer head + Qwen3.5-9B-Base, bf16, ~18 GB) download on the first start into $SIDM_MODELS_DIR/hf-home.
#   kev_backend/serve.sh                     (KEV_RUN=jaredpalmer/kev-4b for the 4B checkpoint)
#   LOWMEM=1 kev_backend/serve.sh            (one question row per forward pass, 256 MB MLX cache, no prefix cache;
#                                             on an M3 Pro 36 GB it cut the median from 47 s to 12 s per query)
cd "$(dirname "$0")/.."
root="${SIDM_MODELS_DIR:-$HOME/sidm-models}"
for m in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do ollama stop "$m"; done
OLLAYA_MODELS="$root/ollaya-models" OLLAYA_HOST=127.0.0.1:11435 "$root/ollaya/bin/ollaya" stop 2>/dev/null
export HF_HOME="$root/hf-home" HF_XET_CHUNK_CACHE_SIZE_BYTES=0
args=(--run "${KEV_RUN:-jaredpalmer/kev-9b}" --host 127.0.0.1 --port 11700 "$@")
if [[ -n "$LOWMEM" ]]; then
  exec caffeinate -i uv run --project kev_backend python kev_backend/serve_lowmem.py "${args[@]}"
fi
exec caffeinate -i uv run --project kev_backend python -m kev.serve "${args[@]}"
