#!/bin/zsh
# Serve AWS Strands Labs' Strands Decider on :11600 (its own /v1/systemone server) for the `decider` backend.
# Unloads Ollama's models and stops Ollaya first: run only one model on the GPU at a time.
# Weights (LoRA + head + Qwen3.5-2B-Base, ~4 GB) download on the first start into $SIDM_MODELS_DIR/hf-home.
#   decider_backend/serve.sh                 (DEVICE=mps to use torch/MPS instead of MLX)
cd "$(dirname "$0")/.."
root="${SIDM_MODELS_DIR:-$HOME/sidm-models}"
for m in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do ollama stop "$m"; done
OLLAYA_MODELS="$root/ollaya-models" OLLAYA_HOST=127.0.0.1:11435 "$root/ollaya/bin/ollaya" stop 2>/dev/null
export HF_HOME="$root/hf-home" HF_XET_CHUNK_CACHE_SIZE_BYTES=0
# --strict-window: refuse (HTTP 422) prompts longer than the window instead of silently cutting the query
exec caffeinate -i uv run --project decider_backend strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 \
  --host 127.0.0.1 --port 11600 --device "${DEVICE:-mlx}" --strict-window --model-name strands-decider-2b "$@"
