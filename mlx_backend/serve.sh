#!/bin/zsh
# Start the MLX ParallelScorer server. Unloads Ollama's nimble first: both together don't fit in memory.
#   mlx_backend/serve.sh [--config <nimble-model.json>] [--port 11500]
cd "$(dirname "$0")/.."
ollama stop nimble 2>/dev/null
config_default="${SIDM_MODELS_DIR:-$HOME/sidm-models}/bespoke-nimble-9b-q8/nimble-model.json"
if [[ "$*" != *--config* ]]; then set -- --config "$config_default" "$@"; fi
# caffeinate -i: no idle sleep while serving (long eval runs otherwise stall when the display turns off)
exec caffeinate -i uv run --project mlx_backend python -u mlx_backend/server.py "$@"
