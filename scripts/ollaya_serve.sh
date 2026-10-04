#!/bin/zsh
# Start Ollaya's server for the `ollaya` backend, with models kept under $SIDM_MODELS_DIR.
# Unloads Ollama's models first: run only one model server on the GPU at a time.
#   scripts/ollaya_serve.sh        (stop with: ollaya stop, or kill the process)
root="${SIDM_MODELS_DIR:-$HOME/sidm-models}"
export OLLAYA_MODELS="$root/ollaya-models" OLLAYA_HOST="127.0.0.1:11435"
for m in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do ollama stop "$m"; done
exec caffeinate -i "$root/ollaya/bin/ollaya" serve
