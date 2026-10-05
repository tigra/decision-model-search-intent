#!/bin/zsh
# Regenerate all result files from the saved predictions (no model calls; CPU only, a few minutes):
#   results/results_{eval,dev}.{md,json}   main tables (metrics with counts and 95% intervals, McNemar tests)
#   results/mcnemar_{eval,dev}.txt         readable pairwise significance tests
#   results/report_eval_<run>.md           detailed per-run reports (per-attribute filters, confusions, worst queries)
#   docs/figures/*.svg                     the README figures (src/sidm/figures.py)
# The runs are the presets in src/sidm/runs.py (EVAL_RUNS, DEV_RUNS); edit them there.
# Runs without predictions are skipped with a warning; pass --strict (make results STRICT=1) to fail instead.
cd "$(dirname "$0")/.."
for split in eval dev; do
  uv run python -m sidm.results_table --preset $split --split $split --md results/results_$split.md "$@" | tail -1
  uv run python -m sidm.results_table --show-tests results/results_$split.json > results/mcnemar_$split.txt
  echo "written to results/mcnemar_$split.txt"
done
uv run python -m sidm.runs --reports eval
uv run --group figures python -m sidm.figures
