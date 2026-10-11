"""Low-memory launcher for kev.serve. Kev source is not modified; three settings are patched at start:

1. One question row per forward pass: rows_per_pass budget = 1 (default 16,384 tokens). Kev documents that rows are
   independent and the answers do not depend on the split (kev/model.py rows_per_pass docstring).
2. MLX buffer cache limit 256 MB (default 1 GB, kev/mlx_model.py CACHE_LIMIT).
3. State-prefix cache off (KEV_PREFIX_CACHE=0): every query has its own state, so it never hits (smoke: 0 of 8).

Used by kev_backend/serve.sh when LOWMEM=1 (machines that swap with the 9B model loaded, e.g. 36 GB).
"""
import os
import runpy
import sys

os.environ.setdefault("KEV_PREFIX_CACHE", "0")

import kev.model  # noqa: E402
import kev.mlx_model  # noqa: E402

kev.model.rows_per_pass.__defaults__ = (0, int(os.environ.get("KEV_ROW_PASS_BUDGET", "1")))
kev.mlx_model.CACHE_LIMIT = int(os.environ.get("KEV_MLX_CACHE_LIMIT", str(256 << 20)))

sys.argv = ["kev.serve"] + sys.argv[1:]
runpy.run_module("kev.serve", run_name="__main__", alter_sys=True)
