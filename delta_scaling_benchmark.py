#!/usr/bin/env python3
"""
Delta-scaling benchmark: fix n_h=200, vary delta ∈ {1,2,3,4}.
Uses the same per-sample 300s subprocess-timeout harness as realistic_large_benchmark.py.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import realistic_large_benchmark as rlb

# Override configuration for delta scaling
rlb.N_HIDDEN_LIST = [200]
rlb.DELTA_LIST = [1, 2, 3, 4]  # marker; used below, not inside rlb directly

# We need to call rlb.run_full_matrix once per delta.
# Patch rlb.DELTA before each run and merge results.
import json, time
from collections import defaultdict

RESULTS_DIR = ROOT / "bench_results"
RESULTS_DIR.mkdir(exist_ok=True)

all_results = []
run_id = time.strftime("%m%d%H%M%S")
print(f"Delta-scaling run_id = {run_id}")

# Exclude Z3 (known to timeout at n_h=100 already)
rlb.METHODS = [m for m in rlb.METHODS if m[0] != "z3"]

for d in rlb.DELTA_LIST:
    print(f"\n{'=' * 60}")
    print(f"=== DELTA = {d} ===")
    print(f"{'=' * 60}")
    rlb.DELTA = d
    # Fresh run_id internally used by rlb; but we'll override the result file.
    out_json, results = rlb.run_full_matrix()
    for r in results:
        r["delta"] = d
    all_results.extend(results)

merged = {
    "config": {
        "num_steps": rlb.NUM_STEPS,
        "delta_list": rlb.DELTA_LIST,
        "samples": rlb.NUM_SAMPLES,
        "per_sample_timeout_s": rlb.PER_SAMPLE_TIMEOUT_S,
        "n_hidden": rlb.N_HIDDEN_LIST[0],
        "methods": [{"key": k, "label": lbl} for k, _, _, lbl in rlb.METHODS],
        "run_id": run_id,
        "seed": rlb.SEED,
    },
    "results": all_results,
}
out_path = RESULTS_DIR / f"delta_scaling_{run_id}.json"
out_path.write_text(json.dumps(merged, indent=2))
print(f"\nMerged delta-scaling results → {out_path}")
