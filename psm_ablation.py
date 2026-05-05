#!/usr/bin/env python3
"""
PSM isolation benchmark — evaluate PSM's independent contribution.

Runs two new method variants at n_h=200, delta=2, same 10 samples as
delta_scaling_0416213348.json (seed=42), so results compose with existing
bnb_legacy / bnb_nofilter / exhaustive baselines.

Configs:
  - bnb_legacy_psm:   EFFICIENT=0 LEGACY=1 PSM=1   (PSM on top of sound legacy filter)
  - bnb_nofilter_psm: EFFICIENT=0 LEGACY=0 PSM=1   (PSM on top of no filter)
"""
import sys, os, time, json
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import realistic_large_benchmark as rlb

rlb.N_HIDDEN_LIST = [200]
rlb.DELTA = 2
rlb.NUM_SAMPLES = 10

rlb.METHODS = [
    ("bnb_legacy_psm",   ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "1"}, "BnB (legacy+PSM)"),
    ("bnb_nofilter_psm", ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "1"}, "BnB (nofilter+PSM)"),
]

out_path, results = rlb.run_full_matrix()
print(f"\nPSM ablation results -> {out_path}")
