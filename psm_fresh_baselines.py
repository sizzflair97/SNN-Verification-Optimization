#!/usr/bin/env python3
"""
Re-run baselines (bnb_legacy, bnb_nofilter) at target cells with CURRENT code.

Necessary because adv_rob_mnist_module.py has uncommitted changes that post-date
the existing baseline files — apples-to-apples comparison requires fresh runs.

Covers:
  (nh=200, δ=3) — bnb_legacy, bnb_nofilter
  (nh=500, δ=2) — bnb_legacy      (bnb_nofilter at nh=500 δ=2 already queued
                                   in psm_scale_bench.py, so not redone here)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import psm_scale_bench as psb

psb.JOBS = [
    (200, 3, "bnb_legacy",   ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0"}, "BnB (legacy)  nh=200 δ=3 [fresh]"),
    (200, 3, "bnb_nofilter", ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "0"}, "BnB (nofilter) nh=200 δ=3 [fresh]"),
    (500, 2, "bnb_legacy",   ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0"}, "BnB (legacy)  nh=500 δ=2 [fresh]"),
]

psb.main()
