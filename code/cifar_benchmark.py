#!/usr/bin/env python3
"""CIFAR-10 benchmark: baseline vs BC-IBP at T=5, n_h=512, delta=1,2"""
import subprocess, os, time, re
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
LOG_DIR = WORKDIR / "log"

BASE_ENV = {**os.environ, "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_EFFICIENT": "0", "SNN_BNB_PSM": "0"}
CONFIGS = [
    ("cifar_base_d1", {"SNN_BNB_IBP": "0"}, 1, "baseline δ=1"),
    ("cifar_ibp_d1",  {"SNN_BNB_IBP": "1", "SNN_BNB_IBP_COUPLED": "1", "SNN_BNB_IBP_EVERY": "100"}, 1, "BC-IBP δ=1"),
    ("cifar_base_d2", {"SNN_BNB_IBP": "0"}, 2, "baseline δ=2"),
    ("cifar_ibp_d2",  {"SNN_BNB_IBP": "1", "SNN_BNB_IBP_COUPLED": "1", "SNN_BNB_IBP_EVERY": "100"}, 2, "BC-IBP δ=2"),
]

def parse_log(prefix):
    results = []
    for f in sorted(LOG_DIR.glob(f"*{prefix}*")):
        text = f.read_text(errors="ignore")
        times = re.findall(r"Checking done in time ([\d.]+)", text)
        verdicts = re.findall(r"(Not robust|Robust) for sample", text)
        for t, v in zip(times, verdicts):
            results.append((float(t), "NR" if "Not" in v else "R"))
    return results

for prefix, extra_env, delta, label in CONFIGS:
    env = {**BASE_ENV, **extra_env}
    cmd = ["python", "batch_test.py", "-p", prefix, "--test-type", "cifar",
           "--num-samples", "10", "--n-hidden-neurons", "512", "--num-steps", "5",
           "--delta-max", str(delta), "--seed", "42", "--np"]
    print(f"\nRunning: {label} ...", flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(WORKDIR), timeout=3100)
    wall = time.time() - t0
    results = parse_log(prefix)
    solver_sum = sum(t for t, _ in results)
    n_r = sum(1 for _, v in results if v == "R")
    n_nr = sum(1 for _, v in results if v == "NR")
    print(f"  {label}: wall={wall:.1f}s solver_sum={solver_sum:.2f}s R={n_r} NR={n_nr}")
    for i, (t, v) in enumerate(results):
        print(f"    sample {i}: {t:.3f}s {v}")

print("\nDone.")
