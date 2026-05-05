#!/usr/bin/env python3
"""Ablation at MNIST n_h=200, delta=2: adds BC-IBP coupled variant to existing data."""
import os, re, subprocess, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"
OUT = ROOT / "bench_results" / f"ablation_nh200_d2_{time.strftime('%m%d%H%M')}.json"

BASE_ENV = {**os.environ}

CONFIGS = [
    ("bnb_legacy_ibp_coupled", {
        "SNN_BNB_EFFICIENT": "0",
        "SNN_BNB_LEGACY_ACTIVE_SET": "1",
        "SNN_BNB_PSM": "0",
        "SNN_BNB_IBP": "1",
        "SNN_BNB_IBP_COUPLED": "1",
        "SNN_BNB_IBP_EVERY": "100",
    }),
]

VERDICT_PATS = [
    (re.compile(r"Not robust for sample\s+(\d+)"), "not_robust"),
    (re.compile(r"\bRobust for sample\s+(\d+)"), "robust"),
]
TIME_PAT = re.compile(r"Checking done in time (\d+\.?\d*)")

def run(tag, env_over):
    env = {**BASE_ENV, **env_over}
    log_prefix = f"abl_{tag}_nh200_d2"
    cmd = [
        "/home/uoscisai/venv312/bin/python", "batch_test.py",
        "--test-type", "mnist", "--num-samples", "10", "--num-steps", "5",
        "--delta-max", "2", "--seed", "42", "--np",
        "-p", log_prefix, "--n-hidden-neurons", "200",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=4000)
    wall = time.time() - t0
    # Find log
    logs = sorted(LOG_DIR.glob(f"*{log_prefix}*"))
    text = logs[-1].read_text(errors="ignore") if logs else (r.stdout + r.stderr)
    # Parse per-sample
    per_sample = []
    for line in text.splitlines():
        for pat, verd in VERDICT_PATS:
            m = pat.search(line)
            if m:
                per_sample.append({"sample_no": int(m.group(1)), "verdict": verd})
    times = [float(m.group(1)) for m in TIME_PAT.finditer(text)]
    return {"tag": tag, "wall_s": wall, "solver_times": times, "verdicts": per_sample,
            "log": str(logs[-1]) if logs else None}

results = []
for tag, env in CONFIGS:
    print(f"Running {tag}...", flush=True)
    res = run(tag, env)
    print(f"  wall={res['wall_s']:.1f}s, #solver_times={len(res['solver_times'])}, verdicts={res['verdicts']}")
    results.append(res)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"config": {"n_h": 200, "delta": 2, "T": 5, "num_samples": 10}, "results": results}, indent=2))
print(f"\nResults → {OUT}")
