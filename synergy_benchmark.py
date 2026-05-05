#!/usr/bin/env python3
"""
Synergy benchmark: PSM only / BC-IBP only / PSM+BC-IBP on T=64, n_h=256, delta=2.
Runs 4 BnB configurations on the SAME 10 MNIST samples (seed=42) and compares.
"""

import subprocess
import re
import os
import sys
import time
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
LOG_DIR = WORKDIR / "log"
BENCH_DIR = WORKDIR / "bench_results"
BENCH_DIR.mkdir(exist_ok=True)

# Common parameters
NUM_SAMPLES = 10
N_HIDDEN = 256
NUM_STEPS = 64
DELTA = 2
SEED = 42
TIMEOUT = 3100  # per-config timeout in seconds

# Base environment: voltage-margin filter always on, efficient off
BASE_ENV = {
    **os.environ,
    "SNN_BNB_LEGACY_ACTIVE_SET": "1",
    "SNN_BNB_EFFICIENT": "0",
    "SNN_BNB_PSM": "0",
    "SNN_BNB_IBP": "0",
    "SNN_BNB_IBP_COUPLED": "0",
    "SNN_BNB_IBP_EVERY": "100",
}

# 4 configurations
CONFIGS = [
    {
        "name": "baseline",
        "prefix": "synergy_baseline",
        "psm_flag": False,
        "env_overrides": {},
    },
    {
        "name": "PSM_only",
        "prefix": "synergy_psm",
        "psm_flag": True,
        "env_overrides": {"SNN_BNB_PSM": "1"},
    },
    {
        "name": "BC-IBP_only",
        "prefix": "synergy_ibp",
        "psm_flag": False,
        "env_overrides": {
            "SNN_BNB_IBP": "1",
            "SNN_BNB_IBP_COUPLED": "1",
            "SNN_BNB_IBP_EVERY": "100",
        },
    },
    {
        "name": "PSM+BC-IBP",
        "prefix": "synergy_both",
        "psm_flag": True,
        "env_overrides": {
            "SNN_BNB_PSM": "1",
            "SNN_BNB_IBP": "1",
            "SNN_BNB_IBP_COUPLED": "1",
            "SNN_BNB_IBP_EVERY": "100",
        },
    },
]


def build_cmd(cfg):
    cmd = [
        sys.executable, "batch_test.py",
        "-p", cfg["prefix"],
        "--test-type", "mnist",
        "--num-samples", str(NUM_SAMPLES),
        "--n-hidden-neurons", str(N_HIDDEN),
        "--num-steps", str(NUM_STEPS),
        "--delta-max", str(DELTA),
        "--seed", str(SEED),
        "--np",
    ]
    if cfg["psm_flag"]:
        cmd.append("--psm")
    return cmd


def find_log_file(prefix):
    """Find the most recently created log file matching the prefix."""
    pattern = f"*{prefix}*"
    candidates = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_log(log_path):
    """Parse a log file and extract per-sample times and verdicts."""
    results = []
    if log_path is None or not log_path.exists():
        return results

    text = log_path.read_text()
    # Find all "Checking done in time X" lines
    time_matches = re.findall(r"Checking done in time ([\d.]+)", text)
    # Find all verdict lines
    verdict_matches = re.findall(
        r"(Robust|Not robust) for sample (\d+) and delta=(\d+)", text
    )

    for i, (verdict_str, sample_no, delta_val) in enumerate(verdict_matches):
        t = float(time_matches[i]) if i < len(time_matches) else float("nan")
        robust = verdict_str == "Robust"
        results.append({
            "sample": int(sample_no),
            "delta": int(delta_val),
            "robust": robust,
            "time": t,
        })
    return results


def run_config(cfg):
    """Run a single configuration and return (log_path, results)."""
    env = {**BASE_ENV, **cfg["env_overrides"]}
    cmd = build_cmd(cfg)

    print(f"\n{'='*70}")
    print(f"Running config: {cfg['name']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Env overrides: {cfg['env_overrides']}")
    print(f"{'='*70}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            env=env,
            timeout=TIMEOUT,
            capture_output=False,  # let stdout/stderr flow through
        )
        elapsed_wall = time.time() - t0
        print(f"\n  Config '{cfg['name']}' finished in {elapsed_wall:.1f}s (exit code {result.returncode})")
    except subprocess.TimeoutExpired:
        elapsed_wall = time.time() - t0
        print(f"\n  Config '{cfg['name']}' TIMED OUT after {elapsed_wall:.1f}s")

    sys.stdout.flush()

    # Find the generated log file
    log_path = find_log_file(cfg["prefix"])
    if log_path:
        print(f"  Log file: {log_path}")
    else:
        print(f"  WARNING: No log file found for prefix '{cfg['prefix']}'")

    results = parse_log(log_path) if log_path else []
    return log_path, results, elapsed_wall


def print_summary(all_results):
    """Print a formatted summary table."""
    print(f"\n\n{'='*90}")
    print(f"SYNERGY BENCHMARK SUMMARY — T={NUM_STEPS}, n_h={N_HIDDEN}, delta={DELTA}, samples={NUM_SAMPLES}")
    print(f"{'='*90}")

    # Header
    config_names = [c["name"] for c in CONFIGS]
    header = f"{'Sample':>8} | "
    for name in config_names:
        header += f"  {name:>16}  |"
    print(header)
    print("-" * len(header))

    # Per-sample rows
    for sample_idx in range(NUM_SAMPLES):
        row = f"{sample_idx:>8} | "
        for cfg_name in config_names:
            results = all_results.get(cfg_name, [])
            if sample_idx < len(results):
                r = results[sample_idx]
                verdict = "R" if r["robust"] else "NR"
                row += f"  {r['time']:>10.2f}s {verdict:>3}  |"
            else:
                row += f"  {'N/A':>14}  |"
        print(row)

    print("-" * len(header))

    # Summary row: totals
    row = f"{'TOTAL':>8} | "
    for cfg_name in config_names:
        results = all_results.get(cfg_name, [])
        if results:
            total_t = sum(r["time"] for r in results)
            n_robust = sum(1 for r in results if r["robust"])
            row += f"  {total_t:>9.1f}s {n_robust}R/{len(results)-n_robust}NR |"
        else:
            row += f"  {'N/A':>14}  |"
    print(row)

    # Speedup relative to baseline
    baseline_results = all_results.get("baseline", [])
    if baseline_results:
        baseline_total = sum(r["time"] for r in baseline_results)
        row = f"{'Speedup':>8} | {'1.00x':>16}  |"
        for cfg_name in config_names[1:]:
            results = all_results.get(cfg_name, [])
            if results:
                cfg_total = sum(r["time"] for r in results)
                speedup = baseline_total / cfg_total if cfg_total > 0 else float("inf")
                row += f"  {speedup:>14.2f}x  |"
            else:
                row += f"  {'N/A':>14}  |"
        print(row)

    print(f"{'='*90}")

    # Verdict consistency check
    print("\nVerdict consistency check:")
    for sample_idx in range(NUM_SAMPLES):
        verdicts = {}
        for cfg_name in config_names:
            results = all_results.get(cfg_name, [])
            if sample_idx < len(results):
                verdicts[cfg_name] = results[sample_idx]["robust"]
        unique_verdicts = set(verdicts.values())
        if len(unique_verdicts) > 1:
            print(f"  WARNING: Sample {sample_idx} has INCONSISTENT verdicts: {verdicts}")
        elif len(unique_verdicts) == 1:
            v = "Robust" if list(unique_verdicts)[0] else "Not Robust"
            print(f"  Sample {sample_idx}: {v} (consistent)")
        else:
            print(f"  Sample {sample_idx}: No data")


def main():
    print(f"Synergy Benchmark: T={NUM_STEPS}, n_h={N_HIDDEN}, delta={DELTA}")
    print(f"Model: models/{NUM_STEPS}_784_{N_HIDDEN}_10/")
    print(f"Samples: {NUM_SAMPLES}, Seed: {SEED}")
    print(f"Per-config timeout: {TIMEOUT}s")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    all_wall_times = {}

    for cfg in CONFIGS:
        log_path, results, wall_time = run_config(cfg)
        all_results[cfg["name"]] = results
        all_wall_times[cfg["name"]] = wall_time

    print_summary(all_results)

    print(f"\nWall-clock times per config:")
    for cfg in CONFIGS:
        print(f"  {cfg['name']:>16}: {all_wall_times[cfg['name']]:>10.1f}s")

    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
