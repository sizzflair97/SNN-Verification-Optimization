#!/usr/bin/env python3
"""FashionMNIST benchmark: baseline vs BC-IBP for T=256, n_h=512 and n_h=1024."""

import subprocess
import time
import os
import re
import glob
import sys

WORKDIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKDIR, "log")

CONFIGS = [
    {
        "label": "fmnist_base_512",
        "desc": "T=256, n_h=512, delta=2, baseline (voltage margin only)",
        "env": {
            "SNN_BNB_EFFICIENT": "0",
            "SNN_BNB_LEGACY_ACTIVE_SET": "1",
            "SNN_BNB_IBP": "0",
        },
        "prefix": "fmnist_base_512",
        "n_hidden": 512,
    },
    {
        "label": "fmnist_ibp_512",
        "desc": "T=256, n_h=512, delta=2, BC-IBP coupled",
        "env": {
            "SNN_BNB_EFFICIENT": "0",
            "SNN_BNB_LEGACY_ACTIVE_SET": "1",
            "SNN_BNB_IBP": "1",
            "SNN_BNB_IBP_COUPLED": "1",
            "SNN_BNB_IBP_EVERY": "100",
        },
        "prefix": "fmnist_ibp_512",
        "n_hidden": 512,
    },
    {
        "label": "fmnist_base_1024",
        "desc": "T=256, n_h=1024, delta=2, baseline (voltage margin only)",
        "env": {
            "SNN_BNB_EFFICIENT": "0",
            "SNN_BNB_LEGACY_ACTIVE_SET": "1",
            "SNN_BNB_IBP": "0",
        },
        "prefix": "fmnist_base_1024",
        "n_hidden": 1024,
    },
    {
        "label": "fmnist_ibp_1024",
        "desc": "T=256, n_h=1024, delta=2, BC-IBP coupled",
        "env": {
            "SNN_BNB_EFFICIENT": "0",
            "SNN_BNB_LEGACY_ACTIVE_SET": "1",
            "SNN_BNB_IBP": "1",
            "SNN_BNB_IBP_COUPLED": "1",
            "SNN_BNB_IBP_EVERY": "100",
        },
        "prefix": "fmnist_ibp_1024",
        "n_hidden": 1024,
    },
]

COMMON_ARGS = [
    sys.executable, "batch_test.py",
    "--test-type", "fmnist",
    "--num-samples", "10",
    "--num-steps", "256",
    "--delta-max", "2",
    "--seed", "42",
    "--np",
]

TOTAL_TIMEOUT = 3100  # safety timeout per config


def find_log_file(prefix: str, n_hidden: int) -> str | None:
    """Find the most recent log file matching a config."""
    # Log name pattern: MMDDHHMM_fmnist_PREFIX_np_256_784_NH_10_delta(2,).log
    pattern = os.path.join(LOG_DIR, f"*_fmnist_{prefix}_np_256_784_{n_hidden}_10_delta(2,).log")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def parse_log(log_path: str) -> list[dict]:
    """Parse a log file for per-sample solver times and verdicts.

    Pairs each "Checking done in time T" with the next "(Not )Robust for
    sample X" line. Cannot rely on prior "sample X is drawn" lines because
    all samples are drawn upfront before any checking begins.
    """
    results = []
    with open(log_path, "r") as f:
        lines = f.readlines()

    pending_time = None
    for line in lines:
        m = re.search(r"Checking done in time ([\d.]+)", line)
        if m:
            pending_time = float(m.group(1))
            continue

        m = re.search(r"(Not robust|Robust) for sample (\d+)", line)
        if m:
            verdict = "NOT_ROBUST" if m.group(1) == "Not robust" else "ROBUST"
            sample_no = int(m.group(2))
            results.append({
                "sample": sample_no,
                "time": pending_time if pending_time is not None else 0.0,
                "verdict": verdict,
            })
            pending_time = None

    return results


def run_config(cfg: dict) -> dict:
    """Run a single benchmark config and return results."""
    label = cfg["label"]
    desc = cfg["desc"]
    prefix = cfg["prefix"]
    n_hidden = cfg["n_hidden"]

    print(f"\n{'='*70}")
    print(f"CONFIG: {label}")
    print(f"  {desc}")
    print(f"{'='*70}")
    sys.stdout.flush()

    env = os.environ.copy()
    env.update(cfg["env"])

    cmd = COMMON_ARGS + ["-p", prefix, "--n-hidden-neurons", str(n_hidden)]
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  ENV: {' '.join(f'{k}={v}' for k, v in cfg['env'].items())}")
    sys.stdout.flush()

    wall_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=WORKDIR,
            timeout=TOTAL_TIMEOUT,
            capture_output=True,
            text=True,
        )
        wall_time = time.time() - wall_start
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr

        if returncode != 0:
            print(f"  ERROR (returncode={returncode})")
            print(f"  STDERR: {stderr[:1000]}")
            print(f"  STDOUT: {stdout[:1000]}")
            sys.stdout.flush()
    except subprocess.TimeoutExpired:
        wall_time = time.time() - wall_start
        returncode = -1
        stdout = ""
        stderr = "TIMEOUT"
        print(f"  TIMEOUT after {wall_time:.1f}s")
        sys.stdout.flush()

    print(f"  Wall time: {wall_time:.1f}s  (returncode={returncode})")
    sys.stdout.flush()

    # Parse log
    log_path = find_log_file(prefix, n_hidden)
    samples = []
    if log_path:
        print(f"  Log file: {log_path}")
        samples = parse_log(log_path)
        for s in samples:
            print(f"    Sample {s['sample']:>6d}  |  {s['time']:>10.3f}s  |  {s['verdict']}")
    else:
        print(f"  WARNING: No log file found for prefix={prefix}, n_hidden={n_hidden}")
    sys.stdout.flush()

    return {
        "label": label,
        "desc": desc,
        "wall_time": wall_time,
        "returncode": returncode,
        "log_path": log_path,
        "samples": samples,
        "stdout": stdout,
        "stderr": stderr,
    }


def print_summary(all_results: list[dict]):
    """Print a summary table."""
    print(f"\n\n{'='*90}")
    print("FMNIST BENCHMARK SUMMARY")
    print(f"{'='*90}")

    # Config-level summary
    header = f"{'Config':<25s} {'Wall(s)':>10s} {'#Samples':>8s} {'#Robust':>8s} {'#NotRob':>8s} {'MeanT(s)':>10s} {'MedianT(s)':>10s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        samples = r["samples"]
        n = len(samples)
        n_robust = sum(1 for s in samples if s["verdict"] == "ROBUST")
        n_not_robust = sum(1 for s in samples if s["verdict"] == "NOT_ROBUST")
        times = [s["time"] for s in samples]
        mean_t = sum(times) / len(times) if times else 0
        median_t = sorted(times)[len(times) // 2] if times else 0
        print(f"{r['label']:<25s} {r['wall_time']:>10.1f} {n:>8d} {n_robust:>8d} {n_not_robust:>8d} {mean_t:>10.3f} {median_t:>10.3f}")

    # Per-sample comparison tables
    for nh in [512, 1024]:
        base = next((r for r in all_results if r["label"] == f"fmnist_base_{nh}"), None)
        ibp = next((r for r in all_results if r["label"] == f"fmnist_ibp_{nh}"), None)
        if not base or not ibp:
            continue

        print(f"\n\n--- n_h={nh}: Baseline vs BC-IBP (per-sample) ---")
        header2 = f"{'Sample':>8s} {'Base(s)':>12s} {'IBP(s)':>12s} {'Speedup':>10s} {'Base_V':>10s} {'IBP_V':>10s}"
        print(header2)
        print("-" * len(header2))

        base_map = {s["sample"]: s for s in base["samples"]}
        ibp_map = {s["sample"]: s for s in ibp["samples"]}

        all_samples = sorted(set(base_map.keys()) | set(ibp_map.keys()))
        for sno in all_samples:
            b = base_map.get(sno)
            i = ibp_map.get(sno)
            bt = f"{b['time']:.3f}" if b else "N/A"
            it = f"{i['time']:.3f}" if i else "N/A"
            bv = b["verdict"] if b else "N/A"
            iv = i["verdict"] if i else "N/A"
            if b and i and i["time"] > 0:
                speedup = f"{b['time'] / i['time']:.2f}x"
            else:
                speedup = "N/A"
            print(f"{sno:>8d} {bt:>12s} {it:>12s} {speedup:>10s} {bv:>10s} {iv:>10s}")

        # Aggregate speedup
        base_total = sum(s["time"] for s in base["samples"])
        ibp_total = sum(s["time"] for s in ibp["samples"])
        if ibp_total > 0:
            print(f"\n  Total: base={base_total:.1f}s, ibp={ibp_total:.1f}s, speedup={base_total/ibp_total:.2f}x")
        print(f"  Wall:  base={base['wall_time']:.1f}s, ibp={ibp['wall_time']:.1f}s")


def main():
    print(f"FashionMNIST Benchmark")
    print(f"Working directory: {WORKDIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Per-sample timeout: 300s (enforced by BnB solver)")
    print(f"Per-config safety timeout: {TOTAL_TIMEOUT}s")
    sys.stdout.flush()

    all_results = []
    for cfg in CONFIGS:
        result = run_config(cfg)
        all_results.append(result)

    print_summary(all_results)

    print(f"\nFinished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
