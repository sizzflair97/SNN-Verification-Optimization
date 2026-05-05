#!/usr/bin/env python3
"""
PSM scale-up benchmark
======================
Tests where PSM's prefix-set reuse should actually pay off:
  - higher delta (more pixel-subset combinatorics, more overlap)
  - higher n_hidden (more expensive per-node work)

Configurations:
  A) n_h=200, delta=3   (existing baselines: bnb_legacy, bnb_nofilter from delta_scaling_0416213348.json)
  B) n_h=500, delta=2   (existing baseline: bnb_legacy from nh500_delta2_ibp.json; need bnb_nofilter)

Methods run here (sound variants only):
  - bnb_legacy_psm     : legacy margin filter + PSM
  - bnb_nofilter_psm   : no filter + PSM
  - bnb_nofilter       : no filter baseline (only at nh=500 delta=2 — missing from prior runs)

Same 10 samples (seed=42) as prior benchmarks for pairing.
"""
import json, os, sys, time, re, subprocess
from pathlib import Path
from random import sample as random_sample, seed as random_seed

import numpy as np

ROOT = Path(__file__).parent.resolve()
LOG_DIR = ROOT / "log"
RESULTS_DIR = ROOT / "bench_results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
NUM_STEPS = 5
NUM_SAMPLES = 10
PER_SAMPLE_TIMEOUT_S = 300
SUBPROC_BUFFER_S = 30

# (n_hidden, delta, method_key, cli_args, env_overrides, label)
JOBS = [
    # Config A: n_h=200, delta=3
    (200, 3, "bnb_legacy_psm",   ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "1"}, "BnB (legacy+PSM) nh=200 δ=3"),
    (200, 3, "bnb_nofilter_psm", ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "1"}, "BnB (nofilter+PSM) nh=200 δ=3"),

    # Config B: n_h=500, delta=2
    (500, 2, "bnb_legacy_psm",   ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "1"}, "BnB (legacy+PSM) nh=500 δ=2"),
    (500, 2, "bnb_nofilter_psm", ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "1"}, "BnB (nofilter+PSM) nh=500 δ=2"),
    (500, 2, "bnb_nofilter",     ["--np"],          {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "0"}, "BnB (no filter) nh=500 δ=2"),
]


def select_samples(n_hidden, delta):
    """Reproduce the same 10-sample selection used by prior benchmarks (seed=42)."""
    sys.path.insert(0, str(ROOT))
    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights

    cfg = CFG(
        log_name=f"probe_{n_hidden}",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        deltas=(delta,),
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=NUM_STEPS,
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist(cfg)

    random_seed(SEED)
    np.random.seed(SEED)
    pool = random_sample([*range(len(images))], k=NUM_SAMPLES * 3)

    chosen = []
    for sample_no in pool:
        img = images[sample_no]
        ft = []
        forward(cfg, weights, img, ft)
        if len(np.argwhere(ft[-1] == np.min(ft[-1]))[0]) != 1:
            continue
        chosen.append(int(sample_no))
        if len(chosen) >= NUM_SAMPLES:
            break
    return chosen


VERDICT_PATTERNS = [
    (re.compile(r"Not robust for sample .* and delta"), "not_robust"),
    (re.compile(r"\bRobust for sample .* and delta"), "robust"),
]
TIME_PAT = re.compile(r"Checking done in time (\d+\.?\d*)")
PSM_PAT = re.compile(r"PSM cache:\s*(\d+)\s*checks,\s*(\d+)\s*hits")
STATE_PAT = re.compile(r"State cache:\s*(\d+)\s*checks,\s*(\d+)\s*hits")
FILTER_PAT = re.compile(r"Filtered pixels:\s*(\d+)\s*->\s*(\d+)")


def parse_log(log_path):
    """Extract verdict, solver time, PSM/state cache stats."""
    out = {"verdict": "unknown", "solver_time": None,
           "psm_checks": None, "psm_hits": None,
           "state_checks": None, "state_hits": None,
           "pixels_before": None, "pixels_after": None}
    try:
        text = Path(log_path).read_text(errors="ignore")
    except FileNotFoundError:
        return out
    for pat, v in VERDICT_PATTERNS:
        if pat.search(text):
            out["verdict"] = v
            break
    m = TIME_PAT.search(text)
    if m:
        out["solver_time"] = float(m.group(1))
    m = PSM_PAT.search(text)
    if m:
        out["psm_checks"], out["psm_hits"] = int(m.group(1)), int(m.group(2))
    m = STATE_PAT.search(text)
    if m:
        out["state_checks"], out["state_hits"] = int(m.group(1)), int(m.group(2))
    m = FILTER_PAT.search(text)
    if m:
        out["pixels_before"], out["pixels_after"] = int(m.group(1)), int(m.group(2))
    return out


def run_one(method_key, cli_args, env_overrides, n_hidden, delta, sample_no, run_id):
    prefix = f"psmscale_{run_id}_{method_key}_nh{n_hidden}_d{delta}_s{sample_no}"
    cmd = [
        sys.executable,
        str(ROOT / "batch_test.py"),
        "-p", prefix,
        "--test-type", "mnist",
        "--num-samples", "1",
        "--manual-indices", str(sample_no),
        "--n-hidden-neurons", str(n_hidden),
        "--num-steps", str(NUM_STEPS),
        "--delta-max", str(delta),
        "--seed", str(SEED),
        *cli_args,
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(ROOT),
            timeout=PER_SAMPLE_TIMEOUT_S + SUBPROC_BUFFER_S,
            capture_output=True, text=True,
        )
        rc = proc.returncode
        stderr_tail = (proc.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        rc = -1
        stderr_tail = "TIMEOUT"
        timed_out = True
    wall = time.time() - start

    log_candidates = sorted(LOG_DIR.glob(f"*_{prefix}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_info = {"verdict": "unknown", "solver_time": None}
    log_used = None
    if log_candidates:
        log_used = log_candidates[0].name
        log_info = parse_log(log_candidates[0])

    verdict = "timeout" if timed_out else log_info.get("verdict", "unknown")

    return {
        "method": method_key,
        "n_hidden": n_hidden,
        "delta": delta,
        "sample_no": sample_no,
        "wall_time": wall,
        "solver_time": log_info.get("solver_time"),
        "verdict": verdict,
        "timed_out": timed_out,
        "returncode": rc,
        "stderr_tail": stderr_tail,
        "log_file": log_used,
        "psm_checks": log_info.get("psm_checks"),
        "psm_hits": log_info.get("psm_hits"),
        "state_checks": log_info.get("state_checks"),
        "state_hits": log_info.get("state_hits"),
        "pixels_before": log_info.get("pixels_before"),
        "pixels_after": log_info.get("pixels_after"),
    }


def main():
    run_id = time.strftime("%m%d%H%M%S")
    print(f"[PSM-scale] run_id={run_id}", flush=True)

    samples_by_nh = {}
    for nh in sorted({j[0] for j in JOBS}):
        s = select_samples(nh, 2)  # sample selection is delta-independent
        samples_by_nh[nh] = s
        print(f"[PSM-scale] nh={nh} samples={s}", flush=True)

    all_results = []
    out = RESULTS_DIR / f"psm_scale_{run_id}.json"

    # Group by (nh, delta, method), apply same early-stopping discipline as realistic_large_benchmark
    grouped = {}
    for (nh, delta, key, cli, env, label) in JOBS:
        grouped.setdefault((nh, delta), []).append((key, cli, env, label))

    total = sum(len(v) * len(samples_by_nh[nh]) for (nh, _), v in grouped.items())
    done = 0

    for (nh, delta), methods in grouped.items():
        samples = samples_by_nh[nh]
        for (method_key, cli_args, env_over, label) in methods:
            consecutive_timeouts = 0
            EARLY = 2
            for sno in samples:
                done += 1
                tag = f"[{done}/{total}] nh={nh} δ={delta} m={method_key} s={sno}"
                if consecutive_timeouts >= EARLY:
                    print(tag + " SKIPPED (early-stop)", flush=True)
                    all_results.append({
                        "method": method_key, "label": label, "n_hidden": nh, "delta": delta,
                        "sample_no": sno, "wall_time": None, "solver_time": None,
                        "verdict": "skipped_early", "timed_out": False,
                    })
                    continue
                print(tag + " ...", flush=True)
                r = run_one(method_key, cli_args, env_over, nh, delta, sno, run_id)
                r["label"] = label
                all_results.append(r)
                if r.get("timed_out"):
                    consecutive_timeouts += 1
                else:
                    consecutive_timeouts = 0
                w = r.get("wall_time")
                ws = f"{w:.2f}s" if isinstance(w, (int, float)) else "N/A"
                psm = ""
                if r.get("psm_checks") is not None:
                    psm = f" psm={r['psm_hits']}/{r['psm_checks']}"
                state = ""
                if r.get("state_checks") is not None:
                    state = f" state={r['state_hits']}/{r['state_checks']}"
                print(f"   wall={ws} v={r.get('verdict')}{psm}{state}", flush=True)

                # Incremental dump
                out.write_text(json.dumps({
                    "config": {
                        "num_steps": NUM_STEPS, "samples": NUM_SAMPLES,
                        "per_sample_timeout_s": PER_SAMPLE_TIMEOUT_S,
                        "seed": SEED, "run_id": run_id,
                        "jobs": [{"nh": nh, "delta": d, "method": k, "label": l}
                                 for (nh, d, k, _, _, l) in JOBS],
                    },
                    "results": all_results,
                }, indent=2))

    print(f"\n[PSM-scale] done → {out}", flush=True)


if __name__ == "__main__":
    main()
