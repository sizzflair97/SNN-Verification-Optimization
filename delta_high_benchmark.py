#!/usr/bin/env python3
"""delta>=3 sweep across MNIST (n_h=200, 500) / N-MNIST / DVS Gesture.

Goal: probe BC-IBP's own scaling with perturbation budget delta. Baseline
(voltage-margin only) is expected to T.O. on most rows; the informative
signal is BC-IBP wall time growth with delta.

Soundness: uses SNN_BNB_LEGACY_ACTIVE_SET=1 for all configs (efficient/PSM
filters are unsound at delta>=2 per prior experience).
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
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

METHODS = [
    ("bnb_legacy", ["--np"],
        {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0"},
        "BnB (voltage-margin filter)"),
    ("bnb_legacy_ibp_coupled", ["--np"],
        {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0",
         "SNN_BNB_IBP": "1", "SNN_BNB_IBP_COUPLED": "1", "SNN_BNB_IBP_EVERY": "100"},
        "BnB + BC-IBP coupled"),
]

DATASETS = {
    "mnist":       {"loader": "load_mnist",       "input_shape": (28, 28),  "input_size": 784,    "num_classes": 10, "cli_type": "mnist"},
    "nmnist":      {"loader": "load_nmnist",      "input_shape": (64, 32),  "input_size": 2048,   "num_classes": 10, "cli_type": "nmnist"},
    "dvs_gesture": {"loader": "load_dvs_gesture", "input_shape": (256, 128),"input_size": 32768,  "num_classes": 11, "cli_type": "dvs_gesture"},
}

# (subtype, n_hidden, [deltas])
SWEEPS = [
    ("mnist",       200, [3, 4, 5]),
    ("mnist",       500, [3, 4, 5]),
    ("nmnist",      100, [3]),
    ("dvs_gesture", 100, [3]),
]


def _load_module():
    sys.path.insert(0, str(ROOT))
    from utils import load as L
    return L


def select_samples(subtype, n_hidden):
    L = _load_module()
    from utils.config import CFG
    from utils.mnist_net import forward, prepare_weights

    info = DATASETS[subtype]
    loader = getattr(L, info["loader"])
    cfg = CFG(
        log_name=f"probe_{subtype}", subtype=subtype,
        load_data_func=loader, seed=SEED, num_samples=NUM_SAMPLES,
        deltas=(1,),
        n_layer_neurons=(info["input_size"], n_hidden, info["num_classes"]),
        layer_shapes=(info["input_shape"], (n_hidden, 1), (info["num_classes"], 1)),
        num_steps=NUM_STEPS,
    )
    weights = prepare_weights(cfg=cfg, subtype=subtype, load_data_func=loader)
    images, labels, *_ = loader(cfg)

    random_seed(SEED); np.random.seed(SEED)
    pool = random_sample([*range(len(images))], k=min(len(images), NUM_SAMPLES * 5))
    chosen = []
    for sno in pool:
        ft = []
        forward(cfg, weights, images[sno], ft)
        if len(np.argwhere(ft[-1] == np.min(ft[-1]))[0]) != 1:
            continue
        chosen.append(int(sno))
        if len(chosen) >= NUM_SAMPLES:
            break
    return chosen


VERDICT_PATTERNS = [
    (re.compile(r"Not robust for sample .* and delta"), "not_robust"),
    (re.compile(r"\bRobust for sample .* and delta"), "robust"),
]
TIME_PAT   = re.compile(r"Checking done in time (\d+\.?\d*)")
FILTER_PAT = re.compile(r"Filtered pixels:\s*(\d+)\s*->\s*(\d+)")
IBP_PAT    = re.compile(r"IBP:\s*(\d+)\s*calls,\s*(\d+)\s*prunes")


def parse_log(log_path):
    out = {"verdict": "unknown", "solver_time": None,
           "pixels_before": None, "pixels_after": None,
           "ibp_calls": None, "ibp_prunes": None}
    try:
        text = Path(log_path).read_text(errors="ignore")
    except FileNotFoundError:
        return out
    for pat, v in VERDICT_PATTERNS:
        if pat.search(text):
            out["verdict"] = v; break
    m = TIME_PAT.search(text)
    if m: out["solver_time"] = float(m.group(1))
    m = FILTER_PAT.search(text)
    if m: out["pixels_before"], out["pixels_after"] = int(m.group(1)), int(m.group(2))
    m = IBP_PAT.search(text)
    if m: out["ibp_calls"], out["ibp_prunes"] = int(m.group(1)), int(m.group(2))
    return out


def run_one(subtype, n_hidden, method_key, cli_args, env_overrides, delta, sample_no, run_id):
    info = DATASETS[subtype]
    prefix = f"dhi_{run_id}_{subtype}_nh{n_hidden}_{method_key}_d{delta}_s{sample_no}"
    cmd = [
        sys.executable, str(ROOT / "batch_test.py"),
        "-p", prefix, "--test-type", info["cli_type"],
        "--num-samples", "1", "--manual-indices", str(sample_no),
        "--n-hidden-neurons", str(n_hidden),
        "--num-steps", str(NUM_STEPS),
        "--delta-max", str(delta), "--seed", str(SEED), *cli_args,
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    start = time.time(); timed = False
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                              timeout=PER_SAMPLE_TIMEOUT_S + SUBPROC_BUFFER_S,
                              capture_output=True, text=True)
        rc = proc.returncode
        stderr_tail = (proc.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        rc = -1; stderr_tail = "TIMEOUT"; timed = True
    wall = time.time() - start

    cands = sorted(LOG_DIR.glob(f"*_{prefix}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_info = {"verdict": "unknown"}; log_used = None
    if cands:
        log_used = cands[0].name
        log_info = parse_log(cands[0])
    verdict = "timeout" if timed else log_info.get("verdict", "unknown")

    return {
        "subtype": subtype, "n_hidden": n_hidden,
        "method": method_key, "delta": delta, "sample_no": sample_no,
        "wall_time": wall, "solver_time": log_info.get("solver_time"),
        "verdict": verdict, "timed_out": timed, "returncode": rc,
        "stderr_tail": stderr_tail, "log_file": log_used,
        "pixels_before": log_info.get("pixels_before"),
        "pixels_after": log_info.get("pixels_after"),
        "ibp_calls": log_info.get("ibp_calls"),
        "ibp_prunes": log_info.get("ibp_prunes"),
    }


def main():
    run_id = time.strftime("%m%d%H%M%S")
    print(f"[DeltaHi] run_id={run_id}  sweeps={len(SWEEPS)}", flush=True)

    # Pre-select samples per (subtype, n_hidden); BC-IBP and baseline share the set.
    sample_map = {}
    for subtype, n_hidden, _deltas in SWEEPS:
        key = (subtype, n_hidden)
        if key in sample_map: continue
        samples = select_samples(subtype, n_hidden)
        sample_map[key] = samples
        print(f"[DeltaHi] {subtype} n_h={n_hidden} samples={samples}", flush=True)

    results = []
    out = RESULTS_DIR / f"delta_high_{run_id}.json"
    EARLY_STOP = 2

    total_runs = sum(len(d) * len(METHODS) * len(sample_map[(s, nh)])
                     for s, nh, d in SWEEPS)
    done = 0
    for subtype, n_hidden, deltas in SWEEPS:
        samples = sample_map[(subtype, n_hidden)]
        for delta in deltas:
            for (mkey, cli_args, env_over, label) in METHODS:
                consec_to = 0
                for sno in samples:
                    done += 1
                    tag = f"[{done}/{total_runs}] {subtype} nh={n_hidden} δ={delta} m={mkey} s={sno}"
                    if consec_to >= EARLY_STOP:
                        print(tag + " SKIPPED (early-stop)", flush=True)
                        results.append({
                            "subtype": subtype, "n_hidden": n_hidden,
                            "method": mkey, "label": label, "delta": delta,
                            "sample_no": sno, "wall_time": None, "solver_time": None,
                            "verdict": "skipped_early", "timed_out": False,
                        })
                        continue
                    print(tag + " ...", flush=True)
                    r = run_one(subtype, n_hidden, mkey, cli_args, env_over, delta, sno, run_id)
                    r["label"] = label
                    results.append(r)
                    consec_to = consec_to + 1 if r.get("timed_out") else 0
                    w = r.get("wall_time"); ws = f"{w:.2f}s" if isinstance(w, (int, float)) else "N/A"
                    extra = ""
                    if r.get("ibp_prunes") is not None:
                        extra += f" ibp={r['ibp_prunes']}/{r['ibp_calls']}"
                    if r.get("pixels_before") is not None:
                        extra += f" px={r['pixels_after']}/{r['pixels_before']}"
                    print(f"   wall={ws} v={r.get('verdict')}{extra}", flush=True)

                    out.write_text(json.dumps({
                        "config": {
                            "run_id": run_id, "seed": SEED,
                            "num_samples": NUM_SAMPLES,
                            "per_sample_timeout_s": PER_SAMPLE_TIMEOUT_S,
                            "num_steps": NUM_STEPS,
                            "sweeps": [{"subtype": s, "n_hidden": nh, "deltas": d} for s, nh, d in SWEEPS],
                            "methods": [{"key": k, "label": l} for k, _, _, l in METHODS],
                        },
                        "results": results,
                    }, indent=2))

    print(f"\n[DeltaHi] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
