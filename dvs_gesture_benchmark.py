#!/usr/bin/env python3
"""DVS Gesture verification benchmark.

Runs BnB pipeline (voltage-margin filter, ±BC-IBP) on DVS Gesture TTFS
(T=5, inputs=2*128*128=32768, n_h=100, 11 classes). Mirrors the N-MNIST
benchmark protocol: per-sample subprocess, seed=42, 10 samples, 300s timeout.
"""
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

N_HIDDEN = 100
INPUT_SHAPE = (256, 128)
INPUT_SIZE = 256 * 128  # 32768
NUM_CLASSES = 11

METHODS = [
    ("bnb_legacy", ["--np"],
        {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0"},
        "BnB (voltage-margin filter)"),
    ("bnb_legacy_ibp_coupled", ["--np"],
        {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0",
         "SNN_BNB_IBP": "1", "SNN_BNB_IBP_COUPLED": "1", "SNN_BNB_IBP_EVERY": "100"},
        "BnB + BC-IBP coupled"),
]


def select_samples():
    sys.path.insert(0, str(ROOT))
    from utils.config import CFG
    from utils.load import load_dvs_gesture
    from utils.mnist_net import forward, prepare_weights

    cfg = CFG(
        log_name="probe_dvsgesture", subtype="dvs_gesture",
        load_data_func=load_dvs_gesture, seed=SEED, num_samples=NUM_SAMPLES,
        deltas=(1,),
        n_layer_neurons=(INPUT_SIZE, N_HIDDEN, NUM_CLASSES),
        layer_shapes=(INPUT_SHAPE, (N_HIDDEN, 1), (NUM_CLASSES, 1)),
        num_steps=NUM_STEPS,
    )
    weights = prepare_weights(cfg=cfg, subtype="dvs_gesture", load_data_func=load_dvs_gesture)
    images, labels, *_ = load_dvs_gesture(cfg)

    random_seed(SEED)
    np.random.seed(SEED)
    # Pool is larger than needed; filter by prediction uniqueness.
    pool = random_sample([*range(len(images))], k=min(len(images), NUM_SAMPLES * 5))

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
FILTER_PAT = re.compile(r"Filtered pixels:\s*(\d+)\s*->\s*(\d+)")
IBP_PAT = re.compile(r"IBP:\s*(\d+)\s*calls,\s*(\d+)\s*prunes")


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
            out["verdict"] = v
            break
    m = TIME_PAT.search(text)
    if m: out["solver_time"] = float(m.group(1))
    m = FILTER_PAT.search(text)
    if m: out["pixels_before"], out["pixels_after"] = int(m.group(1)), int(m.group(2))
    m = IBP_PAT.search(text)
    if m: out["ibp_calls"], out["ibp_prunes"] = int(m.group(1)), int(m.group(2))
    return out


def run_one(method_key, cli_args, env_overrides, delta, sample_no, run_id):
    prefix = f"dvs_{run_id}_{method_key}_d{delta}_s{sample_no}"
    cmd = [
        sys.executable, str(ROOT / "batch_test.py"),
        "-p", prefix, "--test-type", "dvs_gesture",
        "--num-samples", "1", "--manual-indices", str(sample_no),
        "--n-hidden-neurons", str(N_HIDDEN),
        "--num-steps", str(NUM_STEPS),
        "--delta-max", str(delta), "--seed", str(SEED), *cli_args,
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    start = time.time()
    timed = False
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
    print(f"[DVS-Gesture] run_id={run_id}  n_h={N_HIDDEN}, T={NUM_STEPS}", flush=True)

    samples = select_samples()
    print(f"[DVS-Gesture] selected samples: {samples}", flush=True)

    results = []
    out = RESULTS_DIR / f"dvsgesture_bench_{run_id}.json"

    DELTAS = [1, 2]
    total = len(DELTAS) * len(METHODS) * len(samples)
    done = 0
    for delta in DELTAS:
        for (mkey, cli_args, env_over, label) in METHODS:
            consec_to = 0
            EARLY_STOP = 2
            for sno in samples:
                done += 1
                tag = f"[{done}/{total}] δ={delta} m={mkey} s={sno}"
                if consec_to >= EARLY_STOP:
                    print(tag + " SKIPPED (early-stop)", flush=True)
                    results.append({
                        "method": mkey, "label": label, "delta": delta,
                        "sample_no": sno, "wall_time": None, "solver_time": None,
                        "verdict": "skipped_early", "timed_out": False,
                    })
                    continue
                print(tag + " ...", flush=True)
                r = run_one(mkey, cli_args, env_over, delta, sno, run_id)
                r["label"] = label
                results.append(r)
                if r.get("timed_out"):
                    consec_to += 1
                else:
                    consec_to = 0
                w = r.get("wall_time")
                ws = f"{w:.2f}s" if isinstance(w, (int, float)) else "N/A"
                extra = ""
                if r.get("ibp_prunes") is not None:
                    extra += f" ibp={r['ibp_prunes']}/{r['ibp_calls']}"
                if r.get("pixels_before") is not None:
                    extra += f" px={r['pixels_after']}/{r['pixels_before']}"
                print(f"   wall={ws} v={r.get('verdict')}{extra}", flush=True)

                out.write_text(json.dumps({
                    "config": {
                        "subtype": "dvs_gesture",
                        "n_hidden": N_HIDDEN, "num_steps": NUM_STEPS,
                        "deltas": DELTAS, "samples": NUM_SAMPLES,
                        "per_sample_timeout_s": PER_SAMPLE_TIMEOUT_S,
                        "seed": SEED, "run_id": run_id,
                        "methods": [{"key": k, "label": l} for k, _, _, l in METHODS],
                    },
                    "results": results,
                }, indent=2))

    print(f"\n[DVS-Gesture] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
