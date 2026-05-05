#!/usr/bin/env python3
"""
T-sweep verification benchmark (Phase 1 of cross-timestep tightening)
=====================================================================
Measures how current coupled-IBP bound tightness degrades as T grows.
Reuses existing MNIST MLP models at various (T, n_h):

    Primary axis (n_h=50): T in {5, 20, 256}
    Secondary axis (n_h=256): T in {5, 256}

Methods: bnb_legacy (filter-only) vs bnb_legacy_ibp_coupled.
delta=2, 10 samples, seed=42, 300s per-sample timeout.
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
DELTA = 2
NUM_SAMPLES = 10
PER_SAMPLE_TIMEOUT_S = 300
SUBPROC_BUFFER_S = 30

# (T, n_hidden) configurations — only uses models that exist in models/
# Primary sweep (n_h=50, T wide range):
#   (5,50), (20,50), (256,50)
# Secondary sweep (large n_h, T ∈ {128,256}):
#   (128,256), (256,256)
CONFIGS = [
    (5,   50),
    (20,  50),
    (256, 50),
    (128, 256),
    (256, 256),
]

METHODS = [
    ("bnb_legacy", ["--np"],
        {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0"},
        "BnB (voltage-margin filter)"),
    ("bnb_legacy_ibp_coupled", ["--np"],
        {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0",
         "SNN_BNB_IBP": "1", "SNN_BNB_IBP_COUPLED": "1", "SNN_BNB_IBP_EVERY": "100"},
        "BnB + BC-IBP coupled"),
]


def select_samples(num_steps, n_hidden):
    sys.path.insert(0, str(ROOT))
    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights

    cfg = CFG(
        log_name=f"probe_T{num_steps}_nh{n_hidden}",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        deltas=(DELTA,),
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=num_steps,
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist(cfg)

    random_seed(SEED)
    np.random.seed(SEED)
    pool = random_sample([*range(len(images))], k=NUM_SAMPLES * 5)

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
STATE_PAT = re.compile(r"State cache:\s*(\d+)\s*checks,\s*(\d+)\s*hits")
FILTER_PAT = re.compile(r"Filtered pixels:\s*(\d+)\s*->\s*(\d+)")
IBP_PAT = re.compile(r"IBP:\s*(\d+)\s*calls,\s*(\d+)\s*prunes")


def parse_log(log_path):
    out = {"verdict": "unknown", "solver_time": None,
           "state_checks": None, "state_hits": None,
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
    m = STATE_PAT.search(text)
    if m: out["state_checks"], out["state_hits"] = int(m.group(1)), int(m.group(2))
    m = FILTER_PAT.search(text)
    if m: out["pixels_before"], out["pixels_after"] = int(m.group(1)), int(m.group(2))
    m = IBP_PAT.search(text)
    if m: out["ibp_calls"], out["ibp_prunes"] = int(m.group(1)), int(m.group(2))
    return out


def run_one(method_key, cli_args, env_overrides, T, n_hidden, sample_no, run_id):
    prefix = f"tsweep_{run_id}_{method_key}_T{T}_nh{n_hidden}_s{sample_no}"
    cmd = [
        sys.executable,
        str(ROOT / "batch_test.py"),
        "-p", prefix,
        "--test-type", "mnist",
        "--num-samples", "1",
        "--manual-indices", str(sample_no),
        "--n-hidden-neurons", str(n_hidden),
        "--num-steps", str(T),
        "--delta-max", str(DELTA),
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

    cands = sorted(LOG_DIR.glob(f"*_{prefix}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_info = {"verdict": "unknown"}
    log_used = None
    if cands:
        log_used = cands[0].name
        log_info = parse_log(cands[0])
    verdict = "timeout" if timed_out else log_info.get("verdict", "unknown")

    return {
        "method": method_key, "T": T, "n_hidden": n_hidden, "sample_no": sample_no,
        "wall_time": wall, "solver_time": log_info.get("solver_time"),
        "verdict": verdict, "timed_out": timed_out, "returncode": rc,
        "stderr_tail": stderr_tail, "log_file": log_used,
        "state_checks": log_info.get("state_checks"),
        "state_hits": log_info.get("state_hits"),
        "pixels_before": log_info.get("pixels_before"),
        "pixels_after": log_info.get("pixels_after"),
        "ibp_calls": log_info.get("ibp_calls"),
        "ibp_prunes": log_info.get("ibp_prunes"),
    }


def main():
    run_id = time.strftime("%m%d%H%M%S")
    print(f"[T-sweep] run_id={run_id} δ={DELTA}, configs={CONFIGS}", flush=True)

    samples_per_cfg = {}
    for (T, nh) in CONFIGS:
        picks = select_samples(T, nh)
        samples_per_cfg[(T, nh)] = picks
        print(f"[T-sweep] (T={T}, n_h={nh}) samples: {picks}", flush=True)

    all_results = []
    out = RESULTS_DIR / f"tsweep_bench_{run_id}.json"

    total = sum(len(samples_per_cfg[k]) for k in samples_per_cfg) * len(METHODS)
    done = 0
    for (T, nh) in CONFIGS:
        samples = samples_per_cfg[(T, nh)]
        for (method_key, cli_args, env_over, label) in METHODS:
            consecutive_timeouts = 0
            EARLY_STOP = 2
            for sno in samples:
                done += 1
                tag = f"[{done}/{total}] T={T} nh={nh} m={method_key} s={sno}"
                if consecutive_timeouts >= EARLY_STOP:
                    print(tag + " SKIPPED (early-stop)", flush=True)
                    all_results.append({
                        "method": method_key, "label": label, "T": T, "n_hidden": nh,
                        "sample_no": sno, "wall_time": None, "solver_time": None,
                        "verdict": "skipped_early", "timed_out": False,
                    })
                    continue
                print(tag + " ...", flush=True)
                r = run_one(method_key, cli_args, env_over, T, nh, sno, run_id)
                r["label"] = label
                all_results.append(r)
                if r.get("timed_out"):
                    consecutive_timeouts += 1
                else:
                    consecutive_timeouts = 0
                w = r.get("wall_time")
                ws = f"{w:.2f}s" if isinstance(w, (int, float)) else "N/A"
                extra = ""
                if r.get("state_checks") is not None:
                    extra += f" nodes={r['state_checks']}"
                if r.get("ibp_prunes") is not None:
                    extra += f" ibp={r['ibp_prunes']}/{r['ibp_calls']}"
                if r.get("pixels_before") is not None:
                    extra += f" px={r['pixels_after']}/{r['pixels_before']}"
                print(f"   wall={ws} v={r.get('verdict')}{extra}", flush=True)

                out.write_text(json.dumps({
                    "config": {
                        "subtype": "mnist",
                        "delta": DELTA, "samples": NUM_SAMPLES,
                        "per_sample_timeout_s": PER_SAMPLE_TIMEOUT_S,
                        "seed": SEED, "run_id": run_id,
                        "configs": [{"T": t, "n_hidden": n} for (t, n) in CONFIGS],
                        "methods": [{"key": k, "label": l} for k, _, _, l in METHODS],
                    },
                    "results": all_results,
                }, indent=2))

    print(f"\n[T-sweep] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
