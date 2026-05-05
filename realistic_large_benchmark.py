#!/usr/bin/env python3
"""
Realistic-Large-Network Benchmark
=================================
Compares proposed BnB variants vs Z3/MILP/Exhaustive baselines on
T=5, n_hidden ∈ {100, 200, 300, 500} MLP SNNs (MNIST, delta=1).

Per-sample timeout: 300s (subprocess-level).
Fixed seed=42 → deterministic sample selection across methods.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from random import sample as random_sample, seed as random_seed

import numpy as np

ROOT = Path(__file__).parent.resolve()
LOG_DIR = ROOT / "log"
RESULTS_DIR = ROOT / "bench_results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
NUM_STEPS = 5
DELTA = 1
NUM_SAMPLES = 10
PER_SAMPLE_TIMEOUT_S = 300
SUBPROC_BUFFER_S = 30  # extra budget for import + model load + logging flush
N_HIDDEN_LIST = [100, 200, 300, 500]

# (method_key, cli_args, env_overrides, label)
METHODS = [
    ("exhaustive", None, {}, "Exhaustive DFS (oracle)"),
    ("z3", ["--z3"], {}, "Z3 SMT"),
    ("milp", ["--milp"], {}, "MILP (CBC)"),
    ("bnb_efficient", ["--np"], {"SNN_BNB_EFFICIENT": "1", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "0"}, "BnB (efficient)"),
    ("bnb_legacy", ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0"}, "BnB (legacy margin)"),
    ("bnb_legacy_ibp", ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0", "SNN_BNB_IBP": "1", "SNN_BNB_IBP_EVERY": "100"}, "BnB (legacy+IBP uncoupled)"),
    ("bnb_legacy_ibp_coupled", ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "1", "SNN_BNB_PSM": "0", "SNN_BNB_IBP": "1", "SNN_BNB_IBP_COUPLED": "1", "SNN_BNB_IBP_EVERY": "100"}, "BnB (legacy+IBP coupled)"),
    ("bnb_nofilter", ["--np"], {"SNN_BNB_EFFICIENT": "0", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "0"}, "BnB (no filter)"),
    ("bnb_psm", ["--np", "--psm"], {"SNN_BNB_EFFICIENT": "1", "SNN_BNB_LEGACY_ACTIVE_SET": "0", "SNN_BNB_PSM": "1"}, "BnB (efficient+PSM)"),
]


def load_model_and_samples(n_hidden):
    """Load weights and deterministically select NUM_SAMPLES valid sample indices."""
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
        deltas=(DELTA,),
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=NUM_STEPS,
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist(cfg)

    random_seed(SEED)
    np.random.seed(SEED)
    sample_pool = random_sample([*range(len(images))], k=NUM_SAMPLES * 3)  # more than needed, filter later

    chosen = []
    for sample_no in sample_pool:
        img = images[sample_no]
        ft = []
        orig_pred = forward(cfg, weights, img, ft)
        if len(np.argwhere(ft[-1] == np.min(ft[-1]))[0]) != 1:
            continue  # tie-breaking ambiguity → skip (matches run_test behavior)
        chosen.append({"sample_no": int(sample_no), "orig_pred": int(orig_pred), "label": int(labels[sample_no])})
        if len(chosen) >= NUM_SAMPLES:
            break

    return chosen


VERDICT_PATTERNS = [
    # BnB / Z3
    (re.compile(r"Not robust for sample .* and delta"), "not_robust"),
    (re.compile(r"\bRobust for sample .* and delta"), "robust"),
    # MILP
    (re.compile(r"status:\s*Not Robust"), "not_robust"),
    (re.compile(r"status:\s*Robust"), "robust"),
    (re.compile(r"status:\s*Unknown"), "unknown"),
]

TIME_PATTERNS = [
    re.compile(r"Checking done in time (\d+\.?\d*)"),
    re.compile(r"time:\s*(\d+\.\d+)"),
]


def parse_log(log_path):
    """Extract verdict + solver-reported time from a log file."""
    verdict = "unknown"
    solver_time = None
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return verdict, solver_time
    for pat, v in VERDICT_PATTERNS:
        if pat.search(text):
            verdict = v
            break
    for pat in TIME_PATTERNS:
        m = pat.search(text)
        if m:
            solver_time = float(m.group(1))
            break
    return verdict, solver_time


def run_baseline_subprocess(method_key, cli_args, env_overrides, n_hidden, sample_no, run_id):
    """Invoke batch_test.py for a single sample. Returns dict with result info."""
    prefix = f"bench_{run_id}_{method_key}_nh{n_hidden}_s{sample_no}"
    cmd = [
        sys.executable,
        str(ROOT / "batch_test.py"),
        "-p", prefix,
        "--test-type", "mnist",
        "--num-samples", "1",
        "--manual-indices", str(sample_no),
        "--n-hidden-neurons", str(n_hidden),
        "--num-steps", str(NUM_STEPS),
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
            cmd,
            env=env,
            cwd=str(ROOT),
            timeout=PER_SAMPLE_TIMEOUT_S + SUBPROC_BUFFER_S,
            capture_output=True,
            text=True,
        )
        rc = proc.returncode
        stderr_tail = (proc.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        rc = -1
        stderr_tail = "TIMEOUT"
        timed_out = True
    wall = time.time() - start

    # Locate the most recent log file for this prefix
    log_candidates = sorted(LOG_DIR.glob(f"*_{prefix}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if log_candidates:
        verdict, solver_time = parse_log(log_candidates[0])
        log_used = str(log_candidates[0].name)
    else:
        verdict, solver_time, log_used = "unknown", None, None

    if timed_out:
        verdict = "timeout"

    return {
        "method": method_key,
        "n_hidden": n_hidden,
        "sample_no": sample_no,
        "wall_time": wall,
        "solver_time": solver_time,
        "verdict": verdict,
        "timed_out": timed_out,
        "returncode": rc,
        "stderr_tail": stderr_tail,
        "log_file": log_used,
    }


def run_exhaustive_inprocess(cfg_kwargs, weights, sample):
    """Inline exhaustive DFS (ground-truth oracle). Respects signal.alarm."""
    import signal
    from utils.config import CFG
    from utils.mnist_net import forward

    cfg = CFG(**cfg_kwargs)
    num_classes = weights[-1].shape[0]
    max_t = cfg.num_steps
    pixels = [(i, j) for i in range(28) for j in range(28)]

    img_np = np.array(sample["img"])
    orig_pred = sample["orig_pred"]
    delta = cfg.deltas[0]
    found = [False]

    def dfs(cur, pos, rem):
        if found[0]:
            return
        spks = []
        forward(cfg, weights, cur, spks)
        last = spks[-1]
        target = last[orig_pred]
        min_nt = min(last[i] for i in range(num_classes) if i != orig_pred)
        is_adv = (min_nt < target) or (
            min_nt == target and any(last[i] == target for i in range(orig_pred))
        )
        if is_adv:
            if int(np.sum(np.abs(cur.astype(int) - img_np.astype(int)))) <= delta:
                found[0] = True
                return
        if pos == len(pixels) or rem == 0:
            return
        x, y = pixels[pos]
        ov = int(cur[x, y])
        for v in range(max_t):
            c = abs(v - ov)
            if c > rem:
                continue
            cur[x, y] = v
            dfs(cur, pos + 1, rem - c)
            cur[x, y] = ov
            if found[0]:
                return

    def handler(signum, frame):
        raise TimeoutError("per-sample timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(PER_SAMPLE_TIMEOUT_S)
    start = time.time()
    timed_out = False
    try:
        dfs(img_np.copy(), 0, delta)
        verdict = "not_robust" if found[0] else "robust"
    except TimeoutError:
        verdict = "timeout"
        timed_out = True
    finally:
        signal.alarm(0)
    wall = time.time() - start
    return {
        "wall_time": wall,
        "solver_time": wall,
        "verdict": verdict,
        "timed_out": timed_out,
    }


def run_full_matrix():
    run_id = time.strftime("%m%d%H%M%S")
    all_results = []
    print(f"Benchmark run_id = {run_id}")
    print(f"Config: num_steps={NUM_STEPS}, delta={DELTA}, samples={NUM_SAMPLES}, per-sample timeout={PER_SAMPLE_TIMEOUT_S}s")
    print()

    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import prepare_weights

    per_nh_samples = {}
    for n_hidden in N_HIDDEN_LIST:
        print(f"=== Loading model n_hidden={n_hidden} and selecting samples ===")
        picks = load_model_and_samples(n_hidden)
        # Attach images for inline exhaustive runs
        cfg_kw = dict(
            log_name=f"probe_{n_hidden}",
            subtype="mnist",
            load_data_func=load_mnist,
            seed=SEED,
            num_samples=NUM_SAMPLES,
            deltas=(DELTA,),
            n_layer_neurons=(784, n_hidden, 10),
            layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
            num_steps=NUM_STEPS,
        )
        cfg = CFG(**cfg_kw)
        images, *_ = load_mnist(cfg)
        for p in picks:
            p["img"] = images[p["sample_no"]].tolist()
        per_nh_samples[n_hidden] = {"cfg_kw": cfg_kw, "samples": picks}
        print(f"  Selected {len(picks)} samples: {[p['sample_no'] for p in picks]}")

    total_cases = sum(len(v["samples"]) for v in per_nh_samples.values()) * len(METHODS)
    done = 0
    method_blacklist_per_nh = set()  # (method_key) if it timed out on all samples at some n_h, skip larger

    for n_hidden in N_HIDDEN_LIST:
        info = per_nh_samples[n_hidden]
        weights = prepare_weights(cfg=CFG(**info["cfg_kw"]), subtype="mnist", load_data_func=load_mnist)

        for method_key, cli_args, env_over, label in METHODS:
            if method_key in method_blacklist_per_nh:
                print(f"[SKIP] {label} on n_h={n_hidden} (blacklisted from smaller n_h)")
                for s in info["samples"]:
                    done += 1
                    all_results.append({
                        "method": method_key, "label": label, "n_hidden": n_hidden,
                        "sample_no": s["sample_no"], "wall_time": None, "solver_time": None,
                        "verdict": "skipped", "timed_out": False,
                        "orig_pred": s["orig_pred"],
                    })
                continue

            method_timeouts = 0
            consecutive_timeouts = 0
            EARLY_STOP_CONSEC = 2
            skipped_samples = []
            for s in info["samples"]:
                done += 1
                tag = f"[{done}/{total_cases}] nh={n_hidden} method={method_key} sample={s['sample_no']}"

                if consecutive_timeouts >= EARLY_STOP_CONSEC:
                    print(tag + " SKIPPED (early-stop)", flush=True)
                    all_results.append({
                        "method": method_key, "label": label, "n_hidden": n_hidden,
                        "sample_no": s["sample_no"], "wall_time": None, "solver_time": None,
                        "verdict": "skipped_early", "timed_out": False,
                        "orig_pred": s["orig_pred"],
                    })
                    skipped_samples.append(s["sample_no"])
                    continue

                print(tag + " ...", flush=True)

                if method_key == "exhaustive":
                    r = run_exhaustive_inprocess(info["cfg_kw"], weights, s)
                    r["method"] = method_key
                    r["n_hidden"] = n_hidden
                    r["sample_no"] = s["sample_no"]
                else:
                    r = run_baseline_subprocess(method_key, cli_args, env_over, n_hidden, s["sample_no"], run_id)

                r["label"] = label
                r["orig_pred"] = s["orig_pred"]
                all_results.append(r)
                if r.get("timed_out"):
                    method_timeouts += 1
                    consecutive_timeouts += 1
                else:
                    consecutive_timeouts = 0
                wall = r.get('wall_time')
                st = r.get('solver_time')
                wall_str = f"{wall:.3f}s" if isinstance(wall, (int, float)) else "N/A"
                print(f"   wall={wall_str} solver={st} verdict={r.get('verdict')}")

            if skipped_samples or (method_timeouts == len(info["samples"])):
                method_blacklist_per_nh.add(method_key)
                print(f"[BLACKLIST] {label} after n_h={n_hidden} (skipped={len(skipped_samples)}, timeouts={method_timeouts})")

            # Dump incrementally
            out = RESULTS_DIR / f"realistic_bench_{run_id}.json"
            out.write_text(json.dumps({
                "config": {
                    "num_steps": NUM_STEPS, "delta": DELTA, "samples": NUM_SAMPLES,
                    "per_sample_timeout_s": PER_SAMPLE_TIMEOUT_S,
                    "n_hidden_list": N_HIDDEN_LIST,
                    "methods": [{"key": k, "label": lbl} for k, _, _, lbl in METHODS],
                    "run_id": run_id,
                    "seed": SEED,
                },
                "results": all_results,
            }, indent=2))

    print(f"\nBenchmark complete. Results → {out}")
    return out, all_results


if __name__ == "__main__":
    run_full_matrix()
