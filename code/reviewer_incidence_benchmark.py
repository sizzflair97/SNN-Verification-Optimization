#!/usr/bin/env python3
"""Exact Delta=1/2 adversarial-incidence pilot for the OpenReview response.

The script selects deterministically chosen, correctly classified MNIST
examples with a unique earliest output spike. Each verification runs in its
own subprocess so that a hard instance cannot block the full pilot. Delta=2
is run only for examples proved robust at Delta=1; non-robustness is
monotonic in the perturbation budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "bench_results"
WITNESS_DIR = RESULTS_DIR / "reviewer_witnesses"

VERDICT_RE = re.compile(r"^(Not robust|Robust) for sample .* delta=(\d+)", re.MULTILINE)
TIME_RE = re.compile(r"Checking done in time (\d+\.?\d*)")
FILTER_RE = re.compile(r"Filtered pixels:\s*(\d+)\s*->\s*(\d+)")
IBP_RE = re.compile(r"IBP:\s*(\d+) calls,\s*(\d+) prunes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--n-hidden", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def select_examples(
    n_samples: int, n_hidden: int, seed: int, split: str = "test"
) -> list[dict]:
    cfg = CFG(
        log_name="reviewer_incidence_select",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=seed,
        num_samples=n_samples,
        deltas=(1,),
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=5,
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images_train, labels_train, images_test, labels_test = load_mnist(cfg)
    if split == "test":
        images, labels = images_test, labels_test
    elif split == "train":
        images, labels = images_train, labels_train
    else:
        raise ValueError("split must be either 'train' or 'test'")
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for idx in rng.permutation(len(images)):
        firing_times: list[np.ndarray] = []
        pred = int(forward(cfg, weights, images[int(idx)], firing_times))
        earliest = firing_times[-1] == np.min(firing_times[-1])
        if int(earliest.sum()) != 1 or pred != int(labels[int(idx)]):
            continue
        selected.append(
            {"index": int(idx), "label": int(labels[int(idx)]), "orig_pred": pred}
        )
        if len(selected) == n_samples:
            break
    if len(selected) != n_samples:
        raise RuntimeError(f"Only selected {len(selected)} of {n_samples} requested examples")
    return selected


def run_one(example: dict, delta: int, args: argparse.Namespace, run_id: str) -> dict:
    idx = int(example["index"])
    witness_path = WITNESS_DIR / f"{run_id}_idx{idx}_d{delta}.json"
    cmd = [
        sys.executable,
        str(ROOT / "batch_test.py"),
        "-p", f"review_{run_id}_idx{idx}_d{delta}",
        "--test-type", "mnist",
        "--np",
        "--n-hidden-neurons", str(args.n_hidden),
        "--num-steps", "5",
        "--delta-max", str(delta),
        "--manual-indices", str(idx),
        "--seed", str(args.seed),
    ]
    env = os.environ.copy()
    env.update(
        {
            "SNN_BNB_LEGACY_ACTIVE_SET": "1",
            "SNN_BNB_EFFICIENT": "0",
            "SNN_BNB_PSM": "0",
            "SNN_BNB_IBP": "1",
            "SNN_BNB_IBP_COUPLED": "1",
            "SNN_BNB_IBP_EVERY": "100",
            "SNN_WITNESS_PATH": str(witness_path),
            "SNN_DATA_SPLIT": args.split,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    tic = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        timed_out = False
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        match = VERDICT_RE.search(text)
        verdict = (
            "not_robust" if match and match.group(1) == "Not robust"
            else "robust" if match
            else "error"
        )
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        verdict = "timeout"
        returncode = -1
        text = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + "\n"
        text += ((exc.stderr or "") if isinstance(exc.stderr, str) else "")

    t_match = TIME_RE.search(text)
    f_match = FILTER_RE.search(text)
    i_match = IBP_RE.search(text)
    witness = None
    if witness_path.exists():
        witness = json.loads(witness_path.read_text())
        witness["sample_no"] = idx
        witness_path.write_text(json.dumps(witness, indent=2) + "\n")
    return {
        **example,
        "delta": delta,
        "verdict": verdict,
        "timed_out": timed_out,
        "returncode": returncode,
        "wall_time_s": time.time() - tic,
        "solver_time_s": float(t_match.group(1)) if t_match else None,
        "pixels_before": int(f_match.group(1)) if f_match else None,
        "pixels_after": int(f_match.group(2)) if f_match else None,
        "ibp_calls": int(i_match.group(1)) if i_match else None,
        "ibp_prunes": int(i_match.group(2)) if i_match else None,
        "witness": witness,
        "output_tail": text[-1200:] if verdict == "error" else None,
    }


def run_phase(
    examples: list[dict],
    delta: int,
    args: argparse.Namespace,
    run_id: str,
    results: list[dict],
    output: Path,
) -> None:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, ex, delta, args, run_id): ex for ex in examples
        }
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            print(
                f"[Delta={delta} {done}/{len(examples)}] idx={row['index']} "
                f"{row['verdict']} wall={row['wall_time_s']:.2f}s",
                flush=True,
            )
            output.write_text(
                json.dumps(
                    {
                        "config": vars(args) | {"output": str(output), "run_id": run_id},
                        "selected": selected_global,
                        "results": results,
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )


selected_global: list[dict] = []


def main() -> None:
    global selected_global
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    WITNESS_DIR.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = args.output or RESULTS_DIR / f"reviewer_incidence_{run_id}.json"
    selected_global = select_examples(
        args.n_samples, args.n_hidden, args.seed, args.split
    )
    print(f"Selected indices: {[x['index'] for x in selected_global]}", flush=True)

    results: list[dict] = []
    run_phase(selected_global, 1, args, run_id, results, output)
    d1_by_idx = {r["index"]: r for r in results if r["delta"] == 1}
    delta2_examples = [
        ex for ex in selected_global
        if d1_by_idx[ex["index"]]["verdict"] == "robust"
    ]
    run_phase(delta2_examples, 2, args, run_id, results, output)

    # A Delta=1 counterexample remains a counterexample for the Delta=2 ball.
    for ex in selected_global:
        if d1_by_idx[ex["index"]]["verdict"] == "not_robust":
            results.append(
                {
                    **ex,
                    "delta": 2,
                    "verdict": "not_robust",
                    "inherited_from_delta": 1,
                }
            )
    output.write_text(
        json.dumps(
            {
                "config": vars(args) | {"output": str(output), "run_id": run_id},
                "selected": selected_global,
                "results": results,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
