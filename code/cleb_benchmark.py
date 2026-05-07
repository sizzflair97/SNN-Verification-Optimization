"""End-to-end CLEB benchmark for Table 4 of the paper.

Per (architecture, Δ) row, runs on N forward-correct samples:
  - Exhaustive DFS oracle    → ground truth (robust / not_robust / timeout)
  - BC-IBP-multilayer at root → verdict + time
  - CLEB at root              → verdict + time

Restricts the reported root-prove rate denominator to oracle-confirmed
truly-robust samples, eliminating the forward-correct vs truly-robust
confound.

Usage:
  python cleb_benchmark.py --hidden 100 100 --tmax 4 --delta 1 --n-samples 10
  python cleb_benchmark.py --hidden 100 100 --tmax 4 --delta 2 --n-samples 10
  python cleb_benchmark.py --hidden 200 200 --tmax 4 --delta 1 --n-samples 10
  python cleb_benchmark.py --hidden 200 200 --tmax 4 --delta 2 --n-samples 10
"""
from __future__ import annotations
import argparse
import json
import os
import os.path as osp
import time

import numpy as np

from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.dictionary_mnist import threshold

from bnb_ibp import ibp_prove_robust_multilayer
from bnb_ibp_pair import ibp_prove_robust_multilayer_exact_gpu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", required=True,
                   help="Hidden-layer widths (e.g. 100 100, 200 200)")
    p.add_argument("--tmax", type=int, default=4)
    p.add_argument("--delta", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oracle-timeout", type=float, default=300.0)
    p.add_argument("--save-dir", type=str, default="bench_results")
    return p.parse_args()


def exhaustive_oracle(cfg, weights, img, orig_pred, delta, T, timeout):
    """DFS over all admissible TTFS perturbations within L¹ budget."""
    found = [False]
    timed_out = [False]
    pixels = [(i, j) for i in range(28) for j in range(28)]
    n_pixels = len(pixels)
    tic = time.time()

    def dfs(current_img, pixel_pos, rem_budget):
        if found[0] or timed_out[0]:
            return
        if time.time() - tic > timeout:
            timed_out[0] = True
            return
        pred = forward(cfg, weights, current_img)
        if pred != orig_pred:
            l1 = int(np.sum(np.abs(current_img.astype(int) - img.astype(int))))
            if l1 <= delta:
                found[0] = True
                return
        if pixel_pos == n_pixels or rem_budget == 0:
            return
        px, py = pixels[pixel_pos]
        orig_val = int(current_img[px, py])
        for new_val in range(T):
            cost = abs(new_val - orig_val)
            if cost > rem_budget:
                continue
            current_img[px, py] = new_val
            dfs(current_img, pixel_pos + 1, rem_budget - cost)
            if found[0] or timed_out[0]:
                current_img[px, py] = orig_val
                return
            current_img[px, py] = orig_val

    dfs(img.copy(), 0, delta)
    return ("not_robust" if found[0] else
            "timeout" if timed_out[0] else "robust")


def bcibp_root(weights, img, orig_pred, delta, T):
    """BC-IBP-multilayer at the BnB root: try every (rem_neg, rem_pos) split.
    Returns (verdict, wall-clock-time)."""
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)],
                          dtype=int)
    tic = time.time()
    proved = True
    for rn in range(delta + 1):
        rp = delta - rn
        ok = ibp_prove_robust_multilayer(
            pixel_img=img.copy(),
            remaining_pixels=all_pixels,
            rem_neg=rn, rem_pos=rp,
            weights_list=list(weights),
            num_steps=T,
            threshold=threshold,
            orig_pred=orig_pred,
            num_classes=10,
        )
        if not ok:
            proved = False
            break
    return ("robust" if proved else "inconclusive"), time.time() - tic


def cleb_root(weights, img, orig_pred, delta, T):
    """CLEB at the BnB root via the GPU enumerate path.
    Returns (verdict, wall-clock-time)."""
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)],
                          dtype=int)
    tic = time.time()
    proved = True
    for rn in range(delta + 1):
        rp = delta - rn
        ok = ibp_prove_robust_multilayer_exact_gpu(
            pixel_img=img.copy(),
            remaining_pixels=all_pixels,
            rem_neg=rn, rem_pos=rp,
            weights_list=list(weights),
            num_steps=T,
            threshold=threshold,
            orig_pred=orig_pred,
            num_classes=10,
        )
        if not ok:
            proved = False
            break
    return ("robust" if proved else "inconclusive"), time.time() - tic


def main():
    args = parse_args()
    T = args.tmax + 1
    n_layer_neurons = tuple([784] + list(args.hidden) + [10])
    layer_shapes = ((28, 28), *[(h, 1) for h in args.hidden], (10, 1))

    cfg = CFG(
        log_name="cleb_bench",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=args.seed,
        num_samples=args.n_samples,
        deltas=(args.delta,),
        n_layer_neurons=n_layer_neurons,
        layer_shapes=layer_shapes,
        num_steps=T,
    )
    arch_tag = "_".join(str(d) for d in n_layer_neurons)
    print(f"Architecture: {arch_tag}, T={T}, Δ={args.delta}")

    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    print(f"Loaded {len(weights)} weight layers, shapes: {[w.shape for w in weights]}")

    images, labels, *_ = load_mnist(cfg)
    np.random.seed(args.seed)
    indices = np.random.permutation(len(images))[:args.n_samples]

    rows = []
    print(f"\n{'idx':>5} {'pred':>4} {'true':>4} {'fwd':>4} | "
          f"{'oracle':>10} {'orc(s)':>7} | "
          f"{'BC-IBP':>12} {'bc(s)':>6} | "
          f"{'CLEB':>12} {'cleb(s)':>7}")
    print("-" * 100)
    for idx in indices:
        img = images[int(idx)].astype(int)
        true_lbl = int(labels[int(idx)])
        orig_pred = int(forward(cfg, weights, img))
        forward_correct = (orig_pred == true_lbl)

        if not forward_correct:
            print(f"{idx:>5} {orig_pred:>4} {true_lbl:>4} {'no':>4} | "
                  f"{'-':>10} {'-':>7} | {'-':>12} {'-':>6} | "
                  f"{'-':>12} {'-':>7}")
            rows.append({
                "idx": int(idx), "pred": orig_pred, "true": true_lbl,
                "forward_correct": False,
            })
            continue

        # Oracle
        tic = time.time()
        oracle_v = exhaustive_oracle(cfg, weights, img, orig_pred,
                                     args.delta, T, args.oracle_timeout)
        oracle_t = time.time() - tic

        # BC-IBP-multilayer (root only)
        bc_v, bc_t = bcibp_root(weights, img, orig_pred, args.delta, T)

        # CLEB (root only)
        cleb_v, cleb_t = cleb_root(weights, img, orig_pred, args.delta, T)

        rows.append({
            "idx": int(idx), "pred": orig_pred, "true": true_lbl,
            "forward_correct": True,
            "oracle": oracle_v, "oracle_time_s": oracle_t,
            "bcibp": bc_v, "bcibp_time_s": bc_t,
            "cleb": cleb_v, "cleb_time_s": cleb_t,
        })
        print(f"{idx:>5} {orig_pred:>4} {true_lbl:>4} {'yes':>4} | "
              f"{oracle_v:>10} {oracle_t:>7.2f} | "
              f"{bc_v:>12} {bc_t:>6.3f} | "
              f"{cleb_v:>12} {cleb_t:>7.3f}")

    # Aggregate
    fc = [r for r in rows if r.get("forward_correct")]
    truly_robust = [r for r in fc if r.get("oracle") == "robust"]
    bc_proved = [r for r in truly_robust if r.get("bcibp") == "robust"]
    cleb_proved = [r for r in truly_robust if r.get("cleb") == "robust"]

    print("\n=== Summary ===")
    print(f"Forward-correct: {len(fc)} / {args.n_samples}")
    print(f"Oracle robust  : {len(truly_robust)} / {len(fc)}  (denominator)")
    print(f"BC-IBP root-prove: {len(bc_proved)} / {len(truly_robust)}")
    print(f"CLEB   root-prove: {len(cleb_proved)} / {len(truly_robust)}")
    if truly_robust:
        bc_times = [r["bcibp_time_s"] for r in truly_robust]
        cleb_times = [r["cleb_time_s"] for r in truly_robust]
        print(f"BC-IBP time on truly-robust: "
              f"min={min(bc_times):.3f}, max={max(bc_times):.3f}, "
              f"mean={np.mean(bc_times):.3f}")
        print(f"CLEB   time on truly-robust: "
              f"min={min(cleb_times):.3f}, max={max(cleb_times):.3f}, "
              f"mean={np.mean(cleb_times):.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = osp.join(args.save_dir,
                        f"cleb_bench_{arch_tag}_d{args.delta}.json")
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
