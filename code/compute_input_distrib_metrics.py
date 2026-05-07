"""Compute per-dataset spike-time distribution statistics.

For each dataset's TTFS-encoded inputs we compute, per sample:
  p_x(t) = #{p : orig-TTFS(x)_p = t} / N           (histogram over T bins)
  C(x)   = max_t p_x(t)                            (peak concentration)
  H(x)   = -sum_t p_x(t) * log p_x(t)              (Shannon entropy, nats)

Aggregates median, mean, std across N samples per dataset (default 200) and
prints them so we can compare which statistic correlates most cleanly with
the BC-IBP solver-time table.
"""
from __future__ import annotations
import argparse
import sys
import os.path as osp

import numpy as np

_HERE = osp.dirname(osp.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.config import CFG
from utils.load import (
    load_mnist, load_fmnist, load_cifar,
    load_nmnist, load_dvs_gesture, load_cifar10_dvs,
)


# Dataset registry: load_func, T, layer_shapes, n_samples
DATASETS = {
    "MNIST":         (load_mnist,         5,   ((28, 28), (1, 1), (10, 1)),       200),
    "FashionMNIST":  (load_fmnist,        256, ((28, 28), (1, 1), (10, 1)),       200),
    "CIFAR10":       (load_cifar,         5,   ((96, 32), (1, 1), (10, 1)),       200),
    "N-MNIST":       (load_nmnist,        5,   ((64, 32), (1, 1), (10, 1)),       200),
    "DVS-Gesture":   (load_dvs_gesture,   5,   ((256, 128), (1, 1), (11, 1)),     200),
    "CIFAR10-DVS":   (load_cifar10_dvs,   5,   ((256, 128), (1, 1), (10, 1)),     200),
}


def stats_for_image(x: np.ndarray, T: int):
    """x is integer spike-time array (any shape) with values in [0, T-1].
    Returns (C, H, N)."""
    flat = x.flatten().astype(int)
    N = flat.size
    counts = np.bincount(flat, minlength=T).astype(np.float64)
    p = counts / N
    C = float(p.max())
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())   # natural log; in nats
    return C, H, N


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()),
                    help="Subset of datasets to compute on.")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="How many samples per dataset (overrides registry).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"{'Dataset':<14}  {'T':>3}  {'N':>7}   "
          f"{'C med':>6}  {'C mean':>7}  "
          f"{'H med':>6}  {'H mean':>7}  {'H/log T':>8}  {'1/exp(H)':>9}")
    print("-" * 95)
    results = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"  [skip] unknown dataset {name}")
            continue
        load_fn, T, layer_shapes, default_n = DATASETS[name]
        n = args.n_samples or default_n
        cfg = CFG(
            log_name="distrib",
            subtype=name.lower().replace("-", "_"),
            load_data_func=load_fn,
            seed=args.seed,
            num_samples=n,
            n_layer_neurons=(int(np.prod(layer_shapes[0])), 1, layer_shapes[-1][0]),
            layer_shapes=layer_shapes,
            num_steps=T,
        )
        try:
            images, labels, *_ = load_fn(cfg)
        except Exception as e:
            print(f"{name:<14}  load failed: {e}")
            continue
        n_use = min(n, len(images))
        Cs, Hs, Ns = [], [], []
        for i in range(n_use):
            C, H, N = stats_for_image(np.asarray(images[i]), T)
            Cs.append(C); Hs.append(H); Ns.append(N)
        Cs = np.array(Cs); Hs = np.array(Hs)
        N_unique = sorted(set(Ns))
        N_str = str(N_unique[0]) if len(N_unique) == 1 else f"{min(Ns)}-{max(Ns)}"
        norm_h = Hs / max(np.log(T), 1e-12)         # entropy normalised by log T
        eff = 1.0 / np.exp(Hs)                       # 1 / effective # of bins (∝ "concentration")
        print(f"{name:<14}  {T:>3}  {N_str:>7}   "
              f"{np.median(Cs):>6.3f}  {np.mean(Cs):>7.3f}  "
              f"{np.median(Hs):>6.3f}  {np.mean(Hs):>7.3f}  "
              f"{np.median(norm_h):>8.3f}  {np.median(eff):>9.4f}")
        results[name] = {
            "T": T, "N": int(np.median(Ns)), "n": n_use,
            "C_median": float(np.median(Cs)), "C_mean": float(np.mean(Cs)),
            "H_median": float(np.median(Hs)), "H_mean": float(np.mean(Hs)),
            "H_norm_median": float(np.median(norm_h)),
            "inv_eff_median": float(np.median(eff)),
        }
    return results


if __name__ == "__main__":
    main()
