#!/usr/bin/env python3
"""Synthetic CLEB enumeration scaling benchmark for Delta in {1, 2}.

This isolates the computational question raised by the reviewer. Random
weights are sufficient because candidate construction and batched-forward
cost depend on architecture and active-set size, not model accuracy.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import time

import numpy as np
import torch

from bnb_ibp_pair import _build_pixel_offset_options, _enumerate_gpu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--active-pixels", type=int, nargs="+", default=[64, 128, 256, 512, 784])
    p.add_argument("--widths", type=int, nargs="+", default=[100, 200])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("bench_results/cleb_scaling.json"))
    return p.parse_args()


def count_candidates(img, pixels, delta, w1, time_steps):
    total_options = 0
    total_pairs = 0
    total_trajectories = 0
    split_rows = []
    for rem_neg in range(delta + 1):
        rem_pos = delta - rem_neg
        p_idx, cp, cn, _ = _build_pixel_offset_options(
            img, pixels, rem_neg, rem_pos, w1, time_steps
        )
        n_opt = len(p_idx)
        n_pairs = 0
        if delta == 2 and n_opt:
            cp_pair = cp[:, None] + cp[None, :]
            cn_pair = cn[:, None] + cn[None, :]
            feasible = (
                (p_idx[:, None] != p_idx[None, :])
                & np.triu(np.ones((n_opt, n_opt), dtype=bool), k=1)
                & (cp_pair <= rem_pos)
                & (cn_pair <= rem_neg)
            )
            n_pairs = int(feasible.sum())
        split_rows.append(
            {
                "rem_neg": rem_neg,
                "rem_pos": rem_pos,
                "options": n_opt,
                "pairs": n_pairs,
                "trajectories_including_baseline": 1 + n_opt + n_pairs,
            }
        )
        total_options += n_opt
        total_pairs += n_pairs
        total_trajectories += 1 + n_opt + n_pairs
    return total_options, total_pairs, total_trajectories, split_rows


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    rng = np.random.default_rng(args.seed)
    time_steps = 5
    threshold = 100.0
    img = rng.integers(1, time_steps - 1, size=(28, 28), dtype=np.int64)
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)], dtype=int)
    rows = []

    for width in args.widths:
        w1 = rng.normal(0, 5, size=(width, 28, 28))
        w2 = rng.normal(0, 5, size=(width, width))
        w_out = rng.normal(0, 5, size=(10, width))
        for active in args.active_pixels:
            pixels = all_pixels[:active]
            for delta in (1, 2):
                option_count, pair_count, trajectory_count, splits = count_candidates(
                    img, pixels, delta, w1, time_steps
                )
                elapsed = []
                peak_memory = []
                for _ in range(args.repeats):
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    tic = time.perf_counter()
                    for rem_neg in range(delta + 1):
                        _enumerate_gpu(
                            img,
                            pixels,
                            rem_neg,
                            delta - rem_neg,
                            w1,
                            w2,
                            w_out,
                            threshold,
                            time_steps,
                            10,
                            device=args.device,
                        )
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                        peak_memory.append(torch.cuda.max_memory_allocated())
                    elapsed.append(time.perf_counter() - tic)
                row = {
                    "width": width,
                    "active_pixels": active,
                    "delta": delta,
                    "option_count_across_splits": option_count,
                    "pair_count_across_splits": pair_count,
                    "trajectory_count_across_splits": trajectory_count,
                    "time_s": elapsed,
                    "time_median_s": float(np.median(elapsed)),
                    "peak_memory_bytes": peak_memory,
                    "peak_memory_max_bytes": max(peak_memory) if peak_memory else None,
                    "splits": splits,
                }
                rows.append(row)
                print(
                    f"width={width:>3} P={active:>3} Delta={delta} "
                    f"trajectories={trajectory_count:>9} "
                    f"median={row['time_median_s']:.4f}s "
                    f"peak={row['peak_memory_max_bytes'] / 2**20:.1f}MiB",
                    flush=True,
                )
                args.output.parent.mkdir(exist_ok=True)
                args.output.write_text(
                    json.dumps({"config": vars(args), "rows": rows}, indent=2, default=str)
                    + "\n"
                )

    # Direction-agnostic upper-bound illustration: at most two directions per
    # pixel. This is not timed because the implementation intentionally does
    # not enumerate Delta=3.
    projected = []
    for active in args.active_pixels:
        projected.append(
            {
                "active_pixels": active,
                "delta3_two_direction_term": 8 * comb(active, 3),
            }
        )
    payload = {"config": vars(args), "rows": rows, "delta3_projection": projected}
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

