#!/usr/bin/env python3
"""BC-IBP verifier scaling sweep over Delta=1,...,5 for reviewer response."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from reviewer_incidence_benchmark import run_one, select_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=int, nargs="+", default=[200, 500])
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("bench_results/reviewer_delta_scaling.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_delta_scaling"
    selected_by_width = {}
    jobs = []
    for width in args.widths:
        selected = select_examples(args.n_samples, width, args.seed, args.split)
        selected_by_width[str(width)] = selected
        for delta in args.deltas:
            for example in selected:
                jobs.append((width, delta, example))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}
        for width, delta, example in jobs:
            worker_args = argparse.Namespace(
                n_hidden=width,
                seed=args.seed,
                timeout=args.timeout,
                split=args.split,
            )
            future = pool.submit(run_one, example, delta, worker_args, run_id)
            future_map[future] = (width, delta, example["index"])
        for done, future in enumerate(as_completed(future_map), start=1):
            width, delta, idx = future_map[future]
            row = future.result()
            row["n_hidden"] = width
            rows.append(row)
            print(
                f"[{done}/{len(jobs)}] width={width} Delta={delta} idx={idx} "
                f"{row['verdict']} solver={row['solver_time_s']} "
                f"wall={row['wall_time_s']:.2f}s",
                flush=True,
            )
            args.output.write_text(
                json.dumps(
                    {
                        "config": vars(args) | {"run_id": run_id},
                        "selected_by_width": selected_by_width,
                        "rows": rows,
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
