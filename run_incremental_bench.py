#!/usr/bin/env python3
"""
Incremental benchmark runner: re-uses previously recorded (method, n_h, delta, sample)
results from earlier JSON files, and only runs the (method, n_h, delta, sample) tuples
that aren't already covered.

Usage:
  python run_incremental_bench.py --methods bnb_legacy_ibp \
      --n-hidden 200 --deltas 1 2 3 4 \
      --reuse bench_results/delta_scaling_0416213348.json \
      --output bench_results/delta_scaling_with_ibp.json
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import realistic_large_benchmark as rlb


def load_existing(path: Path):
    """Return (config, results_list) from an existing JSON, or (None, [])."""
    if not path or not path.exists():
        return None, []
    data = json.loads(path.read_text())
    results = data.get("results", [])
    # Normalize: every record must have 'delta' (fill from top-level config if missing)
    top_delta = data.get("config", {}).get("delta")
    for r in results:
        if "delta" not in r and top_delta is not None:
            r["delta"] = top_delta
    return data.get("config"), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True, help="method keys to RUN (others loaded from --reuse)")
    ap.add_argument("--n-hidden", type=int, nargs="+", default=[200])
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--reuse", type=str, default=None, help="path to prior JSON to merge")
    ap.add_argument("--output", type=str, required=True, help="output JSON path")
    ap.add_argument("--timeout", type=int, default=300, help="per-sample timeout (s)")
    args = ap.parse_args()

    run_id = time.strftime("%m%d%H%M%S")

    # Configure rlb
    rlb.N_HIDDEN_LIST = list(args.n_hidden)
    rlb.NUM_SAMPLES = args.samples
    rlb.PER_SAMPLE_TIMEOUT_S = args.timeout

    # Index: only run the requested methods
    method_map = {m[0]: m for m in rlb.METHODS}
    missing = [m for m in args.methods if m not in method_map]
    if missing:
        raise SystemExit(f"Unknown methods: {missing}. Available: {list(method_map.keys())}")
    to_run_methods = [method_map[m] for m in args.methods]

    # Load prior results for re-use
    prior_cfg, prior_results = load_existing(Path(args.reuse)) if args.reuse else (None, [])
    # Filter prior to configs we're interested in
    seen = set()  # (method, n_h, delta, sample_no) already recorded
    filtered_prior = []
    for r in prior_results:
        key = (r.get("method"), r.get("n_hidden"), r.get("delta"), r.get("sample_no"))
        if r.get("n_hidden") in args.n_hidden and r.get("delta") in args.deltas:
            filtered_prior.append(r)
            seen.add(key)

    print(f"Re-using {len(filtered_prior)} prior results from {args.reuse or '(none)'}")

    new_results = []
    for d in args.deltas:
        print(f"\n=== DELTA = {d} ===")
        rlb.DELTA = d
        rlb.METHODS = to_run_methods
        # run_full_matrix iterates nh in N_HIDDEN_LIST and methods in METHODS and samples internally
        _out, results = rlb.run_full_matrix()
        for r in results:
            r["delta"] = d
            new_results.append(r)

    # Merge: prior + new (new takes precedence if key collision, though shouldn't happen)
    merged = {r_key(r): r for r in filtered_prior}
    for r in new_results:
        merged[r_key(r)] = r
    all_results = list(merged.values())

    # Infer method list from all_results for config (preserve order from prior + new)
    method_labels = {}
    if prior_cfg:
        for m in prior_cfg.get("methods", []):
            method_labels[m["key"]] = m["label"]
    for k, _, _, lbl in rlb.METHODS:
        method_labels[k] = lbl
    method_order = list(method_labels.keys())

    out_cfg = {
        "num_steps": rlb.NUM_STEPS,
        "delta_list": args.deltas,
        "samples": args.samples,
        "per_sample_timeout_s": args.timeout,
        "n_hidden": args.n_hidden[0] if len(args.n_hidden) == 1 else args.n_hidden,
        "methods": [{"key": k, "label": method_labels[k]} for k in method_order],
        "run_id": run_id,
        "seed": rlb.SEED,
        "reused_from": args.reuse,
    }
    Path(args.output).write_text(json.dumps({"config": out_cfg, "results": all_results}, indent=2))
    print(f"\nMerged results ({len(all_results)} records) → {args.output}")


def r_key(r):
    return (r.get("method"), r.get("n_hidden"), r.get("delta"), r.get("sample_no"))


if __name__ == "__main__":
    main()
