#!/usr/bin/env python3
"""Aggregate realistic_large_benchmark.py output into a markdown comparison report."""
import json
import sys
from pathlib import Path
from statistics import mean, median
from collections import defaultdict


def load_latest(bench_dir: Path):
    files = sorted(bench_dir.glob("realistic_bench_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("no benchmark results found")
    return files[-1]


def summarize(json_path: Path):
    data = json.loads(json_path.read_text())
    cfg = data["config"]
    results = data["results"]

    methods_order = [m["key"] for m in cfg["methods"]]
    method_labels = {m["key"]: m["label"] for m in cfg["methods"]}
    n_hidden_list = cfg["n_hidden_list"]

    # group: (method, n_hidden) -> list of results
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["method"], r["n_hidden"])].append(r)

    lines = []
    lines.append(f"# Realistic Large-Network Benchmark Results\n")
    lines.append(
        f"**Config**: num_steps={cfg['num_steps']}, delta={cfg['delta']}, "
        f"samples/config={cfg['samples']}, per-sample timeout={cfg['per_sample_timeout_s']}s, seed={cfg['seed']}\n"
    )
    lines.append(f"**Run ID**: `{cfg['run_id']}`\n")
    lines.append(f"**Source**: `{json_path.name}`\n\n")

    # === Solver-time table ===
    lines.append("## Solver time (seconds) — primary metric\n")
    lines.append("Reports solver time only (excludes Python startup + model load).\n")
    lines.append("Format: `mean (median) [completed/total]`; `T.O.` = timeout; `-` = skipped.\n\n")

    header = "| Method | " + " | ".join(f"n_h={nh}" for nh in n_hidden_list) + " |"
    sep = "|" + "---|" * (len(n_hidden_list) + 1)
    lines.append(header)
    lines.append(sep)
    for mk in methods_order:
        row = [method_labels[mk]]
        for nh in n_hidden_list:
            rs = grouped.get((mk, nh), [])
            if not rs:
                row.append("—")
                continue
            n_to = sum(1 for r in rs if r.get("verdict") == "timeout")
            n_skip = sum(1 for r in rs if r.get("verdict") in ("skipped", "skipped_early"))
            n_total = len(rs)
            valid = [r["solver_time"] for r in rs if isinstance(r.get("solver_time"), (int, float))]
            if valid:
                m, md = mean(valid), median(valid)
                suffix = ""
                if n_to > 0:
                    suffix += f" +{n_to}T.O."
                if n_skip > 0:
                    suffix += f" +{n_skip}skip"
                row.append(f"{m:.3f} ({md:.3f}) [{len(valid)}/{n_total}]{suffix}")
            else:
                if n_to == n_total:
                    row.append(f"T.O. [0/{n_total}]")
                elif n_skip == n_total:
                    row.append(f"— (skipped)")
                else:
                    row.append(f"? [{len(valid)}/{n_total}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Wall-time table ===
    lines.append("## Wall time (seconds) — end-to-end including subprocess startup\n")
    lines.append(header)
    lines.append(sep)
    for mk in methods_order:
        row = [method_labels[mk]]
        for nh in n_hidden_list:
            rs = grouped.get((mk, nh), [])
            if not rs:
                row.append("—")
                continue
            valid = [r["wall_time"] for r in rs if isinstance(r.get("wall_time"), (int, float))]
            if valid:
                row.append(f"{mean(valid):.2f} ({median(valid):.2f})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Verdict consistency ===
    lines.append("## Verdict consistency check\n")
    lines.append("Per-sample verdicts across methods. Discrepancies indicate unsoundness.\n\n")
    # group by (n_h, sample_no) -> {method: verdict}
    per_sample = defaultdict(dict)
    for r in results:
        key = (r["n_hidden"], r["sample_no"])
        per_sample[key][r["method"]] = r.get("verdict", "?")

    for nh in n_hidden_list:
        lines.append(f"### n_h={nh}\n")
        header2 = "| Sample | " + " | ".join(methods_order) + " |"
        sep2 = "|" + "---|" * (len(methods_order) + 1)
        lines.append(header2)
        lines.append(sep2)
        samples_for_nh = sorted({s for (h, s) in per_sample if h == nh})
        mismatches = 0
        for s in samples_for_nh:
            verdicts = per_sample[(nh, s)]
            resolvable = {v for v in verdicts.values() if v in ("robust", "not_robust")}
            mismatch_flag = " ⚠" if len(resolvable) > 1 else ""
            if len(resolvable) > 1:
                mismatches += 1
            row = [str(s)]
            for mk in methods_order:
                v = verdicts.get(mk, "-")
                short = {"robust": "R", "not_robust": "NR", "timeout": "T", "skipped": "S", "skipped_early": "s", "unknown": "?"}.get(v, v)
                row.append(short)
            row[-1] = row[-1] + mismatch_flag
            lines.append("| " + " | ".join(row) + " |")
        lines.append(f"\n**Mismatches in n_h={nh}**: {mismatches}\n")

    # === Speedup table (vs exhaustive) ===
    lines.append("## Speedup vs Exhaustive DFS (solver time, completed samples only)\n")
    lines.append(header)
    lines.append(sep)
    for mk in methods_order:
        if mk == "exhaustive":
            continue
        row = [method_labels[mk]]
        for nh in n_hidden_list:
            rs_m = grouped.get((mk, nh), [])
            rs_e = grouped.get(("exhaustive", nh), [])
            if not rs_m or not rs_e:
                row.append("—")
                continue
            valid_m = [r["solver_time"] for r in rs_m if isinstance(r.get("solver_time"), (int, float))]
            valid_e = [r["solver_time"] for r in rs_e if isinstance(r.get("solver_time"), (int, float))]
            if not valid_m or not valid_e:
                row.append("—")
                continue
            sp = mean(valid_e) / mean(valid_m)
            row.append(f"{sp:.2f}×")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).parent
    bench_dir = root / "bench_results"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = load_latest(bench_dir)
    out = root / "REALISTIC_LARGE_BENCHMARK_REPORT.md"
    out.write_text(summarize(target))
    print(f"Report written to: {out}")
    print(f"Source: {target}")
