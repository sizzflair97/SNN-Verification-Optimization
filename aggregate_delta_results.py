#!/usr/bin/env python3
"""Aggregate delta_scaling_benchmark output into a markdown report."""
import json, sys
from pathlib import Path
from statistics import mean, median
from collections import defaultdict


def summarize(json_path: Path):
    data = json.loads(json_path.read_text())
    cfg = data["config"]
    results = data["results"]

    method_keys = [m["key"] for m in cfg["methods"]]
    method_labels = {m["key"]: m["label"] for m in cfg["methods"]}
    deltas = cfg["delta_list"]
    n_h = cfg["n_hidden"]

    grouped = defaultdict(list)
    for r in results:
        grouped[(r["method"], r.get("delta"))].append(r)

    lines = []
    lines.append(f"# Delta-Scaling Benchmark\n")
    lines.append(
        f"**Config**: n_hidden={n_h}, num_steps={cfg['num_steps']}, deltas={deltas}, "
        f"samples/config={cfg['samples']}, per-sample timeout={cfg['per_sample_timeout_s']}s, seed={cfg['seed']}\n"
    )
    lines.append(f"**Run ID**: `{cfg['run_id']}`\n\n")

    # === Solver time table ===
    lines.append("## Solver time (seconds) — primary metric\n")
    lines.append("Format: `mean (median) [completed/total]`; `T.O.` = timeout; `-` = skipped.\n\n")
    header = "| Method | " + " | ".join(f"δ={d}" for d in deltas) + " |"
    sep = "|" + "---|" * (len(deltas) + 1)
    lines.append(header)
    lines.append(sep)
    for mk in method_keys:
        row = [method_labels[mk]]
        for d in deltas:
            rs = grouped.get((mk, d), [])
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
                if n_to: suffix += f" +{n_to}T.O."
                if n_skip: suffix += f" +{n_skip}skip"
                row.append(f"{m:.3f} ({md:.3f}) [{len(valid)}/{n_total}]{suffix}")
            elif n_to == n_total:
                row.append(f"T.O. [0/{n_total}]")
            elif n_skip == n_total:
                row.append(f"— (skipped)")
            else:
                row.append(f"? [{len(valid)}/{n_total}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Not-robust rate across delta ===
    lines.append("## Not-robust rate (per delta)\n")
    lines.append("Fraction of completed samples found to be adversarial.\n\n")
    lines.append("| Method | " + " | ".join(f"δ={d}" for d in deltas) + " |")
    lines.append(sep)
    for mk in method_keys:
        row = [method_labels[mk]]
        for d in deltas:
            rs = grouped.get((mk, d), [])
            rob = sum(1 for r in rs if r.get("verdict") == "robust")
            nrob = sum(1 for r in rs if r.get("verdict") == "not_robust")
            total = rob + nrob
            if total == 0:
                row.append("—")
            else:
                row.append(f"{nrob}/{total} ({100*nrob/total:.0f}%)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Consistency check (find BnB filter unsoundness vs exhaustive) ===
    lines.append("## Verdict consistency vs Exhaustive (oracle)\n")
    lines.append("If exhaustive says NR but a BnB variant says R → **unsound** (missed adversarial). Vice versa rare.\n\n")
    lines.append("| Method | " + " | ".join(f"δ={d}" for d in deltas) + " |")
    lines.append(sep)

    # Build per-(delta, sample) exhaustive verdict
    exh_by = {}
    for r in grouped.get(("exhaustive", None), []) + sum([grouped.get(("exhaustive", d), []) for d in deltas], []):
        exh_by[(r.get("delta"), r["sample_no"])] = r.get("verdict")

    for mk in method_keys:
        if mk == "exhaustive":
            continue
        row = [method_labels[mk]]
        for d in deltas:
            rs = grouped.get((mk, d), [])
            mism = 0
            compared = 0
            for r in rs:
                ev = exh_by.get((d, r["sample_no"]))
                mv = r.get("verdict")
                if ev in ("robust", "not_robust") and mv in ("robust", "not_robust"):
                    compared += 1
                    if ev != mv:
                        mism += 1
            if compared == 0:
                row.append("—")
            else:
                marker = " ⚠" if mism else ""
                row.append(f"{mism}/{compared}{marker}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Per-sample verdict table per delta ===
    lines.append("## Per-sample verdicts per delta\n")
    for d in deltas:
        lines.append(f"### δ={d}\n")
        samples = sorted({r["sample_no"] for r in results if r.get("delta") == d})
        header2 = "| Sample | " + " | ".join(mk for mk in method_keys) + " |"
        sep2 = "|" + "---|" * (len(method_keys) + 1)
        lines.append(header2)
        lines.append(sep2)
        for s in samples:
            row = [str(s)]
            for mk in method_keys:
                match = [r for r in grouped.get((mk, d), []) if r["sample_no"] == s]
                v = match[0].get("verdict", "?") if match else "-"
                short = {"robust":"R","not_robust":"NR","timeout":"T","skipped":"S","skipped_early":"s","unknown":"?"}.get(v, v)
                row.append(short)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).parent
    bench_dir = root / "bench_results"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        files = sorted(bench_dir.glob("delta_scaling_*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            print("No delta_scaling_*.json found", file=sys.stderr); sys.exit(1)
        target = files[-1]
    out = root / "DELTA_SCALING_REPORT.md"
    out.write_text(summarize(target))
    print(f"Report: {out}")
    print(f"Source: {target}")
