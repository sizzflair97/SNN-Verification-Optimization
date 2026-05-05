#!/usr/bin/env python3
"""Generate standalone PSM ablation report.

Compares each BnB pre-filter mode with/without PSM on the *same* 10 samples
at n_h=200, delta=2, num_steps=5. Isolates PSM's independent contribution.

Sources:
  - bench_results/delta_scaling_0416213348.json
      (bnb_legacy, bnb_nofilter, bnb_efficient, bnb_psm=efficient+PSM, exhaustive, milp)
  - bench_results/realistic_bench_0417172351.json
      (bnb_legacy_psm, bnb_nofilter_psm)
"""
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).parent.resolve()
MASTER = ROOT / "bench_results/delta_scaling_0416213348.json"
PSM_ABL = ROOT / "bench_results/realistic_bench_0417172351.json"
OUT = ROOT / "PSM_STANDALONE_REPORT.md"

TARGET_NH = 200
TARGET_DELTA = 2


def load(path):
    return json.loads(Path(path).read_text())


def filter_rows(results, method, nh, delta):
    out = []
    for r in results:
        if r.get("method") != method:
            continue
        if r.get("n_hidden") != nh:
            continue
        if delta is not None and r.get("delta") is not None and r["delta"] != delta:
            continue
        out.append(r)
    return out


def by_sample(rows):
    return {r["sample_no"]: r for r in rows}


def agg(rows, timeout_fallback=None):
    """Return (mean, median, robust, not_robust, timeout, other)."""
    walls = []
    robust = nr = to = other = 0
    for r in rows:
        v = r.get("verdict")
        if v == "robust":
            robust += 1
        elif v == "not_robust":
            nr += 1
        elif v == "timeout":
            to += 1
        else:
            other += 1
        w = r.get("wall_time")
        if w is None and v == "timeout" and timeout_fallback is not None:
            w = timeout_fallback
        if w is not None:
            walls.append(w)
    if not walls:
        return None, None, robust, nr, to, other, 0
    return mean(walls), median(walls), robust, nr, to, other, len(walls)


def pair_speedup(no_psm_rows, psm_rows):
    """Compute per-sample speedup (no_psm.wall / psm.wall). Sort by sample_no."""
    no_map = by_sample(no_psm_rows)
    yes_map = by_sample(psm_rows)
    common = sorted(set(no_map) & set(yes_map))
    out = []
    for s in common:
        n = no_map[s]
        p = yes_map[s]
        out.append({
            "sample_no": s,
            "wall_no_psm": n.get("wall_time"),
            "wall_psm": p.get("wall_time"),
            "verdict_no_psm": n.get("verdict"),
            "verdict_psm": p.get("verdict"),
            "speedup": (n["wall_time"] / p["wall_time"]) if (n.get("wall_time") and p.get("wall_time")) else None,
        })
    return out


def main():
    master = load(MASTER)
    psm_ab = load(PSM_ABL)

    rows_leg = filter_rows(master["results"], "bnb_legacy", TARGET_NH, TARGET_DELTA)
    rows_nofilt = filter_rows(master["results"], "bnb_nofilter", TARGET_NH, TARGET_DELTA)
    rows_eff = filter_rows(master["results"], "bnb_efficient", TARGET_NH, TARGET_DELTA)
    rows_eff_psm = filter_rows(master["results"], "bnb_psm", TARGET_NH, TARGET_DELTA)

    # PSM ablation file has no 'delta' field per row; results are all at delta=2.
    rows_leg_psm = [r for r in psm_ab["results"] if r.get("method") == "bnb_legacy_psm"]
    rows_nofilt_psm = [r for r in psm_ab["results"] if r.get("method") == "bnb_nofilter_psm"]

    pairs = [
        ("no filter",   "bnb_nofilter",     "bnb_nofilter_psm",     rows_nofilt,  rows_nofilt_psm),
        ("legacy",      "bnb_legacy",       "bnb_legacy_psm",       rows_leg,     rows_leg_psm),
        ("efficient",   "bnb_efficient",    "bnb_psm",              rows_eff,     rows_eff_psm),
    ]

    lines = []
    lines.append("# PSM Standalone Contribution Report")
    lines.append("")
    lines.append(f"**Setting:** MNIST, T={master['config']['num_steps']}, "
                 f"n_hidden={TARGET_NH}, δ={TARGET_DELTA}, 10 samples (seed=42), "
                 f"per-sample timeout={master['config']['per_sample_timeout_s']}s.")
    lines.append("")
    lines.append("**Question:** What does PSM contribute *independently of* the pre-filter "
                 "(legacy / efficient / nofilter)? Each pair holds the filter fixed and toggles PSM only.")
    lines.append("")
    lines.append("## Note on filter soundness")
    lines.append("")
    lines.append("- `efficient` filter is **unsound for δ ≥ 2** (may miss adversarials) — included for timing only, not robustness claims.")
    lines.append("- `legacy` voltage-margin filter is sound.")
    lines.append("- `no filter` is the pure BnB baseline (sound).")
    lines.append("")
    lines.append("## Aggregate timings (wall seconds)")
    lines.append("")
    lines.append("| Filter | PSM | mean | median | robust | not_robust | timeout |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for filt_name, base_key, psm_key, base_rows, psm_rows in pairs:
        for label, rows in (("off", base_rows), ("on", psm_rows)):
            m, med, rb, nr, to, oth, n = agg(rows)
            m_str = f"{m:.2f}" if m is not None else "-"
            med_str = f"{med:.2f}" if med is not None else "-"
            lines.append(f"| {filt_name} | {label} | {m_str} | {med_str} | {rb} | {nr} | {to} |")
    lines.append("")

    lines.append("## Per-pair speedup (no-PSM wall / PSM wall)")
    lines.append("")
    for filt_name, base_key, psm_key, base_rows, psm_rows in pairs:
        per = pair_speedup(base_rows, psm_rows)
        lines.append(f"### Filter = {filt_name}  ({base_key}  vs  {psm_key})")
        lines.append("")
        lines.append("| sample | verdict (no PSM / PSM) | wall no-PSM (s) | wall PSM (s) | speedup |")
        lines.append("|---:|:--|---:|---:|---:|")
        speedups = []
        for p in per:
            w1 = p["wall_no_psm"]; w2 = p["wall_psm"]; sp = p["speedup"]
            w1s = f"{w1:.2f}" if w1 is not None else "-"
            w2s = f"{w2:.2f}" if w2 is not None else "-"
            sps = f"{sp:.2f}×" if sp is not None else "-"
            if sp is not None:
                speedups.append(sp)
            lines.append(f"| {p['sample_no']} | {p['verdict_no_psm']} / {p['verdict_psm']} | {w1s} | {w2s} | {sps} |")
        if speedups:
            gm = 1.0
            for s in speedups:
                gm *= s
            gm = gm ** (1.0 / len(speedups))
            lines.append("")
            lines.append(f"*n={len(speedups)} paired. Geometric-mean speedup: **{gm:.2f}×**; "
                         f"mean: {mean(speedups):.2f}×; median: {median(speedups):.2f}×; "
                         f"min: {min(speedups):.2f}×; max: {max(speedups):.2f}×.*")
        lines.append("")

    lines.append("## Verdict consistency check")
    lines.append("")
    lines.append("PSM should not change any verdict (both soundness and completeness preserved on sound filters).")
    lines.append("")
    lines.append("| Filter | matched verdicts | disagreements |")
    lines.append("|---|---:|---:|")
    for filt_name, base_key, psm_key, base_rows, psm_rows in pairs:
        per = pair_speedup(base_rows, psm_rows)
        match = sum(1 for p in per if p["verdict_no_psm"] == p["verdict_psm"])
        diff = len(per) - match
        lines.append(f"| {filt_name} | {match}/{len(per)} | {diff} |")
    lines.append("")

    lines.append("## Bottom line")
    lines.append("")
    lines.append("PSM (Prefix-Set Matching) is an optimization layered on top of any pre-filter. ")
    lines.append("The three pairs above isolate its contribution:")
    lines.append("")
    lines.append("- vs **no filter**: measures PSM's raw pruning power on the full pixel set.")
    lines.append("- vs **legacy**: PSM's *additive* benefit beyond sound voltage-margin pruning.")
    lines.append("- vs **efficient**: PSM stacked on sensitivity-based ordering (the shipped default).")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
