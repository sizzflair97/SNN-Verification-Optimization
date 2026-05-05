#!/usr/bin/env python3
"""
Generate PSM scale-up report.

Merges the new PSM scale bench data (bench_results/psm_scale_*.json) with
existing baselines from:
  - bench_results/delta_scaling_0416213348.json  (nh=200, δ=1..4)
  - bench_results/nh500_delta2_ibp.json          (nh=500, δ=2)

Reports, per (nh, δ) cell, a paired bnb_X vs bnb_X_psm comparison — wall time,
verdict match, PSM cache hit rate, and state cache hit rate (extracted from
per-sample logs if available).
"""
import json
import re
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).parent.resolve()
RES = ROOT / "bench_results"
LOG = ROOT / "log"
OUT = ROOT / "PSM_SCALE_REPORT.md"

BASELINE_FILES = [
    RES / "delta_scaling_0416213348.json",
    RES / "nh500_delta2_ibp.json",
]

CELLS = [
    (200, 3),
    (500, 2),
]

PAIRS = [
    ("bnb_legacy", "bnb_legacy_psm", "legacy margin"),
    ("bnb_nofilter", "bnb_nofilter_psm", "no filter"),
]

TIME_PAT = re.compile(r"Checking done in time (\d+\.?\d*)")
PSM_PAT = re.compile(r"PSM cache:\s*(\d+)\s*checks,\s*(\d+)\s*hits")
STATE_PAT = re.compile(r"State cache:\s*(\d+)\s*checks,\s*(\d+)\s*hits")


def load(p):
    return json.loads(Path(p).read_text())


def latest_log_for(prefix_fragment):
    cands = sorted(LOG.glob(f"*{prefix_fragment}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def log_stats_for_row(r):
    if r.get("psm_checks") is not None:
        return {
            "psm_checks": r.get("psm_checks"),
            "psm_hits": r.get("psm_hits"),
            "state_checks": r.get("state_checks"),
            "state_hits": r.get("state_hits"),
        }
    lf = r.get("log_file")
    if lf:
        p = LOG / lf
    else:
        p = latest_log_for(f"{r.get('method')}_nh{r.get('n_hidden')}")
    if not p or not Path(p).exists():
        return {"psm_checks": None, "psm_hits": None, "state_checks": None, "state_hits": None}
    try:
        text = Path(p).read_text(errors="ignore")
    except FileNotFoundError:
        return {"psm_checks": None, "psm_hits": None, "state_checks": None, "state_hits": None}
    out = {"psm_checks": None, "psm_hits": None, "state_checks": None, "state_hits": None}
    m = PSM_PAT.search(text)
    if m:
        out["psm_checks"], out["psm_hits"] = int(m.group(1)), int(m.group(2))
    m = STATE_PAT.search(text)
    if m:
        out["state_checks"], out["state_hits"] = int(m.group(1)), int(m.group(2))
    return out


def collect_rows():
    """Prefer fresh rows from psm_scale_*.json (same code version as PSM runs).
    Fall back to stale external baselines only for cells not covered by the fresh data.
    """
    fresh_rows = []
    psm_scale_files = sorted(RES.glob("psm_scale_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in psm_scale_files:
        d = load(f)
        for r in d.get("results", []):
            fresh_rows.append(dict(r))

    fresh_keys = {(r.get("method"), r.get("n_hidden"), r.get("delta"), r.get("sample_no"))
                  for r in fresh_rows}

    ext_rows = []
    for f in BASELINE_FILES:
        if not f.exists():
            continue
        d = load(f)
        cfg = d.get("config", {})
        default_delta = cfg.get("delta")
        for r in d.get("results", []):
            rr = dict(r)
            if rr.get("delta") is None:
                rr["delta"] = default_delta
            key = (rr.get("method"), rr.get("n_hidden"), rr.get("delta"), rr.get("sample_no"))
            if key in fresh_keys:
                continue  # prefer fresh
            ext_rows.append(rr)
    return fresh_rows + ext_rows


def filter_rows(rows, method, nh, delta):
    return [r for r in rows if r.get("method") == method and r.get("n_hidden") == nh and r.get("delta") == delta]


def agg(rows, timeout_s=300):
    walls = []
    vcounts = {"robust": 0, "not_robust": 0, "timeout": 0, "skipped_early": 0, "unknown": 0, "skipped": 0}
    for r in rows:
        v = r.get("verdict", "unknown")
        vcounts[v] = vcounts.get(v, 0) + 1
        w = r.get("wall_time")
        if w is None and v in ("timeout", "skipped_early"):
            w = timeout_s
        if w is not None:
            walls.append(w)
    return {
        "n": len(rows),
        "mean": mean(walls) if walls else None,
        "median": median(walls) if walls else None,
        "walls_n": len(walls),
        **vcounts,
    }


def paired(rows_base, rows_psm, timeout_s=300):
    """Join by sample_no. Returns list of dicts with wall times + speedup + log stats."""
    by_b = {r["sample_no"]: r for r in rows_base}
    by_p = {r["sample_no"]: r for r in rows_psm}
    out = []
    for s in sorted(set(by_b) & set(by_p)):
        b = by_b[s]
        p = by_p[s]
        wb = b.get("wall_time")
        wp = p.get("wall_time")
        # Treat timeout/skipped as 300s for speedup conservatism
        if b.get("verdict") in ("timeout", "skipped_early") and wb is None:
            wb = timeout_s
        if p.get("verdict") in ("timeout", "skipped_early") and wp is None:
            wp = timeout_s
        sp = (wb / wp) if (wb and wp) else None
        stats_p = log_stats_for_row(p)
        out.append({
            "sample_no": s,
            "verdict_base": b.get("verdict"),
            "verdict_psm": p.get("verdict"),
            "wall_base": wb,
            "wall_psm": wp,
            "speedup": sp,
            "psm_checks": stats_p.get("psm_checks"),
            "psm_hits": stats_p.get("psm_hits"),
            "state_checks": stats_p.get("state_checks"),
            "state_hits": stats_p.get("state_hits"),
        })
    return out


def fmt(x, spec=".2f"):
    if x is None:
        return "-"
    try:
        return format(x, spec)
    except Exception:
        return str(x)


def main():
    rows = collect_rows()
    lines = []
    lines.append("# PSM Scale-Up Report")
    lines.append("")
    lines.append("**Goal:** Test whether PSM's independent contribution grows at larger δ and n_hidden.")
    lines.append("")
    lines.append("**Baselines reused** (same 10 samples, seed=42):")
    lines.append("- `delta_scaling_0416213348.json` — nh=200, δ∈{1..4}, 300s timeout")
    lines.append("- `nh500_delta2_ibp.json` — nh=500, δ=2, 300s timeout")
    lines.append("")
    lines.append("**New runs:** `psm_scale_*.json` — bnb_legacy_psm, bnb_nofilter_psm "
                 "at each target cell; plus bnb_nofilter at (nh=500, δ=2).")
    lines.append("")

    for nh, delta in CELLS:
        lines.append(f"## n_hidden = {nh}, δ = {delta}")
        lines.append("")

        # Aggregate table across all methods at this cell
        lines.append("### Aggregate timings (wall seconds; timeouts counted as 300s)")
        lines.append("")
        lines.append("| method | n | mean | median | robust | not_robust | timeout | skipped |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        methods_here = []
        for base, psm, _ in PAIRS:
            methods_here += [base, psm]
        for m in methods_here:
            rs = filter_rows(rows, m, nh, delta)
            a = agg(rs)
            lines.append(f"| {m} | {a['n']} | {fmt(a['mean'])} | {fmt(a['median'])} | "
                         f"{a['robust']} | {a['not_robust']} | {a['timeout']} | "
                         f"{a.get('skipped_early', 0) + a.get('skipped', 0)} |")
        lines.append("")

        # Paired speedup per filter
        for base, psm, label in PAIRS:
            lines.append(f"### Paired: {label} — `{base}` vs `{psm}`")
            lines.append("")
            pr = paired(filter_rows(rows, base, nh, delta), filter_rows(rows, psm, nh, delta))
            if not pr:
                lines.append("_No paired data available._")
                lines.append("")
                continue
            lines.append("| sample | verdict (base/PSM) | wall base | wall PSM | speedup | PSM hits/checks | state hits/checks |")
            lines.append("|---:|:--|---:|---:|---:|---:|---:|")
            sps = []
            psm_hits_total = psm_checks_total = 0
            state_hits_total = state_checks_total = 0
            for p in pr:
                sp_str = f"{p['speedup']:.2f}×" if p['speedup'] is not None else "-"
                if p['speedup'] is not None:
                    sps.append(p['speedup'])
                if p['psm_checks'] is not None:
                    psm_checks_total += p['psm_checks']
                    psm_hits_total += p['psm_hits'] or 0
                if p['state_checks'] is not None:
                    state_checks_total += p['state_checks']
                    state_hits_total += p['state_hits'] or 0
                psm_str = (f"{p['psm_hits']}/{p['psm_checks']}"
                           if p['psm_checks'] is not None else "-")
                state_str = (f"{p['state_hits']}/{p['state_checks']}"
                             if p['state_checks'] is not None else "-")
                lines.append(f"| {p['sample_no']} | {p['verdict_base']} / {p['verdict_psm']} | "
                             f"{fmt(p['wall_base'])} | {fmt(p['wall_psm'])} | {sp_str} | "
                             f"{psm_str} | {state_str} |")
            if sps:
                gm = 1.0
                for s in sps:
                    gm *= s
                gm = gm ** (1.0 / len(sps))
                lines.append("")
                lines.append(f"*n={len(sps)} paired. Geometric-mean speedup: **{gm:.2f}×**; "
                             f"mean: {mean(sps):.2f}×; median: {median(sps):.2f}×; "
                             f"min: {min(sps):.2f}×; max: {max(sps):.2f}×.*")
                if psm_checks_total > 0:
                    hr = 100.0 * psm_hits_total / psm_checks_total
                    lines.append(f"*PSM cache aggregate: {psm_hits_total}/{psm_checks_total} = {hr:.2f}% hit rate.*")
                if state_checks_total > 0:
                    hr = 100.0 * state_hits_total / state_checks_total
                    lines.append(f"*State cache aggregate: {state_hits_total}/{state_checks_total} = {hr:.2f}% hit rate.*")
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **If PSM/state cache hit rate stays ~0%** across higher δ and nh, the prefix-set")
    lines.append("  matching mechanism is not firing in the current fixed-ordering DFS. Any wall-time")
    lines.append("  movement is then incidental (bookkeeping overhead) rather than structural reuse.")
    lines.append("- **To make PSM appeal stronger** one of the following is likely needed:")
    lines.append("  (a) change the DFS to explore multiple orderings so the same prefix-state is")
    lines.append("  reached via different paths; (b) key the cache on a coarser invariant (e.g.,")
    lines.append("  set of perturbed positions, independent of apply order); (c) report *node*")
    lines.append("  count reduction where PSM pre-emptively prunes subtrees the filter would have")
    lines.append("  otherwise explored.")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
