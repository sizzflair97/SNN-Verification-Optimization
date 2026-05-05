#!/usr/bin/env python3
"""Temporal sparsity vs BC-IBP speedup — cross-dataset correlation analysis.

For each (dataset, sample) pair in our existing benchmarks, compute:
  - Concentration C = max_t p(t)            # fraction of pixels at most-common time
  - Normalized entropy H = H(t) / log2(T)   # 0=all-same-time, 1=uniform
  - Active fraction = 1 - p(no-fire)        # fraction of pixels that spike

And pair with BC-IBP solver time / baseline solver time ⇒ speedup.

Outputs:
  - bench_results/sparsity_analysis.json : per-sample records
  - bench_results/sparsity_summary.md    : per-dataset correlation tables
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from utils.config import CFG
from utils.load import load_mnist, load_nmnist, load_dvs_gesture


def compute_sparsity(img: np.ndarray, num_steps: int) -> dict:
    """Given a TTFS-encoded image (pixel values in [0, T-1]), compute metrics."""
    flat = img.astype(int).ravel()
    n = flat.size
    counts = np.bincount(flat, minlength=num_steps)
    probs = counts / n
    # Concentration = max over t
    conc = float(probs.max())
    # Shannon entropy (log base 2), normalized by log2(T)
    nonzero = probs[probs > 0]
    h = float(-np.sum(nonzero * np.log2(nonzero)))
    h_norm = h / math.log2(num_steps) if num_steps > 1 else 0.0
    # Active fraction: fraction of pixels NOT at T-1 (no-fire sentinel)
    active = float(1.0 - probs[-1])
    return {
        "concentration": conc,
        "entropy_norm": h_norm,
        "active_fraction": active,
        "n_pixels": int(n),
    }


def parse_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _extract_mnist_beta_bench(path: Path) -> list[dict]:
    """beta_bench_*.json: list of records with method, sample, solver_time, verdict."""
    raw = parse_json(path)
    out = []
    # Group by sample; pair bc_ibp vs baseline (we don't have raw legacy timing here
    # — but the bench's bc_ibp IS the BC-IBP time. We need the legacy baseline separately.)
    # For MNIST n_h=500 δ=2 we use IBP_NH500_REPORT legacy values (hard-coded below).
    for r in raw:
        if r["method"] != "bc_ibp":
            continue
        out.append({
            "dataset": "MNIST",
            "config": "n_h=500, δ=2",
            "sample": r["sample"],
            "bcibp_time": r.get("solver_time"),
            "verdict": r.get("verdict"),
        })
    return out


# Legacy solver times at n_h=500 δ=2 (from bench_results/nh500_delta2_ibp.json).
_MNIST_NH500_LEGACY = {
    6717: 0.05, 41905: 0.24, 14628: 122.27, 48598: 116.74, 1639: 115.22,
    48265: 120.39, 7296: 115.62, 18024: 109.78, 16049: 113.90, 9144: 125.44,
}


def _extract_nmnist_bench(path: Path) -> list[dict]:
    raw = parse_json(path)["results"]
    out = []
    # Group by (delta, sample)
    by_pair = defaultdict(dict)
    for r in raw:
        by_pair[(r["delta"], r["sample_no"])][r["method"]] = r
    for (delta, sno), meth_map in by_pair.items():
        bc = meth_map.get("bnb_legacy_ibp_coupled")
        lg = meth_map.get("bnb_legacy")
        if not bc:
            continue
        bcibp_t = bc.get("solver_time")
        lg_t = lg.get("solver_time") if lg else None
        if lg and (lg.get("timed_out") or lg.get("verdict") == "timeout"):
            lg_t = None  # mark as timeout below
        out.append({
            "dataset": "N-MNIST",
            "config": f"n_h=100, δ={delta}",
            "sample": sno,
            "bcibp_time": bcibp_t,
            "legacy_time": lg_t,
            "legacy_timeout": lg is not None and (lg.get("timed_out") or lg.get("verdict") == "timeout"),
            "verdict": bc.get("verdict"),
            "delta": delta,
        })
    return out


def _extract_dvs_bench(path: Path) -> list[dict]:
    raw = parse_json(path)["results"]
    out = []
    by_pair = defaultdict(dict)
    for r in raw:
        by_pair[(r["delta"], r["sample_no"])][r["method"]] = r
    for (delta, sno), meth_map in by_pair.items():
        bc = meth_map.get("bnb_legacy_ibp_coupled")
        lg = meth_map.get("bnb_legacy")
        if not bc:
            continue
        bcibp_t = bc.get("solver_time")
        lg_t = lg.get("solver_time") if lg else None
        lg_to = (lg is not None and (lg.get("timed_out") or lg.get("verdict") == "timeout"
                                      or lg.get("verdict") == "skipped_early"))
        if lg_to:
            lg_t = None
        out.append({
            "dataset": "DVS-Gesture",
            "config": f"n_h=100, δ={delta}",
            "sample": sno,
            "bcibp_time": bcibp_t,
            "legacy_time": lg_t,
            "legacy_timeout": lg_to,
            "verdict": bc.get("verdict"),
            "delta": delta,
        })
    return out


def _load_dataset_images(subtype: str, num_steps: int):
    """Return (images, loader) tuple — images is concatenated train+test for indexing
    by original position in the combined array."""
    if subtype == "mnist":
        cfg = CFG(log_name="ts", subtype="mnist", load_data_func=load_mnist,
                  seed=42, num_samples=1, deltas=(1,),
                  n_layer_neurons=(784, 1, 10), layer_shapes=((28, 28), (1, 1), (10, 1)),
                  num_steps=num_steps)
        xr, _, _, _ = load_mnist(cfg)
        return xr
    if subtype == "nmnist":
        cfg = CFG(log_name="ts", subtype="nmnist", load_data_func=load_nmnist,
                  seed=42, num_samples=1, deltas=(1,),
                  n_layer_neurons=(2048, 1, 10), layer_shapes=((64, 32), (1, 1), (10, 1)),
                  num_steps=num_steps)
        xr, _, _, _ = load_nmnist(cfg)
        return xr
    if subtype == "dvs_gesture":
        cfg = CFG(log_name="ts", subtype="dvs_gesture", load_data_func=load_dvs_gesture,
                  seed=42, num_samples=1, deltas=(1,),
                  n_layer_neurons=(32768, 1, 11), layer_shapes=((256, 128), (1, 1), (11, 1)),
                  num_steps=num_steps)
        xr, _, _, _ = load_dvs_gesture(cfg)
        return xr
    raise ValueError(subtype)


def main() -> None:
    bench_dir = ROOT / "bench_results"
    records: list[dict] = []

    # --- MNIST n_h=500 δ=2 (10 samples from beta_bench) ---
    beta_bench = sorted(bench_dir.glob("beta_bench_*.json"))
    if beta_bench:
        mnist_imgs = _load_dataset_images("mnist", num_steps=5)
        for rec in _extract_mnist_beta_bench(beta_bench[-1]):
            sno = rec["sample"]
            rec["legacy_time"] = _MNIST_NH500_LEGACY.get(sno)
            rec["legacy_timeout"] = False
            sp = compute_sparsity(mnist_imgs[sno], num_steps=5)
            records.append({**rec, **sp})

    # --- N-MNIST ---
    nmnist_bench = sorted(bench_dir.glob("nmnist_bench_*.json"))
    if nmnist_bench:
        nmnist_imgs = _load_dataset_images("nmnist", num_steps=5)
        for rec in _extract_nmnist_bench(nmnist_bench[-1]):
            sno = rec["sample"]
            sp = compute_sparsity(nmnist_imgs[sno], num_steps=5)
            records.append({**rec, **sp})

    # --- DVS Gesture ---
    dvs_bench = sorted(bench_dir.glob("dvsgesture_bench_*.json"))
    if dvs_bench:
        dvs_imgs = _load_dataset_images("dvs_gesture", num_steps=5)
        for rec in _extract_dvs_bench(dvs_bench[-1]):
            sno = rec["sample"]
            sp = compute_sparsity(dvs_imgs[sno], num_steps=5)
            records.append({**rec, **sp})

    # Compute speedup field.
    for r in records:
        lg = r.get("legacy_time"); bc = r.get("bcibp_time")
        if r.get("legacy_timeout"):
            r["speedup"] = None; r["speedup_note"] = "legacy T.O., lower bound ≥300/bc"
        elif lg is None or bc is None or bc <= 0:
            r["speedup"] = None; r["speedup_note"] = "missing"
        else:
            r["speedup"] = lg / bc; r["speedup_note"] = "ok"

    out_json = bench_dir / "sparsity_analysis.json"
    out_json.write_text(json.dumps(records, indent=2))

    # --- Correlation tables ---
    from collections import defaultdict as dd
    groups = dd(list)
    for r in records:
        groups[(r["dataset"], r["config"])].append(r)

    def _pearson(xs, ys):
        xs = np.asarray(xs); ys = np.asarray(ys)
        if xs.size < 2: return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    def _spearman(xs, ys):
        xs = np.asarray(xs); ys = np.asarray(ys)
        if xs.size < 2: return float("nan")
        # Rank and Pearson on ranks
        rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
        return float(np.corrcoef(rx, ry)[0, 1])

    out_md = bench_dir / "sparsity_summary.md"
    lines: list[str] = []
    lines.append("# Temporal sparsity ↔ BC-IBP speedup")
    lines.append("")
    lines.append("Per-dataset Pearson & Spearman correlations between sample-level "
                 "sparsity metrics and BC-IBP speedup (over voltage-margin filter baseline).")
    lines.append("")
    lines.append("| Dataset | Config | n | C (conc) | H (entropy) | active | speedup mean | r(C) | r(H) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (ds, cfg), recs in sorted(groups.items()):
        n = len(recs)
        cs = [r["concentration"] for r in recs]
        hs = [r["entropy_norm"] for r in recs]
        acts = [r["active_fraction"] for r in recs]
        # Speedup can be None (timeout); replace with lower bound 300/bc if legacy TO.
        def _sp(r):
            if r.get("legacy_timeout") and r.get("bcibp_time"):
                return 300.0 / r["bcibp_time"]
            return r.get("speedup")
        sps = [_sp(r) for r in recs]
        sps_valid = [s for s in sps if s is not None and math.isfinite(s)]
        if not sps_valid:
            lines.append(f"| {ds} | {cfg} | {n} | — | — | — | — | — | — |")
            continue
        mean_sp = sum(sps_valid) / len(sps_valid)
        # Correlate only where speedup defined
        pairs_C = [(c, s) for c, s in zip(cs, sps) if s is not None]
        pairs_H = [(h, s) for h, s in zip(hs, sps) if s is not None]
        rC = _pearson([p[0] for p in pairs_C], [p[1] for p in pairs_C])
        rH = _pearson([p[0] for p in pairs_H], [p[1] for p in pairs_H])
        lines.append(f"| {ds} | {cfg} | {n} | {np.mean(cs):.3f} | {np.mean(hs):.3f} | "
                     f"{np.mean(acts):.3f} | {mean_sp:.2f} | {rC:+.2f} | {rH:+.2f} |")
    lines.append("")
    lines.append("**Interpretation**: Higher concentration C → more sparse → better BC-IBP. ")
    lines.append("Lower entropy H → more sparse. Speedup should correlate positively with C ")
    lines.append("and negatively with H. Sign flip per dataset reveals regime.")
    lines.append("")

    # --- Cross-dataset aggregate ---
    lines.append("## Cross-dataset aggregate")
    lines.append("")
    lines.append("| Dataset | Config | C | H | active | BC-IBP time (median) | legacy time (median) | speedup (median) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for (ds, cfg), recs in sorted(groups.items()):
        cs = [r["concentration"] for r in recs]
        hs = [r["entropy_norm"] for r in recs]
        acts = [r["active_fraction"] for r in recs]
        bcs = [r["bcibp_time"] for r in recs if r.get("bcibp_time") is not None]
        lgs = [r["legacy_time"] for r in recs if r.get("legacy_time") is not None]
        sps = []
        for r in recs:
            if r.get("legacy_timeout") and r.get("bcibp_time"):
                sps.append(300.0 / r["bcibp_time"])
            elif r.get("speedup") is not None:
                sps.append(r["speedup"])
        lines.append(f"| {ds} | {cfg} | {np.median(cs):.3f} | {np.median(hs):.3f} | "
                     f"{np.median(acts):.3f} | "
                     f"{np.median(bcs):.3f}s | "
                     f"{(np.median(lgs) if lgs else '—')}{'s' if lgs else ''} | "
                     f"{np.median(sps) if sps else '—'}{'×' if sps else ''} |")
    lines.append("")

    # --- Scatter-plot-ready table ---
    lines.append("## Per-sample records (for scatter)")
    lines.append("")
    lines.append("| dataset | cfg | sample | C | H | active | BC-IBP t | legacy t | speedup |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        lt = "T.O." if r.get("legacy_timeout") else (
            f"{r['legacy_time']:.2f}" if r.get("legacy_time") is not None else "—")
        bt = f"{r['bcibp_time']:.3f}" if r.get("bcibp_time") is not None else "—"
        sp_val = None
        if r.get("legacy_timeout") and r.get("bcibp_time"):
            sp_val = 300.0 / r["bcibp_time"]
            sp_str = f"≥{sp_val:.1f}×"
        elif r.get("speedup") is not None:
            sp_str = f"{r['speedup']:.2f}×"
        else:
            sp_str = "—"
        lines.append(f"| {r['dataset']} | {r['config']} | {r['sample']} | "
                     f"{r['concentration']:.3f} | {r['entropy_norm']:.3f} | "
                     f"{r['active_fraction']:.3f} | {bt} | {lt} | {sp_str} |")

    out_md.write_text("\n".join(lines))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")

    # --- Matplotlib scatter: C vs BC-IBP absolute time (log) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Collect (C, bcibp_time, dataset, config) tuples with valid bcibp_time.
    pts = []
    for r in records:
        bc = r.get("bcibp_time")
        if bc is None or bc <= 0:
            continue
        pts.append((r["concentration"], bc, r["dataset"], r["config"], r.get("entropy_norm")))
    if not pts:
        return

    color_map = {
        "MNIST": "#1f77b4", "N-MNIST": "#ff7f0e", "DVS-Gesture": "#2ca02c",
        "FashionMNIST": "#d62728", "CIFAR-10": "#9467bd",
    }
    marker_map = {1: "o", 2: "s", 3: "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: C vs BC-IBP time (log y)
    by_pair = defaultdict(list)
    for c, t, ds, cfg, h in pts:
        by_pair[(ds, cfg)].append((c, t, h))
    for (ds, cfg), lst in by_pair.items():
        xs = [p[0] for p in lst]; ys = [p[1] for p in lst]
        delta_int = 2 if "δ=2" in cfg else (1 if "δ=1" in cfg else 3)
        ax1.scatter(xs, ys, label=f"{ds} {cfg}",
                    color=color_map.get(ds, "#444"),
                    marker=marker_map.get(delta_int, "x"),
                    alpha=0.7, s=55, edgecolor="k", linewidth=0.3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Concentration  C = max$_t$ p(t)  (higher = more temporally sparse)")
    ax1.set_ylabel("BC-IBP solver time (s, log)")
    ax1.set_title("Per-sample: BC-IBP time vs temporal concentration")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=7, loc="best")

    # Right: C vs speedup (log), with lower-bound arrows for timeouts
    for (ds, cfg), recs in groups.items():
        xs, ys, lbs = [], [], []
        for r in recs:
            bc = r.get("bcibp_time"); c = r["concentration"]
            if bc is None: continue
            if r.get("legacy_timeout"):
                xs.append(c); ys.append(300.0 / bc); lbs.append(True)
            elif r.get("speedup") is not None:
                xs.append(c); ys.append(r["speedup"]); lbs.append(False)
        if not xs: continue
        delta_int = 2 if "δ=2" in cfg else (1 if "δ=1" in cfg else 3)
        ax2.scatter(xs, ys, label=f"{ds} {cfg}",
                    color=color_map.get(ds, "#444"),
                    marker=marker_map.get(delta_int, "x"),
                    alpha=0.7, s=55, edgecolor="k", linewidth=0.3)
        # Arrows for lower-bound points
        for xi, yi, lb in zip(xs, ys, lbs):
            if lb:
                ax2.annotate("", xy=(xi, yi * 3), xytext=(xi, yi),
                             arrowprops=dict(arrowstyle="->",
                                             color=color_map.get(ds, "#444"), alpha=0.5))
    ax2.set_yscale("log")
    ax2.set_xlabel("Concentration  C")
    ax2.set_ylabel("Speedup vs voltage-margin filter (log)")
    ax2.set_title("Speedup vs temporal concentration")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=7, loc="best")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    fig_path = bench_dir / "sparsity_vs_speedup.pdf"
    fig_path_png = bench_dir / "sparsity_vs_speedup.png"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path_png, bbox_inches="tight", dpi=150)
    print(f"wrote {fig_path} and {fig_path_png}")


if __name__ == "__main__":
    main()
