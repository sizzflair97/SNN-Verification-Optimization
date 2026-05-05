"""Generate Appendix figure: MILP and Z3 solver-intrinsic scalability vs n_h.

Data source: bench_results/realistic_bench_0416175051.json
MNIST, T=5, delta=1, 10 samples per n_h, 300s per-sample timeout.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "bench_results/realistic_bench_0416175051.json"
OUT_PDF = ROOT / "paper/NeurIPS 2026/smt_milp_scaling.pdf"

d = json.load(SRC.open())
by = defaultdict(list)
for r in d["results"]:
    m = r["method"]; nh = r["n_hidden"]
    st = r.get("solver_time"); to = r.get("timed_out", False)
    by[(m, nh)].append((st, to, r.get("verdict")))

# Extract per-n_h stats
methods = ["milp", "z3", "bnb_efficient", "bnb_legacy", "exhaustive"]
labels  = {"milp":"MILP (CBC)", "z3":"Z3 SMT",
           "bnb_efficient":"BnB+efficient", "bnb_legacy":"BnB+voltage-margin",
           "exhaustive":"Exhaustive DFS"}
colors  = {"milp":"#d62728", "z3":"#9467bd",
           "bnb_efficient":"#2ca02c", "bnb_legacy":"#1f77b4",
           "exhaustive":"#7f7f7f"}
markers = {"milp":"s", "z3":"D", "bnb_efficient":"o", "bnb_legacy":"^", "exhaustive":"x"}
nhs = sorted({nh for (m, nh) in by.keys()})

fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.3))

for m in methods:
    xs, ys, tops = [], [], []
    for nh in nhs:
        recs = by.get((m, nh), [])
        if not recs: continue
        completed = [st for st, to, v in recs if st is not None and not to]
        n_to = sum(1 for st, to, v in recs if to)
        n = len(recs)
        if completed:
            xs.append(nh)
            ys.append(float(np.mean(completed)))
            tops.append(False)
        elif n_to == n:
            # All timed out → plot lower-bound at 300s with up-arrow
            xs.append(nh)
            ys.append(300.0)
            tops.append(True)
    if not xs: continue
    ax.plot(xs, ys, marker=markers[m], color=colors[m], label=labels[m],
            linewidth=1.8, markersize=8, markeredgecolor="k", markeredgewidth=0.4)
    for x, y, top in zip(xs, ys, tops):
        if top:
            ax.annotate("", xy=(x, y * 1.8), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=colors[m], lw=1.4))

ax.axhline(300.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(nhs[0] * 1.02, 300 * 1.05, "300\\,s per-sample timeout", color="gray", fontsize=8)

ax.set_xlabel("hidden layer width $n_h$", fontsize=11)
ax.set_ylabel("solver time (s, mean over completed)", fontsize=11)
ax.set_yscale("log")
ax.set_xticks(nhs)
ax.set_xticklabels([str(n) for n in nhs])
ax.set_title("Solver-intrinsic scalability on MNIST ($T=5,\\,\\Delta=1$, 10 samples)",
             fontsize=11)
ax.grid(True, which="both", linestyle=":", alpha=0.45)
ax.legend(loc="center right", fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PDF}")
