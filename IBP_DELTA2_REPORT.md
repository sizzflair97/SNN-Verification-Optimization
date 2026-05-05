# IBP Pruning — δ=2 Benchmark Report (updated with budget-coupled variant)

**Config**: n_hidden=200, num_steps=5, delta=2, 10 samples, seed=42, per-sample timeout=300s

## Methods compared

| Tag | Description |
|---|---|
| `legacy` | `SNN_BNB_LEGACY_ACTIVE_SET=1` (baseline) |
| `legacy + IBP (uncoupled, every=50)` | + `SNN_BNB_IBP=1 SNN_BNB_IBP_EVERY=50`. Per-pixel independent bound. |
| `legacy + IBP (coupled, every=50)` | + `SNN_BNB_IBP_COUPLED=1`. 0/1-knapsack budget-coupled bound. |

## Per-sample solver time (seconds)

| Sample | Verdict | legacy | +IBP uncoupled | +IBP **coupled** | Speedup vs legacy |
|---|---|---|---|---|---|
| 6717  | NR | 0.13 | 0.13 | **0.12** | 1.08× |
| 1639  | NR | 1.30 | 1.66 | **0.69** | **1.88×** |
| 41905 | NR | 2.53 | 3.32 | **0.86** | **2.94×** |
| 18024 | R  | 69.05 | 16.26 | **0.10** | **~700×** |
| 16049 | R  | 72.60 | 32.38 | **0.10** | **~730×** |
| 48265 | R  | 73.28 | 29.81 | **0.10** | **~730×** |
| 48598 | R  | 73.47 | 26.82 | **0.10** | **~730×** |
| 7296  | R  | 75.69 | 26.72 | **0.10** | **~760×** |
| 14628 | R  | 75.99 | 38.33 | **5.81** | 13.1× |
| 9144  | R  | 78.01 | 49.28 | **0.10** | **~780×** |

## Aggregate

|  | legacy | +IBP uncoupled | +IBP **coupled** |
|---|---|---|---|
| **Sum (10)** | 522.2 | 224.7 | **8.08** |
| **Mean** | 52.22 | 22.47 | **0.81** |
| **Median** | 72.93 | 26.77 | **0.10** |
| **Speedup vs legacy (mean)** | 1× | 2.32× | **64.4×** |

## Verdict consistency

All 10 samples match exhaustive oracle (3 NR, 7 R). 0 mismatches.

## Why coupled is dramatically faster

**Root-level proof**: 6 of 7 robust samples are proved robust at the BnB root by the budget-coupled IBP alone, skipping BnB entirely (3 IBP calls × ~30ms = 0.1s total). The uncoupled IBP at root allows every pixel to independently flip with full budget — far too loose — so it never fires at the root.

**Budget-coupled bound** formulates max V_h(t) (per hidden h, time t) as a 0/1 knapsack: items are "flip baseline firing status of pixel p at time t", cost = absolute shift needed, budget = remaining L1. Solved exactly by O(N·budget·n_h) DP vectorized over hidden-neuron axis (≈30 ms per IBP call at n_h=200).

**Uncoupled bound** allows each pixel to flip independently — equivalent to giving every pixel its own copy of the full budget. On 784 pixels × δ=2 this adds up to ≈1568 "free" flips of voltage, making the bound useless at shallow depth. Only becomes useful near leaves.

## Beyond δ=2 — early evidence

δ=3, sample 7296 (n_h=200):
- `legacy`: timed out at 300s+ in prior run
- `legacy + coupled IBP (every=50)`: **89.9s complete**, 99.9% prune rate (8607/8617 IBP calls pruned)

First tractable δ=3 result on this configuration.

## Reproduction

```bash
# Coupled IBP (recommended)
SNN_BNB_LEGACY_ACTIVE_SET=1 SNN_BNB_EFFICIENT=0 \
SNN_BNB_IBP=1 SNN_BNB_IBP_COUPLED=1 SNN_BNB_IBP_EVERY=50 \
python batch_test.py -p coupled --test-type mnist --num-samples 10 \
  --n-hidden-neurons 200 --num-steps 5 --delta-max 2 --seed 42 --np
```

## Source

- IBP helper: [bnb_ibp.py](bnb_ibp.py) — `ibp_prove_robust_budgeted`
- BnB integration: [adv_rob_mnist_module.py:603-607,807-829](adv_rob_mnist_module.py)
- Env vars: `SNN_BNB_IBP`, `SNN_BNB_IBP_COUPLED`, `SNN_BNB_IBP_EVERY`
