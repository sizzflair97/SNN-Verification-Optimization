# IBP Scaling — n_h=500 δ=2 Report

**Config**: n_hidden=500, num_steps=5, delta=2, 10 samples, seed=42, per-sample timeout=300s

## Methods compared

| Tag | Description |
|---|---|
| `legacy` | `SNN_BNB_LEGACY_ACTIVE_SET=1` (baseline) |
| `legacy + IBP coupled` | + `SNN_BNB_IBP=1 SNN_BNB_IBP_COUPLED=1 SNN_BNB_IBP_EVERY=50`. 0/1-knapsack budget-coupled bound. |

## Per-sample solver time (seconds)

| Sample | Verdict | Legacy | Legacy+IBP coupled | Speedup |
|---|---|---|---|---|
| 6717  | NR | 0.05  | 0.19  | 0.3× (overhead) |
| 41905 | NR | 0.24  | 0.31  | 0.8× |
| 14628 | R  | 122.27 | 0.18  | **698×** |
| 48598 | R  | 116.74 | 0.17  | **694×** |
| 1639  | R  | 115.22 | 0.18  | **655×** |
| 48265 | R  | 120.39 | 0.19  | **630×** |
| 7296  | R  | 115.62 | 0.96  | 121× |
| 18024 | R  | 109.78 | 6.24  | 17.6× |
| 16049 | R  | 113.90 | 11.93 | 9.5× |
| 9144  | R  | 125.44 | 66.19 | 1.9× |

## Aggregate

|  | Legacy | Legacy+IBP coupled |
|---|---|---|
| **Sum (10)** | 939.6s | 86.5s |
| **Mean** | 93.96s | **8.65s** |
| **Median** | 115.4s | **0.19s** |
| **Speedup (mean)** | 1× | **10.9×** |
| **Speedup (median)** | 1× | **607×** |

## Verdict consistency

All 10 samples match exhaustive oracle (2 NR: 6717, 41905 / 8 R: rest). 0 mismatches.

## Scaling vs n_h=200

| Metric | n_h=200 | n_h=500 |
|---|---|---|
| Legacy mean | 52.2s | 93.96s |
| Coupled IBP mean | 0.81s | 8.65s |
| Mean speedup | 64× | 10.9× |
| Median speedup | 630× | 607× |

Per-sample pattern persists: 4-6 robust samples root-prune with 1 IBP call × 3 splits (~0.2s total); remaining "hard" robust samples still benefit significantly but less dramatically.

## Hard-case analysis: sample 9144

Sample 9144 is the outlier — legacy 125s vs coupled IBP 66s (only 1.9× speedup). Inspection:
- At root, coupled IBP does NOT prove robustness for all delta splits (some rem_neg/rem_pos combinations remain ambiguous).
- BnB must descend into the tree; IBP then fires periodically (every 50 branches) with ~30ms cost per call.
- Most branches don't prune at deeper states either — the adversarial margin is tight.
- Still 1.9× faster than pure legacy because IBP catches some otherwise-explored subtrees.

This sample points to the next optimization target: cross-layer budget-coupled IBP (propagate budget through hidden→output voltage bounds) to tighten outputs in hard cases.

## NR cases: small overhead

Samples 6717, 41905 show slight slowdown (legacy already ~0.05-0.25s) because coupled IBP adds 0.1-0.3s fixed cost. At large `n_h` this overhead is more noticeable, but absolute cost remains sub-second.

## Reproduction

```bash
SNN_BNB_EFFICIENT=0 SNN_BNB_LEGACY_ACTIVE_SET=1 \
SNN_BNB_IBP=1 SNN_BNB_IBP_COUPLED=1 SNN_BNB_IBP_EVERY=50 \
python batch_test.py -p coupled500 --test-type mnist --num-samples 10 \
  --n-hidden-neurons 500 --num-steps 5 --delta-max 2 --seed 42 --np
```

## Source data

- [bench_results/nh500_delta2_ibp.json](bench_results/nh500_delta2_ibp.json) — raw results
- [bench_results/nh500_delta2.log](bench_results/nh500_delta2.log) — execution log
