# PSM Standalone Contribution Report

**Setting:** MNIST, T=5, n_hidden=200, δ=2, 10 samples (seed=42), per-sample timeout=300s.

**Question:** What does PSM contribute *independently of* the pre-filter (legacy / efficient / nofilter)? Each pair holds the filter fixed and toggles PSM only.

## Note on filter soundness

- `efficient` filter is **unsound for δ ≥ 2** (may miss adversarials) — included for timing only, not robustness claims.
- `legacy` voltage-margin filter is sound.
- `no filter` is the pure BnB baseline (sound).

## Aggregate timings (wall seconds)

| Filter | PSM | mean | median | robust | not_robust | timeout |
|---|---|---:|---:|---:|---:|---:|
| no filter | off | 59.04 | 78.65 | 7 | 3 | 0 |
| no filter | on | 55.24 | 73.47 | 7 | 3 | 0 |
| legacy | off | 59.19 | 79.91 | 7 | 3 | 0 |
| legacy | on | 56.26 | 74.52 | 7 | 3 | 0 |
| efficient | off | 7.15 | 7.15 | 10 | 0 | 0 |
| efficient | on | 7.18 | 7.18 | 10 | 0 | 0 |

## Per-pair speedup (no-PSM wall / PSM wall)

### Filter = no filter  (bnb_nofilter  vs  bnb_nofilter_psm)

| sample | verdict (no PSM / PSM) | wall no-PSM (s) | wall PSM (s) | speedup |
|---:|:--|---:|---:|---:|
| 1639 | not_robust / not_robust | 8.25 | 7.28 | 1.13× |
| 6717 | not_robust / not_robust | 7.14 | 6.53 | 1.09× |
| 7296 | robust / robust | 81.95 | 76.11 | 1.08× |
| 9144 | robust / robust | 84.90 | 79.51 | 1.07× |
| 14628 | robust / robust | 83.86 | 80.07 | 1.05× |
| 16049 | robust / robust | 77.30 | 72.56 | 1.07× |
| 18024 | robust / robust | 77.61 | 72.27 | 1.07× |
| 41905 | not_robust / not_robust | 9.51 | 8.45 | 1.13× |
| 48265 | robust / robust | 80.19 | 75.23 | 1.07× |
| 48598 | robust / robust | 79.69 | 74.38 | 1.07× |

*n=10 paired. Geometric-mean speedup: **1.08×**; mean: 1.08×; median: 1.07×; min: 1.05×; max: 1.13×.*

### Filter = legacy  (bnb_legacy  vs  bnb_legacy_psm)

| sample | verdict (no PSM / PSM) | wall no-PSM (s) | wall PSM (s) | speedup |
|---:|:--|---:|---:|---:|
| 1639 | not_robust / not_robust | 8.24 | 7.30 | 1.13× |
| 6717 | not_robust / not_robust | 7.10 | 6.85 | 1.04× |
| 7296 | robust / robust | 82.75 | 78.21 | 1.06× |
| 9144 | robust / robust | 85.05 | 80.13 | 1.06× |
| 14628 | robust / robust | 82.91 | 79.04 | 1.05× |
| 16049 | robust / robust | 79.57 | 73.20 | 1.09× |
| 18024 | robust / robust | 76.03 | 72.17 | 1.05× |
| 41905 | not_robust / not_robust | 9.56 | 8.53 | 1.12× |
| 48265 | robust / robust | 80.25 | 81.32 | 0.99× |
| 48598 | robust / robust | 80.40 | 75.84 | 1.06× |

*n=10 paired. Geometric-mean speedup: **1.06×**; mean: 1.06×; median: 1.06×; min: 0.99×; max: 1.13×.*

### Filter = efficient  (bnb_efficient  vs  bnb_psm)

| sample | verdict (no PSM / PSM) | wall no-PSM (s) | wall PSM (s) | speedup |
|---:|:--|---:|---:|---:|
| 1639 | robust / robust | 7.11 | 7.20 | 0.99× |
| 6717 | robust / robust | 7.19 | 7.13 | 1.01× |
| 7296 | robust / robust | 7.18 | 7.19 | 1.00× |
| 9144 | robust / robust | 7.14 | 7.23 | 0.99× |
| 14628 | robust / robust | 7.16 | 7.23 | 0.99× |
| 16049 | robust / robust | 7.16 | 7.16 | 1.00× |
| 18024 | robust / robust | 7.13 | 7.17 | 1.00× |
| 41905 | robust / robust | 7.16 | 7.20 | 0.99× |
| 48265 | robust / robust | 7.13 | 7.17 | 0.99× |
| 48598 | robust / robust | 7.08 | 7.11 | 1.00× |

*n=10 paired. Geometric-mean speedup: **1.00×**; mean: 1.00×; median: 0.99×; min: 0.99×; max: 1.01×.*

## Verdict consistency check

PSM should not change any verdict (both soundness and completeness preserved on sound filters).

| Filter | matched verdicts | disagreements |
|---|---:|---:|
| no filter | 10/10 | 0 |
| legacy | 10/10 | 0 |
| efficient | 10/10 | 0 |

## Bottom line

PSM (Prefix-Set Matching) is an optimization layered on top of any pre-filter. 
The three pairs above isolate its contribution:

- vs **no filter**: measures PSM's raw pruning power on the full pixel set.
- vs **legacy**: PSM's *additive* benefit beyond sound voltage-margin pruning.
- vs **efficient**: PSM stacked on sensitivity-based ordering (the shipped default).
