# Delta-Scaling Benchmark

**Config**: n_hidden=200, num_steps=5, deltas=[1, 2, 3, 4], samples/config=10, per-sample timeout=300s, seed=42

**Run ID**: `0416213348`


## Solver time (seconds) — primary metric

Format: `mean (median) [completed/total]`; `T.O.` = timeout; `-` = skipped.


| Method | δ=1 | δ=2 | δ=3 | δ=4 |
|---|---|---|---|---|
| Exhaustive DFS (oracle) | 0.637 (0.624) [10/10] | 225.462 (257.390) [10/10] | 254.498 (300.001) [5/10] +3T.O. +5skip | 276.266 (300.001) [5/10] +3T.O. +5skip |
| MILP (CBC) | 74.536 (67.683) [10/10] | ? [0/10] | ? [0/10] | ? [0/10] |
| BnB (efficient) | 0.213 (0.212) [10/10] | 0.220 (0.221) [10/10] | 0.235 (0.235) [10/10] | 0.347 (0.261) [10/10] |
| BnB (legacy margin) | 0.034 (0.035) [10/10] | 52.205 (72.939) [10/10] | 2.368 (2.597) [3/10] +4T.O. +3skip | 1.344 (1.318) [3/10] +4T.O. +3skip |
| BnB (no filter) | 0.198 (0.199) [10/10] | 52.049 (71.637) [10/10] | 2.303 (2.627) [3/10] +4T.O. +3skip | 1.295 (1.312) [3/10] +4T.O. +3skip |
| BnB (efficient+PSM) | 0.212 (0.211) [10/10] | 0.222 (0.222) [10/10] | 0.227 (0.228) [10/10] | 0.345 (0.251) [10/10] |

## Not-robust rate (per delta)

Fraction of completed samples found to be adversarial.


| Method | δ=1 | δ=2 | δ=3 | δ=4 |
|---|---|---|---|---|
| Exhaustive DFS (oracle) | 0/10 (0%) | 3/10 (30%) | 2/2 (100%) | 2/2 (100%) |
| MILP (CBC) | 0/10 (0%) | — | — | — |
| BnB (efficient) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |
| BnB (legacy margin) | 0/10 (0%) | 3/10 (30%) | 3/3 (100%) | 3/3 (100%) |
| BnB (no filter) | 0/10 (0%) | 3/10 (30%) | 3/3 (100%) | 3/3 (100%) |
| BnB (efficient+PSM) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |

## Verdict consistency vs Exhaustive (oracle)

If exhaustive says NR but a BnB variant says R → **unsound** (missed adversarial). Vice versa rare.


| Method | δ=1 | δ=2 | δ=3 | δ=4 |
|---|---|---|---|---|
| MILP (CBC) | 0/10 | — | — | — |
| BnB (efficient) | 0/10 | 3/10 ⚠ | 2/2 ⚠ | 1/2 ⚠ |
| BnB (legacy margin) | 0/10 | 0/10 | 0/2 | 0/2 |
| BnB (no filter) | 0/10 | 0/10 | 0/2 | 0/2 |
| BnB (efficient+PSM) | 0/10 | 3/10 ⚠ | 2/2 ⚠ | 1/2 ⚠ |

## Per-sample verdicts per delta

### δ=1

| Sample | exhaustive | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|
| 1639 | R | R | R | R | R | R |
| 6717 | R | R | R | R | R | R |
| 7296 | R | R | R | R | R | R |
| 9144 | R | R | R | R | R | R |
| 14628 | R | R | R | R | R | R |
| 16049 | R | R | R | R | R | R |
| 18024 | R | R | R | R | R | R |
| 41905 | R | R | R | R | R | R |
| 48265 | R | R | R | R | R | R |
| 48598 | R | R | R | R | R | R |

### δ=2

| Sample | exhaustive | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|
| 1639 | NR | s | R | NR | NR | R |
| 6717 | NR | s | R | NR | NR | R |
| 7296 | R | T | R | R | R | R |
| 9144 | R | s | R | R | R | R |
| 14628 | R | s | R | R | R | R |
| 16049 | R | s | R | R | R | R |
| 18024 | R | s | R | R | R | R |
| 41905 | NR | T | R | NR | NR | R |
| 48265 | R | s | R | R | R | R |
| 48598 | R | s | R | R | R | R |

### δ=3

| Sample | exhaustive | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|
| 1639 | NR | s | R | NR | NR | R |
| 6717 | s | s | R | s | s | R |
| 7296 | T | T | R | T | T | R |
| 9144 | s | s | R | s | s | R |
| 14628 | s | s | R | T | T | R |
| 16049 | s | s | R | T | T | R |
| 18024 | T | s | R | NR | NR | R |
| 41905 | NR | T | R | NR | NR | R |
| 48265 | s | s | R | s | s | R |
| 48598 | T | s | R | T | T | R |

### δ=4

| Sample | exhaustive | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|
| 1639 | NR | s | NR | NR | NR | NR |
| 6717 | s | s | R | s | s | R |
| 7296 | T | T | R | T | T | R |
| 9144 | s | s | R | s | s | R |
| 14628 | s | s | R | T | T | R |
| 16049 | s | s | R | T | T | R |
| 18024 | T | s | R | NR | NR | R |
| 41905 | NR | T | R | NR | NR | R |
| 48265 | s | s | R | s | s | R |
| 48598 | T | s | R | T | T | R |
