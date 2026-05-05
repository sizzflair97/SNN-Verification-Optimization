# Realistic Large-Network Benchmark Results

**Config**: num_steps=5, delta=1, samples/config=10, per-sample timeout=300s, seed=42

**Run ID**: `0416175051`

**Source**: `realistic_bench_0416175051.json`


## Solver time (seconds) — primary metric

Reports solver time only (excludes Python startup + model load).

Format: `mean (median) [completed/total]`; `T.O.` = timeout; `-` = skipped.


| Method | n_h=100 | n_h=200 | n_h=300 | n_h=500 |
|---|---|---|---|---|
| Exhaustive DFS (oracle) | 0.581 (0.585) [10/10] | 0.647 (0.644) [10/10] | 0.666 (0.667) [10/10] | 0.737 (0.744) [10/10] |
| Z3 SMT | ? [0/10] | — (skipped) | — (skipped) | — (skipped) |
| MILP (CBC) | 5.633 (5.773) [10/10] | 74.601 (67.604) [10/10] | 195.824 (206.297) [4/10] +3T.O. +3skip | — (skipped) |
| BnB (efficient) | 0.161 (0.162) [10/10] | 0.213 (0.212) [10/10] | 0.275 (0.274) [10/10] | 0.374 (0.373) [10/10] |
| BnB (legacy margin) | 0.024 (0.027) [10/10] | 0.034 (0.036) [10/10] | 0.041 (0.042) [10/10] | 0.050 (0.056) [10/10] |
| BnB (no filter) | 0.145 (0.159) [10/10] | 0.195 (0.195) [10/10] | 0.229 (0.228) [10/10] | 0.270 (0.294) [10/10] |
| BnB (efficient+PSM) | 0.163 (0.162) [10/10] | 0.213 (0.213) [10/10] | 0.269 (0.267) [10/10] | 0.371 (0.369) [10/10] |

## Wall time (seconds) — end-to-end including subprocess startup

| Method | n_h=100 | n_h=200 | n_h=300 | n_h=500 |
|---|---|---|---|---|
| Exhaustive DFS (oracle) | 0.58 (0.58) | 0.65 (0.64) | 0.67 (0.67) | 0.74 (0.74) |
| Z3 SMT | 330.23 (330.23) | — | — | — |
| MILP (CBC) | 15.61 (15.73) | 87.91 (80.86) | 263.08 (281.00) | — |
| BnB (efficient) | 6.74 (6.74) | 6.74 (6.73) | 7.01 (7.03) | 6.89 (6.88) |
| BnB (legacy margin) | 6.61 (6.60) | 6.54 (6.53) | 6.53 (6.53) | 6.56 (6.56) |
| BnB (no filter) | 6.71 (6.71) | 6.68 (6.68) | 6.71 (6.71) | 6.76 (6.79) |
| BnB (efficient+PSM) | 6.72 (6.72) | 6.71 (6.71) | 6.75 (6.74) | 6.87 (6.87) |

## Verdict consistency check

Per-sample verdicts across methods. Discrepancies indicate unsoundness.


### n_h=100

| Sample | exhaustive | z3 | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|---|
| 1639 | NR | s | NR | NR | NR | NR | NR |
| 6717 | R | s | R | R | R | R | R |
| 7296 | R | T | R | R | R | R | R |
| 9144 | R | s | R | R | R | R | R |
| 14628 | R | s | R | R | R | R | R |
| 16049 | R | s | R | R | R | R | R |
| 18024 | R | s | R | R | R | R | R |
| 41905 | R | T | R | R | R | R | R |
| 48265 | R | s | R | R | R | R | R |
| 48598 | R | s | R | R | R | R | R |

**Mismatches in n_h=100**: 0

### n_h=200

| Sample | exhaustive | z3 | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|---|
| 1639 | R | S | R | R | R | R | R |
| 6717 | R | S | R | R | R | R | R |
| 7296 | R | S | R | R | R | R | R |
| 9144 | R | S | R | R | R | R | R |
| 14628 | R | S | R | R | R | R | R |
| 16049 | R | S | R | R | R | R | R |
| 18024 | R | S | R | R | R | R | R |
| 41905 | R | S | R | R | R | R | R |
| 48265 | R | S | R | R | R | R | R |
| 48598 | R | S | R | R | R | R | R |

**Mismatches in n_h=200**: 0

### n_h=300

| Sample | exhaustive | z3 | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|---|
| 1639 | R | S | R | R | R | R | R |
| 6717 | R | S | s | R | R | R | R |
| 7296 | R | S | R | R | R | R | R |
| 9144 | R | S | s | R | R | R | R |
| 14628 | R | S | T | R | R | R | R |
| 16049 | R | S | T | R | R | R | R |
| 18024 | R | S | R | R | R | R | R |
| 41905 | R | S | T | R | R | R | R |
| 48265 | R | S | s | R | R | R | R |
| 48598 | R | S | R | R | R | R | R |

**Mismatches in n_h=300**: 0

### n_h=500

| Sample | exhaustive | z3 | milp | bnb_efficient | bnb_legacy | bnb_nofilter | bnb_psm |
|---|---|---|---|---|---|---|---|
| 1639 | R | S | S | R | R | R | R |
| 6717 | NR | S | S | NR | NR | NR | NR |
| 7296 | R | S | S | R | R | R | R |
| 9144 | R | S | S | R | R | R | R |
| 14628 | R | S | S | R | R | R | R |
| 16049 | R | S | S | R | R | R | R |
| 18024 | R | S | S | R | R | R | R |
| 41905 | R | S | S | R | R | R | R |
| 48265 | R | S | S | R | R | R | R |
| 48598 | R | S | S | R | R | R | R |

**Mismatches in n_h=500**: 0

## Speedup vs Exhaustive DFS (solver time, completed samples only)

| Method | n_h=100 | n_h=200 | n_h=300 | n_h=500 |
|---|---|---|---|---|
| Z3 SMT | — | — | — | — |
| MILP (CBC) | 0.10× | 0.01× | 0.00× | — |
| BnB (efficient) | 3.60× | 3.03× | 2.42× | 1.97× |
| BnB (legacy margin) | 24.32× | 19.04× | 16.22× | 14.71× |
| BnB (no filter) | 3.99× | 3.32× | 2.91× | 2.73× |
| BnB (efficient+PSM) | 3.56× | 3.03× | 2.48× | 1.99× |
