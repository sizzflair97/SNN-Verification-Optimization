"""Exhaustively validate a counterexample to the stated full active-set filter.

The filter interpretation tested here is deliberately stronger than the prose
in paper/main.tex:

* temporal condition:
      orig(p) <= t_ref + 2 * Delta - tau
* voltage-margin condition:
      |w[p,h]| < |V_h(t) - theta| for every hidden h and every timestep t

A pixel is removed when it is individually below every voltage margin.  The
example shows that two removed pixels can jointly cross a threshold.
"""

from __future__ import annotations

from itertools import product

import numpy as np


THETA = 1.0
T = 13
DELTA = 3
TAU = 2

NAMES = ("q", "r", "u1", "u2", "u3", "p1", "p2", "p3")
ORIG = np.array((0, 1, 8, 8, 8, 9, 9, 9), dtype=int)

# Hidden order: competitor h_c, target h_y.
W_INPUT_HIDDEN = np.array(
    (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6),
        (0.0, 0.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0),
    ),
    dtype=float,
)
W_INPUT_HIDDEN[1, 0] = 2.0
W_INPUT_HIDDEN[1, 1] = -2.0

# Output order: competitor c=0, target y=1.  A tie is won by c.
W_HIDDEN_OUTPUT = np.array(((2.0, 0.0), (0.0, 2.0)), dtype=float)
TARGET = 1


def layer_forward(
    input_times: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative IF/TTFS layer with one-step delay and terminal firing."""
    voltage = np.zeros((weights.shape[0], T), dtype=float)
    for t in range(T):
        voltage[:, t] = weights[:, input_times <= t].sum(axis=1)
    voltage[:, T - 1] = THETA + 1.0
    crossings = voltage > THETA
    spike_times = np.argmax(crossings, axis=1) + 1
    spike_times = np.minimum(spike_times, T - 1)
    return spike_times.astype(int), voltage


def forward(input_times: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    hidden_times, hidden_voltage = layer_forward(input_times, W_INPUT_HIDDEN)
    output_times, _ = layer_forward(hidden_times, W_HIDDEN_OUTPUT)
    return int(np.argmin(output_times)), hidden_times, output_times


def perturbations(indices: tuple[int, ...]):
    choices = []
    for i in indices:
        lo = max(0, int(ORIG[i]) - DELTA)
        hi = min(T - 1, int(ORIG[i]) + DELTA)
        choices.append(range(lo, hi + 1))
    for values in product(*choices):
        candidate = ORIG.copy()
        candidate[list(indices)] = values
        if int(np.abs(candidate - ORIG).sum()) <= DELTA:
            yield candidate


baseline_pred, baseline_hidden, baseline_output = forward(ORIG)
_, baseline_voltage = layer_forward(ORIG, W_INPUT_HIDDEN)
t_ref = int(baseline_output.min())

small_margin = []
temporal_keep = []
for i in range(len(NAMES)):
    below_all_margins = bool(
        np.all(
            np.abs(W_INPUT_HIDDEN[:, i, None])
            < np.abs(baseline_voltage - THETA)
        )
    )
    small_margin.append(below_all_margins)
    temporal_keep.append(bool(ORIG[i] <= t_ref + 2 * DELTA - TAU))

# Under either the "small weight OR temporally late" interpretation or the
# stricter conjunction used to identify disposable late pixels, u_i and p_i
# satisfy both conditions.  q and r remain active.
active = tuple(
    i
    for i in range(len(NAMES))
    if temporal_keep[i] and not small_margin[i]
)

unfiltered_witness = None
for candidate in perturbations(tuple(range(len(NAMES)))):
    pred, _, _ = forward(candidate)
    if pred != baseline_pred:
        unfiltered_witness = candidate
        break

filtered_witness = None
for candidate in perturbations(active):
    pred, _, _ = forward(candidate)
    if pred != baseline_pred:
        filtered_witness = candidate
        break

claimed_witness = ORIG.copy()
claimed_witness[0] = 1  # q: 0 -> 1, merges with r and cancels.
claimed_witness[5] = 8  # p1: 9 -> 8.
claimed_witness[6] = 8  # p2: 9 -> 8; p1+p2 jointly cross theta.
claimed_pred, claimed_hidden, claimed_output = forward(claimed_witness)

print("baseline:", dict(zip(NAMES, ORIG)), baseline_hidden, baseline_output, baseline_pred)
print("t_ref / cutoff:", t_ref, t_ref + 2 * DELTA - TAU)
print("small_margin:", dict(zip(NAMES, small_margin)))
print("temporal_keep:", dict(zip(NAMES, temporal_keep)))
print("active:", tuple(NAMES[i] for i in active))
print(
    "claimed witness:",
    dict(zip(NAMES, claimed_witness)),
    "cost=",
    int(np.abs(claimed_witness - ORIG).sum()),
    claimed_hidden,
    claimed_output,
    claimed_pred,
)
print("unfiltered adversarial exists:", unfiltered_witness is not None)
print("filtered adversarial exists:", filtered_witness is not None)

assert baseline_pred == TARGET
assert all(small_margin[2:])
assert not any(temporal_keep[2:])
assert claimed_pred != baseline_pred
assert int(np.abs(claimed_witness - ORIG).sum()) == DELTA
assert unfiltered_witness is not None
assert filtered_witness is None
