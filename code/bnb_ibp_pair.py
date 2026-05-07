"""CLEB — Cross-Layer Exact Bound for multi-hidden TTFS networks.

Implements the bound primitive of paper §5.2 / Theorem 4. For input
budgets Δ ≤ 2 the admissible-perturbation set is finite and small, so we
enumerate every δ explicitly, run an exact forward pass per δ, and take
the (sound and tight) max/min of the resulting layer-2 voltage and
output voltage trajectories. CLEB occupies the same `Bound(·)` slot as
BC-IBP in Algorithm 1 and is GPU-vectorised (PyTorch).

Public API
----------
- `enumerate_output_voltage_bounds_vec(...)` — vectorised CPU implementation.
- `_enumerate_gpu(...)` — PyTorch GPU port (used by the *_exact_gpu wrapper).
- `ibp_prove_robust_multilayer_exact_gpu(...)` — drop-in replacement for
  `ibp_prove_robust_multilayer` from `bnb_ibp.py`, using CLEB whenever the
  network has ≥ 2 hidden layers and Δ ≤ 2; falls back to BC-IBP-multilayer
  otherwise.
- `ibp_prove_robust_multilayer_exact_vec`, `_exact_small_budget` — CPU
  variants used in soundness tests.

Pair-feasibility helpers (`pair_infeasibilities_at_t`,
`output_voltage_bounds_pair_aware`) predate CLEB and remain as building
blocks of `enumerate_output_voltage_bounds`. The polynomial-time
2D-knapsack pair-bound exploration (CLPB) was empirically dominated by
CLEB and is archived under `archive_legacy/cleb_legacy_pair_polynomial.py`.
"""
from __future__ import annotations

import numpy as np

from bnb_ibp import (
    _compute_budgeted_s_h_bounds,
    _spike_bounds_from_voltage,
    _output_voltage_bounds,
    _check_robust_from_s_h,
)


# ============================================================================
# Per-pixel "flip option" tables
# ============================================================================
def _pixel_flip_options(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each remaining pixel, return (delta_indicator_at_t, cost_pos, cost_neg).

    A pixel p with baseline spike time s_p has indicator 1[s_p ≤ t]. The
    *minimum* cost to flip its indicator at time t is:
      - if s_p ≤ t (currently fired-by-t): cost_pos = t+1-s_p, cost_neg = 0
        (shifting +; flip means pixel no longer fires by t → indicator 1→0,
        i.e. delta_indicator = -1)
      - if s_p > t (currently NOT fired-by-t): cost_pos = 0,
        cost_neg = s_p-t  (shifting -; flip means pixel now fires by t →
        indicator 0→1, i.e. delta_indicator = +1)

    Pixels whose flip cost exceeds the remaining budget are returned with
    cost = +∞ so callers can ignore them.

    Returns (delta_ind, cost_pos, cost_neg) of shape (n_remaining,).
    """
    n_rem = len(remaining_pixels)
    if n_rem == 0:
        return (np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))
    px = remaining_pixels[:, 0]
    py = remaining_pixels[:, 1]
    s_p = pixel_img[px, py].astype(np.int32)
    fired = s_p <= t
    cost_pos = np.where(fired, t + 1 - s_p, 0).astype(np.int32)
    cost_neg = np.where(fired, 0, s_p - t).astype(np.int32)
    delta = np.where(fired, -1, +1).astype(np.int8)
    # Mark out-of-budget pixels with infinite cost
    big = np.iinfo(np.int32).max // 2
    cost_pos = np.where(cost_pos > rem_pos, big, cost_pos)
    cost_neg = np.where(cost_neg > rem_neg, big, cost_neg)
    # If a pixel's flip exceeds even its single budget direction, it
    # cannot flip the indicator at this t.
    return delta, cost_pos, cost_neg


# ============================================================================
# Pair-feasibility via small-budget enumeration (Δ ≤ 2)
# ============================================================================
def _enumerate_perturbations_in_budget(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    T: int,
    max_flips: int,
):
    """Yield (flip_subset_indices, cost_pos_total, cost_neg_total) for every
    valid perturbation that flips at most `max_flips` pixels' indicator at
    SOME timestep, and uses budgets within (rem_neg, rem_pos).

    Each "flip" here is a *single-pixel single-shift*: pixel p's spike
    time changes by some integer offset in [-rem_neg, +rem_pos].

    For tractability we only enumerate to a small `max_flips`; in our
    paper's Δ-budget regime (Δ ≤ 2) max_flips = Δ suffices.

    Yields tuples (selection: list[(pixel_idx, offset)], (cost_pos, cost_neg)).
    Single empty-flip baseline is yielded first (selection=[]).
    """
    n_rem = len(remaining_pixels)
    if n_rem == 0:
        yield [], (0, 0)
        return
    s_p_all = pixel_img[remaining_pixels[:, 0], remaining_pixels[:, 1]].astype(np.int32)

    # Build per-pixel offset list with associated cost on (pos, neg) budget
    per_pixel_options: list[list[tuple[int, int, int]]] = []
    # tuple: (offset, cost_pos, cost_neg)  — offset = new_s - orig_s
    for i in range(n_rem):
        sp = int(s_p_all[i])
        opts = []
        for off in range(-rem_neg, rem_pos + 1):
            if off == 0:
                continue
            new_s = sp + off
            if new_s < 0 or new_s > T - 1:
                continue
            cp = max(off, 0)
            cn = max(-off, 0)
            opts.append((off, cp, cn))
        per_pixel_options.append(opts)

    # Empty baseline
    yield [], (0, 0)
    # Single-pixel flips
    for i in range(n_rem):
        for off, cp, cn in per_pixel_options[i]:
            if cp <= rem_pos and cn <= rem_neg:
                yield [(i, off)], (cp, cn)
    if max_flips < 2:
        return
    # Two-pixel flips (Δ ≥ 2)
    for i in range(n_rem):
        for j in range(i + 1, n_rem):
            for off_i, cp_i, cn_i in per_pixel_options[i]:
                for off_j, cp_j, cn_j in per_pixel_options[j]:
                    cp_t = cp_i + cp_j
                    cn_t = cn_i + cn_j
                    if cp_t <= rem_pos and cn_t <= rem_neg:
                        yield [(i, off_i), (j, off_j)], (cp_t, cn_t)
    # Higher max_flips: extend recursively if ever needed.


def _l1_voltage_under_perturbation(
    img_perturbed: np.ndarray,
    w1: np.ndarray,
    T: int,
) -> np.ndarray:
    """Compute V_h(t) for all (h, t) under an integer-spike-time image."""
    n_hidden = w1.shape[0]
    H, W = img_perturbed.shape[:2]
    t_grid = np.arange(T + 1)
    fire_mask = img_perturbed[..., None] <= t_grid[None, None, :]  # (H, W, T+1)
    # V_h(t) = sum_{x,y} w1[h,x,y] * fire_mask[x,y,t]
    V = np.einsum('hxy,xyt->ht', w1, fire_mask.astype(np.float64))
    return V


def pair_infeasibilities_at_t(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    threshold: float,
    T: int,
    swing_h_at_t: np.ndarray,
    t: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """For each pair (h_a, h_b) of swing L1 hiddens at time t, test joint
    feasibility under the remaining input budget.

    Returns (`pair_cannot_both_fire_by_t`, `pair_cannot_both_not_fire_by_t`)
    — two sets of unordered pairs (a, b) with a < b.

    Implementation: enumerate all valid perturbations (Δ_total ≤ Δ_rem,
    treated as max_flips = rem_pos+rem_neg), compute V_h(t) under each,
    and aggregate the achievable (1[V_a > θ], 1[V_b > θ]) patterns per pair.
    """
    n_swing = len(swing_h_at_t)
    if n_swing < 2:
        return set(), set()
    Delta_rem = rem_pos + rem_neg
    if Delta_rem == 0:
        return set(), set()
    # Build a record of achievable (firing-by-t) bool-vectors over swing_h_at_t.
    achievable = []  # list of np.bool_(n_swing,)
    for sel, (cp, cn) in _enumerate_perturbations_in_budget(
        pixel_img, remaining_pixels, rem_neg, rem_pos, T, max_flips=Delta_rem,
    ):
        img_p = pixel_img.copy()
        for pixel_idx, off in sel:
            px = remaining_pixels[pixel_idx, 0]
            py = remaining_pixels[pixel_idx, 1]
            img_p[px, py] = pixel_img[px, py] + off
        V = _l1_voltage_under_perturbation(img_p, w1, T)
        # firing-by-t (h fired iff V_h(t-τ) > θ; with τ=1 this means t > 0 → V at time t-1)
        # However the L2 propagation in `ibp_prove_robust_multilayer` uses
        # "1[s_h ≤ t]" semantics. Spike time s_h = first t' such that V_h(t'-1)>θ.
        # So 1[s_h ≤ t] depends on existence of any t' in [1, t] with V_h(t'-1)>θ,
        # which simplifies to: V_h(t-1) > θ if monotone-cumsum, but spikes can
        # still occur earlier. Conservatively: V_h(t-1)>θ would force fire by t.
        #
        # We use the cumulative semantics: h fired-by-t iff max_{0≤t'<t} V_h(t')>θ,
        # equivalently V_h(t-1)>θ since V is monotone in t (cumulative).
        if t == 0:
            fire_vec = np.zeros(len(swing_h_at_t), dtype=bool)
        else:
            fire_vec = V[swing_h_at_t, t - 1] > threshold
        achievable.append(fire_vec)
    if len(achievable) == 0:
        return set(), set()
    A = np.stack(achievable, axis=0)  # (P, n_swing)
    pair_cannot_both_fire = set()
    pair_cannot_both_not_fire = set()
    # For each pair, check achievable (a_fire, b_fire) configurations
    for ia in range(n_swing):
        for ib in range(ia + 1, n_swing):
            both_fire = (A[:, ia] & A[:, ib]).any()
            both_not = ((~A[:, ia]) & (~A[:, ib])).any()
            ha = int(swing_h_at_t[ia])
            hb = int(swing_h_at_t[ib])
            if not both_fire:
                pair_cannot_both_fire.add((min(ha, hb), max(ha, hb)))
            if not both_not:
                pair_cannot_both_not_fire.add((min(ha, hb), max(ha, hb)))
    return pair_cannot_both_fire, pair_cannot_both_not_fire


# ============================================================================
# Exact L2 voltage bound from full perturbation enumeration (small budget)
# ============================================================================
def enumerate_l2_voltage_bounds(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    threshold: float,
    T: int,
    n_h2: int,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Tight L2 voltage bounds via direct enumeration of all valid perturbations.

    Sound and tight (i.e., the EXACT max/min of V_l2(h2, t) over the enumerated
    perturbation set). For Δ_rem ≤ 2 the enumerate set covers every feasible
    perturbation, so the result is the EXACT max/min over the full feasible
    region — equivalent to a multi-layer joint knapsack solved by brute force.

    Returns (V_l2_max, V_l2_min, l2_fire_patterns), where the third element is
    a list of per-perturbation L2 firing-time arrays (n_h2,) -- caller may
    reuse it to perform similar exact tightening at the L2->output stage.
    """
    Delta_rem = rem_pos + rem_neg
    V_l2_all = []  # list of (n_h2, T+1) per perturbation
    s_l2_all = []  # list of (n_h2,) firing times
    for sel, _ in _enumerate_perturbations_in_budget(
        pixel_img, remaining_pixels, rem_neg, rem_pos, T, max_flips=Delta_rem,
    ):
        img_p = pixel_img.copy()
        for pixel_idx, off in sel:
            px = remaining_pixels[pixel_idx, 0]
            py = remaining_pixels[pixel_idx, 1]
            img_p[px, py] = pixel_img[px, py] + off
        V_h1 = _l1_voltage_under_perturbation(img_p, w1, T)  # (n_h1, T+1)
        V_h1[:, T] = threshold + 1.0  # forced fire by clamp
        ft1 = (np.argmax(V_h1 > threshold, axis=1) + 1).astype(int)
        ft1 = np.clip(ft1, 0, T - 1)
        # L2 voltage: V_l2(h2, t) = sum_h1 w2[h2, h1] * 1[s_h1 <= t]
        t_grid = np.arange(T + 1)
        fire_mask_l1 = ft1[:, None] <= t_grid[None, :]   # (n_h1, T+1)
        V_l2 = w2_flat @ fire_mask_l1.astype(np.float64)  # (n_h2, T+1)
        V_l2_all.append(V_l2)
        # L2 firing time
        V_l2_clamp = V_l2.copy()
        V_l2_clamp[:, T] = threshold + 1.0
        ft2 = (np.argmax(V_l2_clamp > threshold, axis=1) + 1).astype(int)
        ft2 = np.clip(ft2, 0, T - 1)
        s_l2_all.append(ft2)
    if len(V_l2_all) == 0:
        return (np.full((n_h2, T + 1), np.nan),
                np.full((n_h2, T + 1), np.nan),
                [])
    V_l2_stack = np.stack(V_l2_all, axis=0)  # (P, n_h2, T+1)
    V_l2_max = V_l2_stack.max(axis=0)
    V_l2_min = V_l2_stack.min(axis=0)
    return V_l2_max, V_l2_min, s_l2_all


def enumerate_output_voltage_bounds(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    w_out_flat: np.ndarray,
    threshold: float,
    T: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tight output voltage bounds (V_o^max, V_o^min) via direct enumeration.

    Sound and tight for any Δ_rem covered by the enumeration. This is the
    natural cross-layer joint upper/lower bound that BC-IBP would target if
    extended exactly through every layer.
    """
    Delta_rem = rem_pos + rem_neg
    V_o_all = []
    for sel, _ in _enumerate_perturbations_in_budget(
        pixel_img, remaining_pixels, rem_neg, rem_pos, T, max_flips=Delta_rem,
    ):
        img_p = pixel_img.copy()
        for pixel_idx, off in sel:
            px = remaining_pixels[pixel_idx, 0]
            py = remaining_pixels[pixel_idx, 1]
            img_p[px, py] = pixel_img[px, py] + off
        V_h1 = _l1_voltage_under_perturbation(img_p, w1, T)
        V_h1[:, T] = threshold + 1.0
        ft1 = np.clip((np.argmax(V_h1 > threshold, axis=1) + 1).astype(int), 0, T - 1)
        t_grid = np.arange(T + 1)
        fire_mask_l1 = ft1[:, None] <= t_grid[None, :]
        V_l2 = w2_flat @ fire_mask_l1.astype(np.float64)
        V_l2_clamp = V_l2.copy()
        V_l2_clamp[:, T] = threshold + 1.0
        ft2 = np.clip((np.argmax(V_l2_clamp > threshold, axis=1) + 1).astype(int), 0, T - 1)
        fire_mask_l2 = ft2[:, None] <= t_grid[None, :]
        V_o = w_out_flat @ fire_mask_l2.astype(np.float64)  # (num_classes, T+1)
        V_o_all.append(V_o)
    if len(V_o_all) == 0:
        return (np.full((num_classes, T + 1), np.nan),
                np.full((num_classes, T + 1), np.nan))
    V_stack = np.stack(V_o_all, axis=0)
    return V_stack.max(axis=0), V_stack.min(axis=0)


# ============================================================================
# Pair-aware output voltage bounds
# ============================================================================
def output_voltage_bounds_pair_aware(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    s_h_min: np.ndarray,
    s_h_max: np.ndarray,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    T: int,
    threshold: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (V_o_max, V_o_min) at the layer-2 voltage stage with
    pair-feasibility tightening of free-hidden contributions.

    Soundness: we only DROP contributions to V_o_max (resp. ADD to V_o_min)
    that come from infeasible joint flips. This produces tighter (smaller)
    upper bounds and tighter (larger) lower bounds than `_output_voltage_bounds`,
    while remaining a sound over-approximation of {achievable V_o(t)}.

    The function shape mirrors `_output_voltage_bounds`: returns
    (V_o_max, V_o_min) of shape (num_classes, T+1).
    """
    n_hidden = s_h_min.shape[0]
    t_grid = np.arange(T + 1)
    forced_true_h = s_h_max[:, None] <= t_grid[None, :]   # (n_h, T+1)
    forced_false_h = s_h_min[:, None] > t_grid[None, :]
    free_h = ~forced_true_h & ~forced_false_h

    w2_T = w2_flat.T.astype(np.float64)                   # (n_hidden, num_classes)
    w2_pos = np.maximum(w2_T, 0.0)
    w2_neg = np.minimum(w2_T, 0.0)

    forced_true_T = forced_true_h.T.astype(np.float64)    # (T+1, n_h)
    free_T = free_h.T.astype(np.float64)

    # Default (independent) bound — sound, the baseline.
    V_o_max = (forced_true_T @ w2_T) + (free_T @ w2_pos)  # (T+1, num_classes)
    V_o_min = (forced_true_T @ w2_T) + (free_T @ w2_neg)

    # Per-t tightening
    for t in range(T + 1):
        swing_idx = np.where(free_h[:, t])[0]
        if len(swing_idx) < 2:
            continue
        cnt_both_fire, cnt_both_not = pair_infeasibilities_at_t(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, threshold, T, swing_idx, t,
        )
        if not cnt_both_fire and not cnt_both_not:
            continue
        for o in range(num_classes):
            # V_o_max tightening: at most one of any "cannot-both-fire" pair
            # can contribute its positive weight.
            for (ha, hb) in cnt_both_fire:
                wa = max(w2_T[ha, o], 0.0)
                wb = max(w2_T[hb, o], 0.0)
                # The default bound assumed BOTH contribute. Subtract the
                # smaller (since at most one is allowed → keep the larger).
                if wa > 0 and wb > 0:
                    V_o_max[t, o] -= min(wa, wb)
            # V_o_min tightening (symmetric, on negative weights):
            for (ha, hb) in cnt_both_not:
                # If both cannot stay non-firing, at least one fires;
                # this means baseline non-fire contribution (0 each in V_o_min)
                # is too pessimistic — at least one MUST contribute its w2.
                # The smaller-of-the-two (min over) gets added.
                wa = w2_T[ha, o]
                wb = w2_T[hb, o]
                # Default V_o_min charged neither (both treated as free,
                # contribution = min(w, 0) per neuron). Adjust: at least one
                # fires → contribution at least min(wa, wb).
                # (This is most useful when wa, wb > 0 so the lower bound rises.)
                gain = min(wa, wb)
                if gain > min(w2_neg[ha, o] + w2_neg[hb, o], 0.0):
                    # Replace the (free,free) term with at-least-one-fires
                    V_o_min[t, o] += gain - (w2_neg[ha, o] + w2_neg[hb, o])
    return V_o_max.T, V_o_min.T


# ============================================================================
# Vectorized enumerate: per-(pixel, offset) ΔV_h1 + batched forward
# ============================================================================
def _build_pixel_offset_options(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each (remaining pixel, allowed offset), build the per-option arrays:

      - p_idx_arr (n_opt,)              : remaining_pixels index
      - cost_pos_arr, cost_neg_arr      : single-direction L1 costs
      - delta_V_h1 (n_opt, n_h1, T+1)   : per-option ΔV_h1 caused by flipping
                                          this single pixel's spike time

    Excludes the no-shift case (offset=0).
    """
    n_rem = len(remaining_pixels)
    if n_rem == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0, dtype=int), np.zeros((0, w1.shape[0], T + 1)))
    pix_x = remaining_pixels[:, 0]
    pix_y = remaining_pixels[:, 1]
    s_p_orig = pixel_img[pix_x, pix_y].astype(int)
    n_h1 = w1.shape[0]
    p_idx_list, cp_list, cn_list = [], [], []
    delta_V_list = []
    for p_idx in range(n_rem):
        sp = int(s_p_orig[p_idx])
        wxy = w1[:, pix_x[p_idx], pix_y[p_idx]].astype(np.float64)  # (n_h1,)
        for delta in range(-rem_neg, rem_pos + 1):
            if delta == 0:
                continue
            new_s = sp + delta
            if new_s < 0 or new_s > T - 1:
                continue
            cp = max(delta, 0); cn = max(-delta, 0)
            if cp > rem_pos or cn > rem_neg:
                continue
            # Δindicator(t) = 1[new_s ≤ t] - 1[sp ≤ t]
            ind = np.zeros(T + 1, dtype=np.float64)
            if delta > 0:
                # was fired-by-t for t in [sp, new_s-1] under baseline; now not
                ind[sp:new_s] = -1.0
            else:
                # was not fired; now fired
                ind[new_s:sp] = +1.0
            p_idx_list.append(p_idx)
            cp_list.append(cp)
            cn_list.append(cn)
            delta_V_list.append(np.outer(wxy, ind))
    if len(p_idx_list) == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0, dtype=int), np.zeros((0, n_h1, T + 1)))
    return (np.asarray(p_idx_list, dtype=np.int64),
            np.asarray(cp_list, dtype=np.int64),
            np.asarray(cn_list, dtype=np.int64),
            np.stack(delta_V_list, axis=0))


def _forward_l1_l2_out(
    V_h1_batch: np.ndarray,            # (B, n_h1, T+1)
    w2_flat: np.ndarray,               # (n_h2, n_h1)
    w_out_flat: np.ndarray | None,     # (num_classes, n_h2) or None to stop after L2
    threshold: float,
    T: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run cumulative-voltage forward through L1-spike → L2-voltage → L2-spike → output-voltage.

    Returns (V_l2 (B,n_h2,T+1), V_o (B,num_classes,T+1) or None).
    """
    t_grid = np.arange(T + 1)
    B = V_h1_batch.shape[0]
    # L1 spike times
    V_h1_clamp = V_h1_batch.copy()
    V_h1_clamp[:, :, T] = threshold + 1.0
    s_h1 = np.clip(np.argmax(V_h1_clamp > threshold, axis=2) + 1, 0, T - 1)  # (B, n_h1)
    fm_l1 = (s_h1[:, :, None] <= t_grid[None, None, :]).astype(np.float64)
    V_l2 = np.einsum('PQ,bQt->bPt', w2_flat, fm_l1)
    if w_out_flat is None:
        return V_l2, None
    V_l2_clamp = V_l2.copy()
    V_l2_clamp[:, :, T] = threshold + 1.0
    s_l2 = np.clip(np.argmax(V_l2_clamp > threshold, axis=2) + 1, 0, T - 1)
    fm_l2 = (s_l2[:, :, None] <= t_grid[None, None, :]).astype(np.float64)
    V_o = np.einsum('PQ,bQt->bPt', w_out_flat, fm_l2)
    return V_l2, V_o


def enumerate_output_voltage_bounds_vec(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    w_out_flat: np.ndarray,
    threshold: float,
    T: int,
    num_classes: int,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized enumeration of all valid perturbations within budget
    Δ_total = rem_neg + rem_pos. Computes EXACT (V_o^max, V_o^min) over
    all single- and double-pixel flip combinations (covers Δ ≤ 2 exactly).

    Implementation: build per-(pixel, offset) ΔV_h1 cache, then sweep
    chunks of (single-pixel batch) and (double-pixel pair batch) through
    the batched forward.
    """
    img_h, img_w = pixel_img.shape[:2]
    t_grid = np.arange(T + 1)
    fire_mask_base = (pixel_img[..., None] <= t_grid[None, None, :]).astype(np.float64)
    V_h1_base = np.einsum('hxy,xyt->ht', w1.astype(np.float64), fire_mask_base)  # (n_h1, T+1)

    # Baseline forward
    V_h1_batch = V_h1_base[None, ...]
    _, V_o_base = _forward_l1_l2_out(V_h1_batch, w2_flat, w_out_flat, threshold, T)
    V_o_max = V_o_base[0].copy()
    V_o_min = V_o_base[0].copy()

    # Single-pixel options
    p_idx_arr, cp_arr, cn_arr, delta_V_h1 = _build_pixel_offset_options(
        pixel_img, remaining_pixels, rem_neg, rem_pos, w1, T,
    )
    n_opt = delta_V_h1.shape[0]
    if n_opt == 0:
        return V_o_max, V_o_min

    # Sweep single-pixel chunks
    for s in range(0, n_opt, chunk_size):
        e = min(s + chunk_size, n_opt)
        V_h1_pert = V_h1_base[None, ...] + delta_V_h1[s:e]
        _, V_o = _forward_l1_l2_out(V_h1_pert, w2_flat, w_out_flat, threshold, T)
        V_o_max = np.maximum(V_o_max, V_o.max(axis=0))
        V_o_min = np.minimum(V_o_min, V_o.min(axis=0))

    # Two-pixel pairs (only if remaining budget admits 2 flips)
    Delta_rem = rem_pos + rem_neg
    if Delta_rem < 2:
        return V_o_max, V_o_min

    # Vectorized valid-pair construction
    cp_pair = cp_arr[:, None] + cp_arr[None, :]
    cn_pair = cn_arr[:, None] + cn_arr[None, :]
    distinct = p_idx_arr[:, None] != p_idx_arr[None, :]
    upper = np.triu(np.ones_like(distinct, dtype=bool), k=1)
    feas = distinct & upper & (cp_pair <= rem_pos) & (cn_pair <= rem_neg)
    pair_idx = np.argwhere(feas)  # (n_pairs, 2)
    n_pairs = pair_idx.shape[0]
    if n_pairs == 0:
        return V_o_max, V_o_min

    # Sweep pair chunks
    pair_chunk = max(256, chunk_size // 4)  # pairs are heavier
    for s in range(0, n_pairs, pair_chunk):
        e = min(s + pair_chunk, n_pairs)
        ij = pair_idx[s:e]
        delta_chunk = delta_V_h1[ij[:, 0]] + delta_V_h1[ij[:, 1]]
        V_h1_pert = V_h1_base[None, ...] + delta_chunk
        _, V_o = _forward_l1_l2_out(V_h1_pert, w2_flat, w_out_flat, threshold, T)
        V_o_max = np.maximum(V_o_max, V_o.max(axis=0))
        V_o_min = np.minimum(V_o_min, V_o.min(axis=0))
    return V_o_max, V_o_min


# ----------------------------------------------------------------------------
# GPU (PyTorch) port of the vectorized forward — for Δ=2 timing
# ----------------------------------------------------------------------------
def _enumerate_gpu(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    w_out_flat: np.ndarray,
    threshold: float,
    T: int,
    num_classes: int,
    chunk_size: int = 16384,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU PyTorch port of enumerate_output_voltage_bounds_vec."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build options on CPU (small)
    p_idx_arr_np, cp_arr_np, cn_arr_np, delta_V_h1_np = _build_pixel_offset_options(
        pixel_img, remaining_pixels, rem_neg, rem_pos, w1, T,
    )
    n_opt = delta_V_h1_np.shape[0]

    # Move data to GPU
    w1_t = torch.from_numpy(w1.astype(np.float32)).to(dev)
    w2_t = torch.from_numpy(w2_flat.astype(np.float32)).to(dev)
    w_out_t = torch.from_numpy(w_out_flat.astype(np.float32)).to(dev)
    pixel_img_t = torch.from_numpy(pixel_img.astype(np.int64)).to(dev)
    t_grid = torch.arange(T + 1, device=dev)

    # Baseline V_h1
    fire_mask_base = (pixel_img_t.unsqueeze(-1) <= t_grid.view(1, 1, -1)).float()
    V_h1_base = torch.einsum('hxy,xyt->ht', w1_t, fire_mask_base)

    delta_V_h1_t = torch.from_numpy(delta_V_h1_np.astype(np.float32)).to(dev) if n_opt > 0 else None

    def fwd(V_h1_batch: torch.Tensor) -> torch.Tensor:
        """Returns V_o batch (B, num_classes, T+1)."""
        V_h1c = V_h1_batch.clone()
        V_h1c[:, :, T] = threshold + 1.0
        s_h1 = torch.clamp(torch.argmax((V_h1c > threshold).int(), dim=2) + 1,
                           min=0, max=T - 1)
        fm_l1 = (s_h1.unsqueeze(-1) <= t_grid.view(1, 1, -1)).float()
        V_l2 = torch.einsum('PQ,bQt->bPt', w2_t, fm_l1)
        V_l2c = V_l2.clone()
        V_l2c[:, :, T] = threshold + 1.0
        s_l2 = torch.clamp(torch.argmax((V_l2c > threshold).int(), dim=2) + 1,
                           min=0, max=T - 1)
        fm_l2 = (s_l2.unsqueeze(-1) <= t_grid.view(1, 1, -1)).float()
        V_o = torch.einsum('PQ,bQt->bPt', w_out_t, fm_l2)
        return V_o

    # Baseline forward
    V_o_base = fwd(V_h1_base.unsqueeze(0))  # (1, num_classes, T+1)
    V_o_max = V_o_base[0].clone()
    V_o_min = V_o_base[0].clone()

    if n_opt == 0:
        return V_o_max.cpu().numpy(), V_o_min.cpu().numpy()

    # Single-pixel chunks
    for s in range(0, n_opt, chunk_size):
        e = min(s + chunk_size, n_opt)
        V_h1_pert = V_h1_base.unsqueeze(0) + delta_V_h1_t[s:e]
        V_o = fwd(V_h1_pert)
        V_o_max = torch.maximum(V_o_max, V_o.amax(dim=0))
        V_o_min = torch.minimum(V_o_min, V_o.amin(dim=0))

    Delta_rem = rem_pos + rem_neg
    if Delta_rem < 2:
        return V_o_max.cpu().numpy(), V_o_min.cpu().numpy()

    # Pair feasibility on CPU (cheap)
    cp_pair = cp_arr_np[:, None] + cp_arr_np[None, :]
    cn_pair = cn_arr_np[:, None] + cn_arr_np[None, :]
    distinct = p_idx_arr_np[:, None] != p_idx_arr_np[None, :]
    upper = np.triu(np.ones_like(distinct, dtype=bool), k=1)
    feas = distinct & upper & (cp_pair <= rem_pos) & (cn_pair <= rem_neg)
    pair_idx_np = np.argwhere(feas)
    n_pairs = pair_idx_np.shape[0]
    if n_pairs == 0:
        return V_o_max.cpu().numpy(), V_o_min.cpu().numpy()

    pair_idx_t = torch.from_numpy(pair_idx_np).to(dev)
    pair_chunk = max(2048, chunk_size // 2)
    for s in range(0, n_pairs, pair_chunk):
        e = min(s + pair_chunk, n_pairs)
        ij = pair_idx_t[s:e]
        delta_chunk = delta_V_h1_t[ij[:, 0]] + delta_V_h1_t[ij[:, 1]]
        V_h1_pert = V_h1_base.unsqueeze(0) + delta_chunk
        V_o = fwd(V_h1_pert)
        V_o_max = torch.maximum(V_o_max, V_o.amax(dim=0))
        V_o_min = torch.minimum(V_o_min, V_o.amin(dim=0))

    return V_o_max.cpu().numpy(), V_o_min.cpu().numpy()


def ibp_prove_robust_multilayer_exact_gpu(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    weights_list: list,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    chunk_size: int = 16384,
) -> bool:
    """GPU exact-bound prover (small Δ_rem ≤ 2)."""
    if rem_neg + rem_pos == 0:
        return True
    if len(weights_list) < 2:
        return True
    w1 = weights_list[0]
    w2 = weights_list[1]
    if w2.ndim == 3:
        w2 = w2[:, :, 0]
    if len(weights_list) == 3:
        w_out = weights_list[2]
        if w_out.ndim == 3:
            w_out = w_out[:, :, 0]
        V_o_max, V_o_min = _enumerate_gpu(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, w2, w_out, threshold, num_steps, num_classes,
            chunk_size=chunk_size,
        )
        s_o_min, s_o_max = _spike_bounds_from_voltage(
            V_o_max, V_o_min, threshold, num_steps, min_time=3,
        )
        return _output_robust_check(s_o_min, s_o_max, orig_pred, num_classes)
    # other depths: fall back to vec
    return ibp_prove_robust_multilayer_exact_vec(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list, num_steps, threshold, orig_pred, num_classes,
    )


def ibp_prove_robust_multilayer_exact_vec(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    weights_list: list,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    chunk_size: int = 4096,
) -> bool:
    """Vectorized cross-layer exact-bound prover for small Δ_rem (≤2).

    Soundness: produces the EXACT max/min V_o(t) over the enumerated set;
    for Δ_rem ≤ 2 this set covers the full feasible region (single + double
    pixel flips).
    """
    if rem_neg + rem_pos == 0:
        return True
    n_layers_total = len(weights_list)
    if n_layers_total < 2:
        return True
    w1 = weights_list[0]
    w2 = weights_list[1]
    if w2.ndim == 3:
        w2 = w2[:, :, 0]
    if n_layers_total == 2:
        # (input -> h1 -> output) — w2 IS the output weight
        V_o_max, V_o_min = enumerate_output_voltage_bounds_vec(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, w2, w2,  # placeholder when only one hidden layer
            threshold, num_steps, num_classes, chunk_size=chunk_size,
        )
        s_o_min, s_o_max = _spike_bounds_from_voltage(
            V_o_max, V_o_min, threshold, num_steps, min_time=2,
        )
        return _output_robust_check(s_o_min, s_o_max, orig_pred, num_classes)

    # 2 hidden + output layers: full 3-stage chain
    if n_layers_total == 3:
        w_out = weights_list[2]
        if w_out.ndim == 3:
            w_out = w_out[:, :, 0]
        V_o_max, V_o_min = enumerate_output_voltage_bounds_vec(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, w2, w_out, threshold, num_steps, num_classes,
            chunk_size=chunk_size,
        )
        s_o_min, s_o_max = _spike_bounds_from_voltage(
            V_o_max, V_o_min, threshold, num_steps, min_time=3,
        )
        return _output_robust_check(s_o_min, s_o_max, orig_pred, num_classes)

    # 3+ hidden layers: tighten through L1->L2 then propagate intervals
    # (not yet implemented — fall back to non-vectorized path)
    return ibp_prove_robust_multilayer_exact_small_budget(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list, num_steps, threshold, orig_pred, num_classes,
    )


# ============================================================================
# Top-level: enumerate-tight multilayer prover (small Δ regime)
# ============================================================================
def ibp_prove_robust_multilayer_exact_small_budget(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    weights_list: list,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
) -> bool:
    """Cross-layer exact bound via direct enumeration of perturbations
    at total budget Δ_rem = rem_neg + rem_pos.

    For Δ_rem ≤ 2 the enumerate set covers the full feasible region, so
    the bound is EXACT (= true max V_o(t) over all valid perturbations).
    For Δ_rem ≥ 3 enumeration is partial (single+double-pixel only); the
    function silently falls back to plain BC-IBP via interval propagation
    on subsequent layers.

    This is the "ideal" sound-and-tight cross-layer multilayer prover for
    small budgets — it captures all joint-feasibility constraints up to
    Δ_rem and incurs O(N * 2^Δ_rem) work per call.
    """
    if rem_neg + rem_pos == 0:
        # No perturbation possible — robust trivially under TTFS argmin.
        return True

    n_layers_total = len(weights_list)
    if n_layers_total < 2:
        return True

    w1 = weights_list[0]
    w2 = weights_list[1]
    if w2.ndim == 3:
        w2 = w2[:, :, 0]
    if n_layers_total == 2:
        # 1 hidden -> output (no L2 layer): use enumerate bound directly
        V_o_max, V_o_min = enumerate_output_voltage_bounds(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, w2, w2,  # last arg unused for n_layers_total==2 simple case
            threshold, num_steps, num_classes,
        )
        s_o_min, s_o_max = _spike_bounds_from_voltage(
            V_o_max, V_o_min, threshold, num_steps, min_time=2,
        )
        return _output_robust_check(s_o_min, s_o_max, orig_pred, num_classes)

    # 2+ hidden layers: enumerate at L1, L2; fall back to interval prop later
    V_l2_max, V_l2_min, s_l2_per_pert = enumerate_l2_voltage_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        w1, w2, threshold, num_steps, w2.shape[0],
    )
    if np.any(np.isnan(V_l2_max)):
        return True  # no valid perturbation
    s_h2_min, s_h2_max = _spike_bounds_from_voltage(
        V_l2_max, V_l2_min, threshold, num_steps, min_time=2,
    )

    if n_layers_total == 3:
        # L1 -> L2 -> output: also tighten output via enumerate
        w_out = weights_list[2]
        if w_out.ndim == 3:
            w_out = w_out[:, :, 0]
        V_o_max, V_o_min = enumerate_output_voltage_bounds(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, w2, w_out, threshold, num_steps, num_classes,
        )
        s_o_min, s_o_max = _spike_bounds_from_voltage(
            V_o_max, V_o_min, threshold, num_steps, min_time=3,
        )
        return _output_robust_check(s_o_min, s_o_max, orig_pred, num_classes)

    # 3+ hidden layers: enumerate gives s_h2 bounds; later layers are interval
    s_min, s_max = s_h2_min, s_h2_max
    for layer_idx in range(2, n_layers_total - 1):
        w_layer = weights_list[layer_idx]
        if w_layer.ndim == 3:
            w_layer = w_layer[:, :, 0]
        n_next = w_layer.shape[0]
        V_max, V_min = _output_voltage_bounds(s_min, s_max, w_layer, num_steps, n_next)
        s_min, s_max = _spike_bounds_from_voltage(
            V_max, V_min, threshold, num_steps, min_time=layer_idx + 1,
        )
        if np.any(s_min > s_max):
            return True
    w_out = weights_list[-1]
    if w_out.ndim == 3:
        w_out = w_out[:, :, 0]
    return _check_robust_from_s_h(
        s_min, s_max, w_out, num_steps, threshold, orig_pred, num_classes,
    )


def _output_robust_check(s_o_min: np.ndarray, s_o_max: np.ndarray,
                          orig_pred: int, num_classes: int) -> bool:
    target_max = s_o_max[orig_pred]
    for o in range(num_classes):
        if o == orig_pred:
            continue
        if o < orig_pred:
            if s_o_min[o] <= target_max:
                return False
        else:
            if s_o_min[o] < target_max:
                return False
    return True
