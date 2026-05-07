"""Legacy CLPB (Cross-Layer Pair Bound) — NOT used by current paper.

This file preserves the polynomial-time pair-cost code (2D knapsack DP over
4-quadrant Pareto fronts) we explored before settling on CLEB. CLPB only
recovered 1/10 of the Δ=1 multi-hidden samples and 0/10 at Δ=2 in our
benchmarks, so it was dropped from the paper. The exact-enumeration CLEB
in `bnb_ibp_pair.py` (paper §5.2 / Theorem 4) supersedes everything here.

Kept for reference / future research only.
"""
from __future__ import annotations

import numpy as np

from bnb_ibp import (
    _compute_budgeted_s_h_bounds,
    _spike_bounds_from_voltage,
    _output_voltage_bounds,
    _check_robust_from_s_h,
)
from bnb_ibp_pair import output_voltage_bounds_pair_aware


# ============================================================================
# Top-level: multilayer BC-IBP v2 with pair-aware tightening
# ============================================================================
# ============================================================================
# True polynomial-time pair-cost via 2D knapsack DP over Pareto fronts
# ============================================================================
def _per_pixel_options_at_t(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    T: int,
    t: int,
    w_a: np.ndarray,    # (n_remaining,) weights into hidden a (1D, indexed by remaining_pixels)
    w_b: np.ndarray,    # (n_remaining,) weights into hidden b
):
    """For each remaining pixel and each allowed offset, return:
        (cp, cn, dV_a_at_t, dV_b_at_t) tuples — what flipping pixel p by
        offset δ does to (V_a(t), V_b(t)) and to budgets.
    Shape (n_options, 4) numpy array of float, plus a (n_options,) array of pixel indices.
    """
    n_rem = len(remaining_pixels)
    s_p_orig = pixel_img[remaining_pixels[:, 0], remaining_pixels[:, 1]].astype(int)
    p_idx_list, opts = [], []
    for p_idx in range(n_rem):
        sp = int(s_p_orig[p_idx])
        for delta in range(-rem_neg, rem_pos + 1):
            if delta == 0: continue
            new_s = sp + delta
            if new_s < 0 or new_s > T - 1: continue
            cp = max(delta, 0); cn = max(-delta, 0)
            if cp > rem_pos or cn > rem_neg: continue
            # Δindicator at time t = 1[new_s ≤ t] - 1[sp ≤ t]
            new_fired = new_s <= t
            old_fired = sp <= t
            d_ind = (1 if new_fired else 0) - (1 if old_fired else 0)
            if d_ind == 0:
                continue  # this offset doesn't change V at this t — skip
            dV_a = float(w_a[p_idx]) * d_ind
            dV_b = float(w_b[p_idx]) * d_ind
            p_idx_list.append(p_idx)
            opts.append((cp, cn, dV_a, dV_b))
    if not opts:
        return np.zeros((0,), dtype=int), np.zeros((0, 4), dtype=np.float64)
    return np.asarray(p_idx_list, dtype=int), np.asarray(opts, dtype=np.float64)


def _pareto_prune(points: np.ndarray, mode: str) -> np.ndarray:
    """Prune points (n, 2) to its Pareto front.

    mode='max' keeps points not dominated in (max V_a, max V_b) sense
         (a dominated by b iff V_a(b) >= V_a(a) AND V_b(b) >= V_b(a) AND strict in one).
    mode='min' keeps non-dominated in (min, min) sense.
    For our pair-cost feasibility we need BOTH max-Pareto (for fire targets)
    AND min-Pareto (for not-fire targets), so we keep points along ALL FOUR
    quadrants — i.e. just dedup.
    """
    if points.shape[0] <= 1:
        return points
    # Round to mitigate floating-point noise then dedup
    rounded = np.round(points, 6)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]


def _pareto_keep(points: np.ndarray, sign_a: int, sign_b: int) -> np.ndarray:
    """Pareto-prune (n, 2) points keeping only non-dominated under
    objective (sign_a · V_a, sign_b · V_b) — both maximised.

    A point p dominates q iff sign_a·V_a(p) ≥ sign_a·V_a(q),
    sign_b·V_b(p) ≥ sign_b·V_b(q), and at least one strict.
    Returns reduced points array (k, 2).
    """
    if points.shape[0] == 0:
        return points
    obj = points * np.array([sign_a, sign_b])  # (n, 2) we want to maximise
    # Sort by first axis descending
    order = np.argsort(-obj[:, 0], kind='stable')
    obj = obj[order]
    pts = points[order]
    keep = []
    best_b = -np.inf
    for i in range(pts.shape[0]):
        if obj[i, 1] > best_b:
            keep.append(i)
            best_b = obj[i, 1]
    return pts[keep]


def _solve_pair_2d_knapsack_sound(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    threshold: float,
    T: int,
    h_a: int,
    h_b: int,
    t: int,
):
    """Compute the 4 pair-cost targets for (h_a, h_b) at time t via a
    Pareto-frontier 2D knapsack DP over remaining pixels.

    Returns dict with keys 'fire_fire', 'fire_not', 'not_fire', 'not_not';
    each value is the minimum (cp+cn) sum such that the corresponding
    (V_a, V_b) condition is reachable, or +∞ if infeasible.

    Complexity: O(|opts| · |budget_states| · |Pareto|) per pair.
    """
    """Sound 2D-knapsack pair-cost via 4-quadrant Pareto fronts.

    For each (cp_used, cn_used) state we maintain four Pareto fronts:
      MM = max V_a, max V_b   — for testing 'fire_fire'
      Mm = max V_a, min V_b   — for 'fire_not'
      mM = min V_a, max V_b   — for 'not_fire'
      mm = min V_a, min V_b   — for 'not_not'
    Each front is pruned to non-dominated points under its respective
    (signed) objective. Dropping dominated points is sound because for any
    target, only the corresponding extremal corner of the front matters.

    Returns dict of 4 minimum-budget feasibility costs (∞ if infeasible
    within (rem_pos, rem_neg)).
    """
    INF = rem_neg + rem_pos + 1
    n_rem = len(remaining_pixels)
    out = {'fire_fire': INF, 'fire_not': INF, 'not_fire': INF, 'not_not': INF}
    if n_rem < 0:
        return out

    # "Fires-by-t" semantics: h has fired by voltage-time t iff
    # V_h(t-1) > θ (cumulative voltage exceeds threshold at index t-1).
    # For t == 0 nothing has fired yet (synaptic delay).
    if t == 0:
        out['not_not'] = 0  # baseline pattern: neither has fired
        return out

    t_index = t - 1  # the index at which V is checked

    H, W = pixel_img.shape[:2]
    fire_mask_base_t = (pixel_img <= t_index).astype(np.float64)
    V_a_base = float((w1[h_a] * fire_mask_base_t).sum())
    V_b_base = float((w1[h_b] * fire_mask_base_t).sum())

    w_a = w1[h_a, remaining_pixels[:, 0], remaining_pixels[:, 1]].astype(np.float64)
    w_b = w1[h_b, remaining_pixels[:, 0], remaining_pixels[:, 1]].astype(np.float64)
    s_p_orig = pixel_img[remaining_pixels[:, 0], remaining_pixels[:, 1]].astype(int)

    # State: dict[(cp, cn)] -> {'MM','Mm','mM','mm'} -> (k_q, 2) array
    def empty_quads():
        zero = np.array([[0.0, 0.0]])
        return {'MM': zero.copy(), 'Mm': zero.copy(),
                'mM': zero.copy(), 'mm': zero.copy()}
    state = {(0, 0): empty_quads()}

    quad_signs = {'MM': (+1, +1), 'Mm': (+1, -1), 'mM': (-1, +1), 'mm': (-1, -1)}

    def merge_into(target: dict, key: tuple, pts: np.ndarray):
        if key not in target:
            target[key] = {q: pts.copy() for q in quad_signs}
            for q, (sa, sb) in quad_signs.items():
                target[key][q] = _pareto_keep(target[key][q], sa, sb)
            return
        # merge — append to each quadrant and re-prune
        for q, (sa, sb) in quad_signs.items():
            combined = np.vstack([target[key][q], pts])
            target[key][q] = _pareto_keep(combined, sa, sb)

    # Early-stop on hitting all 4 targets
    def update_targets(state: dict):
        for (cp, cn), quads in state.items():
            cost = cp + cn
            for q, (sa, sb) in quad_signs.items():
                pts = quads[q]
                if pts.shape[0] == 0: continue
                V_a_tot = pts[:, 0] + V_a_base
                V_b_tot = pts[:, 1] + V_b_base
                a_fire = V_a_tot > threshold
                b_fire = V_b_tot > threshold
                if q == 'MM':
                    if (a_fire & b_fire).any() and cost < out['fire_fire']:
                        out['fire_fire'] = cost
                if q == 'Mm':
                    if (a_fire & ~b_fire).any() and cost < out['fire_not']:
                        out['fire_not'] = cost
                if q == 'mM':
                    if (~a_fire & b_fire).any() and cost < out['not_fire']:
                        out['not_fire'] = cost
                if q == 'mm':
                    if (~a_fire & ~b_fire).any() and cost < out['not_not']:
                        out['not_not'] = cost

    update_targets(state)

    for p_idx in range(n_rem):
        sp = int(s_p_orig[p_idx])
        opts = [(0, 0, 0.0, 0.0)]  # skip option
        for delta in range(-rem_neg, rem_pos + 1):
            if delta == 0: continue
            new_s = sp + delta
            if new_s < 0 or new_s > T - 1: continue
            cp = max(delta, 0); cn = max(-delta, 0)
            if cp > rem_pos or cn > rem_neg: continue
            # Δ-indicator at the firing-by-t check time t_index
            d_ind = (1 if new_s <= t_index else 0) - (1 if sp <= t_index else 0)
            opts.append((cp, cn, float(w_a[p_idx]) * d_ind, float(w_b[p_idx]) * d_ind))

        new_state = {}
        for (cp_u, cn_u), quads in state.items():
            for (cp_o, cn_o, dV_a, dV_b) in opts:
                ncp = cp_u + cp_o; ncn = cn_u + cn_o
                if ncp > rem_pos or ncn > rem_neg: continue
                key = (ncp, ncn)
                shift = np.array([[dV_a, dV_b]])
                # Stack & prune per quadrant
                if key not in new_state:
                    new_state[key] = {q: np.empty((0, 2)) for q in quad_signs}
                for q, (sa, sb) in quad_signs.items():
                    src = quads[q]
                    shifted = src + shift
                    combined = np.vstack([new_state[key][q], shifted])
                    new_state[key][q] = _pareto_keep(combined, sa, sb)
        state = new_state
        update_targets(state)
    return out


# Keep a tiny wrapper using the old name so calls don't break
_solve_pair_2d_knapsack = _solve_pair_2d_knapsack_sound


def pair_costs_at_t_2dp(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    threshold: float,
    T: int,
    swing_h_at_t: np.ndarray,
    t: int,
) -> dict:
    """Polynomial-time pair-cost via 2D knapsack DP per pair (no enumerate).

    Returns a dict mapping (h_a, h_b) (h_a < h_b) to the same 4-target
    cost dict as `pair_costs_at_t` (enumerate-based). Used as the
    actual pair-cost engine in CLPB.
    """
    out = {}
    n = len(swing_h_at_t)
    if n < 2:
        return out
    for ia in range(n):
        for ib in range(ia + 1, n):
            ha = int(swing_h_at_t[ia]); hb = int(swing_h_at_t[ib])
            costs = _solve_pair_2d_knapsack_sound(
                pixel_img, remaining_pixels, rem_neg, rem_pos,
                w1, threshold, T, ha, hb, t,
            )
            out[(min(ha, hb), max(ha, hb))] = costs
    return out


# ============================================================================
# Polynomial-time pair-feasibility prover (CLPB — Cross-Layer Pair Bound)
# ============================================================================
def pair_costs_at_t(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    threshold: float,
    T: int,
    swing_h_at_t: np.ndarray,
    t: int,
) -> dict:
    """For each pair (h_a, h_b) of swing hiddens at time t, return
    min joint flip-cost (in Δ=rem_neg+rem_pos) for the four target combinations:
      ((fire, fire), (fire, not), (not, fire), (not, not))

    Costs are in {0,1,...,Δ_total} ∪ {∞} where ∞ means infeasible within budget.

    Implementation: enumerate all valid perturbations within
    (rem_pos, rem_neg), compute layer-1 firing pattern at t-1 for each,
    take min cost achieving each (target_a, target_b) per pair.
    """
    n_swing = len(swing_h_at_t)
    if n_swing < 2:
        return {}
    Delta_rem = rem_pos + rem_neg
    INF = Delta_rem + 1

    # achievable: list of (cost_total, fire_vec_for_swing)
    samples = []  # (total_cost, fire_vec)
    for sel, (cp, cn) in _enumerate_perturbations_in_budget(
        pixel_img, remaining_pixels, rem_neg, rem_pos, T, max_flips=Delta_rem,
    ):
        img_p = pixel_img.copy()
        for pixel_idx, off in sel:
            px = remaining_pixels[pixel_idx, 0]
            py = remaining_pixels[pixel_idx, 1]
            img_p[px, py] = pixel_img[px, py] + off
        V = _l1_voltage_under_perturbation(img_p, w1, T)
        if t == 0:
            fire = np.zeros(len(swing_h_at_t), dtype=bool)
        else:
            fire = V[swing_h_at_t, t - 1] > threshold
        samples.append((cp + cn, fire))
    if not samples:
        return {}

    out = {}
    for ia in range(n_swing):
        for ib in range(ia + 1, n_swing):
            best_costs = [INF, INF, INF, INF]  # (FF, FN, NF, NN)
            for cost, fire in samples:
                a = bool(fire[ia]); b = bool(fire[ib])
                key = (1 if a else 0) * 2 + (1 if b else 0)
                if cost < best_costs[key]:
                    best_costs[key] = cost
            ha = int(swing_h_at_t[ia]); hb = int(swing_h_at_t[ib])
            out[(min(ha, hb), max(ha, hb))] = {
                'fire_fire':       best_costs[(1) * 2 + 1],
                'fire_not':        best_costs[(1) * 2 + 0],
                'not_fire':        best_costs[(0) * 2 + 1],
                'not_not':         best_costs[(0) * 2 + 0],
            }
    return out


def _max_weight_subset_with_pair_constraints(
    weights: np.ndarray,        # (n,) float — gains for each candidate hidden
    forbidden_pairs: list,      # list of (i, j) index pairs that cannot both be in S
) -> float:
    """Max weight independent set on the forbidden-pair graph.

    For small n (≤ 30), solves exactly via brute force enumerate;
    for larger n, uses a greedy approximation that is sound (lower bound) only
    when paired with an upper-bound caller — here we use it as a tighter
    upper bound, so we need EXACT.

    Returns the max sum of weights over subsets that contain no forbidden pair.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    if not forbidden_pairs:
        return float(np.maximum(weights, 0).sum())
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for i, j in forbidden_pairs:
        adj[i].add(j); adj[j].add(i)
    # For small n, exact bitmask DP / enumerate
    if n <= 24:
        best = 0.0
        for mask in range(1 << n):
            # Check independent
            valid = True
            v = 0.0
            for i in range(n):
                if mask & (1 << i):
                    if adj[i] & {j for j in range(n) if (mask >> j) & 1 and j != i}:
                        valid = False; break
                    if weights[i] > 0:
                        v += weights[i]
            if valid and v > best:
                best = v
        return best
    # Larger n: weighted-greedy (sound only as approx). For our setting we cap
    # n via top-K filtering before this is called.
    order = np.argsort(-np.maximum(weights, 0))
    chosen = set()
    forbid = set()
    s = 0.0
    for idx in order:
        if idx in forbid:
            continue
        if weights[idx] <= 0:
            break
        chosen.add(idx)
        forbid |= adj[idx]
        s += weights[idx]
    return s


def output_voltage_bounds_pair_polytight(
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
    n_h2: int,
    top_k_swing: int = 22,
) -> tuple[np.ndarray, np.ndarray]:
    """L1->L2 voltage bounds with full pair-constraint MaxIS tightening.

    Formulation
    -----------
    Each swing layer-1 hidden h has a desired post-perturbation firing status:
        σ_h_max = +1 (fire) if w_{1,h,h2} > 0 else 0 (not_fire)   — for V_max
        σ_h_min = 0       (not_fire) if w_{1,h,h2} > 0 else 1 (fire) — for V_min
    Achieving the desired pattern at all swing h gives the loose baseline bound.
    A pair (h_a, h_b) is "deviation-forcing" if its joint desired pattern has
    pair-cost > Δ_rem; in that case at most one of (x_a, x_b) can equal 1
    (achieve desired). We solve a max-weight independent set on the
    deviation-forcing graph with weights |w_{1,h,h2}|, restricted to the
    top-K swing entries by |w| for tractability.

    The bound is sound: every admissible perturbation realises some pair-feasible
    pattern, so its V_h2 contribution is bounded by the MaxIS objective.
    """
    n_h1 = s_h_min.shape[0]
    t_grid = np.arange(T + 1)
    forced_true_h = s_h_max[:, None] <= t_grid[None, :]
    free_h = ~forced_true_h & ~(s_h_min[:, None] > t_grid[None, :])
    w2_T = w2_flat.T.astype(np.float64)  # (n_h1, n_h2)
    forced_true_T = forced_true_h.T.astype(np.float64)
    free_T = free_h.T.astype(np.float64)
    # Default sound (loose) bounds
    V_max = (forced_true_T @ w2_T) + (free_T @ np.maximum(w2_T, 0.0))
    V_min = (forced_true_T @ w2_T) + (free_T @ np.minimum(w2_T, 0.0))

    Delta_rem = rem_pos + rem_neg
    if Delta_rem == 0:
        return V_max.T, V_min.T

    for t in range(T + 1):
        swing_idx_full = np.where(free_h[:, t])[0]
        if len(swing_idx_full) < 2:
            continue
        # Cache pair costs once per t (independent of h_2). Polynomial 2D-DP.
        pair_costs = pair_costs_at_t_2dp(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            w1, threshold, T, swing_idx_full, t,
        )
        if not pair_costs:
            continue

        for o in range(n_h2):
            ws = w2_T[swing_idx_full, o]                 # (|swing|,) weights to h2=o
            absw = np.abs(ws)
            order = np.argsort(-absw)
            top = order[:top_k_swing]
            top = top[absw[top] > 0]
            if len(top) < 2:
                continue
            # Desired sigma for V_max: 'fire' if w>0, 'not_fire' if w<0
            # Desired sigma for V_min (max -V_h2 = min V_h2): 'fire' if w<0, 'not_fire' if w>0
            sigma_des_max = (ws[top] > 0)   # True = want fire
            sigma_des_min = (ws[top] < 0)
            weights_top = absw[top]

            def pair_cost_lookup(sigma_des, li, lj, pair_costs):
                gi = top[li]; gj = top[lj]
                h_a = int(swing_idx_full[gi]); h_b = int(swing_idx_full[gj])
                key = (min(h_a, h_b), max(h_a, h_b))
                if key not in pair_costs:
                    return Delta_rem  # if missing, treat feasible (conservative)
                # Desired (σ_a_des, σ_b_des) — Note: pair_costs key is (smaller_h, larger_h),
                # so the per-hidden role (a vs b) needs to align with that ordering.
                if h_a < h_b:
                    s_a_des = sigma_des[li]; s_b_des = sigma_des[lj]
                else:
                    s_a_des = sigma_des[lj]; s_b_des = sigma_des[li]
                lookup = ('fire' if s_a_des else 'not') + '_' + ('fire' if s_b_des else 'not')
                # 'fire_not' / 'not_fire' / 'fire_fire' / 'not_not'
                key_str = ('fire_' if s_a_des else 'not_') + ('fire' if s_b_des else 'not')
                return pair_costs[key].get(key_str, Delta_rem + 1)

            def forbidden_pairs(sigma_des, n_top, pair_costs):
                fb = []
                for li in range(n_top):
                    for lj in range(li + 1, n_top):
                        if pair_cost_lookup(sigma_des, li, lj, pair_costs) > Delta_rem:
                            fb.append((li, lj))
                return fb

            # V_max tightening
            fb_max = forbidden_pairs(sigma_des_max, len(top), pair_costs)
            if fb_max:
                bound_max = _max_weight_subset_with_pair_constraints(weights_top, fb_max)
                drop_max = weights_top.sum() - bound_max
                if drop_max > 0:
                    V_max[t, o] -= drop_max

            # V_min tightening (analogous on the min side)
            fb_min = forbidden_pairs(sigma_des_min, len(top), pair_costs)
            if fb_min:
                bound_min = _max_weight_subset_with_pair_constraints(weights_top, fb_min)
                drop_min = weights_top.sum() - bound_min
                if drop_min > 0:
                    V_min[t, o] += drop_min
    return V_max.T, V_min.T


def ibp_prove_robust_multilayer_pair_polytight(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    weights_list: list,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    top_k_swing: int = 24,
) -> bool:
    """Multi-hidden BC-IBP with polynomial-time pair-constraint tightening
    at the layer-1->layer-2 voltage step (CLPB).

    Sound by construction; strictly tighter than per-h-independent
    interval propagation whenever pair-infeasibility constraints are
    discoverable within Δ_rem (Theorem~?).
    """
    s_h_min, s_h_max = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list[0], num_steps, threshold,
    )
    if np.any(s_h_min > s_h_max):
        return True
    n_layers_total = len(weights_list)
    if n_layers_total == 2:
        # 1 hidden -> output: pair tightening at the output bound directly
        w_out = weights_list[1]
        if w_out.ndim == 3:
            w_out = w_out[:, :, 0]
        V_o_max, V_o_min = output_voltage_bounds_pair_polytight(
            pixel_img, remaining_pixels, rem_neg, rem_pos,
            s_h_min, s_h_max, weights_list[0], w_out,
            num_steps, threshold, num_classes, top_k_swing=top_k_swing,
        )
        s_o_min, s_o_max = _spike_bounds_from_voltage(
            V_o_max, V_o_min, threshold, num_steps, min_time=2,
        )
        return _output_robust_check(s_o_min, s_o_max, orig_pred, num_classes)

    # 2+ hidden layers
    w2 = weights_list[1]
    if w2.ndim == 3:
        w2 = w2[:, :, 0]
    n_h2 = w2.shape[0]
    V_h2_max, V_h2_min = output_voltage_bounds_pair_polytight(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        s_h_min, s_h_max, weights_list[0], w2,
        num_steps, threshold, n_h2, top_k_swing=top_k_swing,
    )
    s_h2_min, s_h2_max = _spike_bounds_from_voltage(
        V_h2_max, V_h2_min, threshold, num_steps, min_time=2,
    )
    if np.any(s_h2_min > s_h2_max):
        return True
    # Subsequent layers: plain interval propagation
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


# ============================================================================
# Top-level: multilayer BC-IBP v2 with pair-aware tightening (legacy)
# ============================================================================
def ibp_prove_robust_multilayer_v2(
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
    """Multi-hidden BC-IBP with cross-layer pair-feasibility tightening.

    Tighter than `ibp_prove_robust_multilayer` for the L1→L2 voltage step;
    falls back to plain interval propagation for any subsequent layers.
    """
    s_h_min, s_h_max = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list[0], num_steps, threshold,
    )
    if np.any(s_h_min > s_h_max):
        return True
    n_layers_total = len(weights_list)

    # If only 1 hidden -> output, fall through to standard pair-aware output check
    if n_layers_total == 2:
        # weights_list[1]: (num_classes, n_h, 1)
        w2 = weights_list[1]
        if w2.ndim == 3:
            w2 = w2[:, :, 0]
        return _check_robust_from_s_h(
            s_h_min, s_h_max, w2, num_steps, threshold, orig_pred, num_classes,
        )

    # 2 or more hidden layers: tighten the L1->L2 voltage step
    w2 = weights_list[1]
    if w2.ndim == 3:
        w2 = w2[:, :, 0]  # (n_h2, n_h1)
    n_h2 = w2.shape[0]

    V_h2_max, V_h2_min = output_voltage_bounds_pair_aware(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        s_h_min, s_h_max, weights_list[0], w2, num_steps, threshold, n_h2,
    )
    s_h2_min, s_h2_max = _spike_bounds_from_voltage(
        V_h2_max, V_h2_min, threshold, num_steps, min_time=2,
    )
    if np.any(s_h2_min > s_h2_max):
        return True

    # Subsequent layers: plain interval propagation
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
