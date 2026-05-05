"""
LP relaxation for per-hidden-neuron tightening of s_h bounds.

Used at BnB root only (expensive: ~100ms per hidden neuron per candidate delay t*).
Operates on the SAME BnB state as IBP: a fixed pixel_img with a "remaining" pixel
set that can be perturbed within (rem_neg, rem_pos) budget.

For each hidden neuron h, we want a sound upper bound on
  s_h_max = max over feasible π of s_h(π)
where s_h(π) = first t in [1, T-1] with V_h(t-1, π) > threshold (clamped to T-1).

LP formulation (relaxation of an ILP):
  variables:
    x_p     ∈ [lo_p, hi_p]            — continuous spike time, per remaining pixel p
    y_{p,τ} ∈ [0, 1]                   — continuous indicator for [x_p ≤ τ],
                                         for τ ∈ {0, ..., t_cand - 2}
    d_p^+, d_p^-  ∈ [0, ∞)            — absolute-value slack for |x_p - orig_p|
  linkage (big-M with M = T):
    x_p ≤ τ + M (1 - y_{p,τ})
    x_p ≥ τ + 1 - M  y_{p,τ}
  monotonicity:
    y_{p,τ} ≤ y_{p,τ+1}
  budget:
    d_p^+ - d_p^- = x_p - orig_p
    Σ_p d_p^+ ≤ rem_pos
    Σ_p d_p^- ≤ rem_neg
  per-τ threshold constraint (for s_h to reach t_cand):
    V_h_base(τ) + Σ_p w_{h,p} (y_{p,τ} - baseline_fire(p,τ)) ≤ threshold
  for all τ ∈ {0, ..., t_cand - 2}

If LP is infeasible → s_h < t_cand (strict) → s_h_max ≤ t_cand - 1.

The LP is a RELAXATION of the underlying ILP, so:
  LP-infeasible ⇒ ILP-infeasible ⇒ sound upper bound on s_h_max.
LP-feasible doesn't necessarily mean ILP-feasible; bound stays loose in that case.

Complexity: each LP ~10-100 ms (CBC simplex). For each h, up to T-1 candidate t*'s.
"""
from __future__ import annotations
import numpy as np
import pulp


def lp_tighten_sh_max(
    h_indices: np.ndarray,           # (K,) hidden neurons to tighten
    pixel_img: np.ndarray,           # (28, 28) integer, current BnB state
    remaining_pixels: np.ndarray,    # (N, 2) remaining perturbable pixels
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,                  # (n_hidden, 28, 28)
    num_steps: int,
    threshold: float,
    current_s_h_max: np.ndarray,     # (n_hidden,) from IBP — starting upper bound
    time_limit: float = 0.5,
) -> np.ndarray:
    """Returns refined s_h_max for the given h_indices. Non-targeted hiddens untouched.

    The LP checks feasibility of "∃ π keeping V_h(τ) ≤ threshold for all τ ≤ t*".
    If infeasible → s_h_max = t* + 1.
    Tries t* = current_s_h_max[h] - 1 downward until LP becomes feasible
    or we hit the baseline spike time.
    """
    T = num_steps
    M = float(T)
    out = current_s_h_max.copy()

    # Baseline V_h at each time t using ALL pixels at their current values
    all_px, all_py = np.mgrid[0:28, 0:28].reshape(2, -1)
    times = pixel_img[all_px, all_py].astype(int)
    w_all = w1[:, all_px, all_py]
    t_grid = np.arange(T + 1)
    fire_mask = times[:, None] <= t_grid[None, :]
    V_h_base = w_all @ fire_mask.astype(np.float64)  # (n_hidden, T+1)

    # Precompute remaining-pixel info
    px_arr = remaining_pixels[:, 0]
    py_arr = remaining_pixels[:, 1]
    orig_arr = pixel_img[px_arr, py_arr].astype(int)
    lo_arr = np.maximum(0, orig_arr - rem_neg)
    hi_arr = np.minimum(T - 1, orig_arr + rem_pos)
    N = len(remaining_pixels)

    # For each h: solve LP iteratively
    for idx, h in enumerate(h_indices):
        h = int(h)
        w_h = w1[h, px_arr, py_arr].astype(np.float64)  # (N,)
        cur_max = int(out[h])
        # Start from smallest candidate t* we haven't disproved yet.
        # If LP-infeasible for t* = cur_max - 1, we can tighten to cur_max - 1 or lower.
        for t_cand in range(cur_max, 1, -1):  # try from cur_max down to 2
            # For s_h >= t_cand, need V_h(τ) <= threshold for τ ≤ t_cand - 2.
            t_star = t_cand - 2
            if t_star < 0:
                break  # t_cand = 1 always feasible; no need to check

            feasible = _solve_h_lp(
                h, w_h, V_h_base[h], orig_arr, lo_arr, hi_arr,
                rem_neg, rem_pos, t_star, threshold, M, N, time_limit,
            )
            if not feasible:
                # s_h < t_cand, so s_h_max ≤ t_cand - 1
                out[h] = t_cand - 1
                # Keep trying smaller t_cand in next iter — but our outer for loop
                # is already descending, so just continue.
                continue
            else:
                # Feasible — cannot disprove s_h ≥ t_cand. Stop tightening for this h.
                break

    return out


def _solve_h_lp(
    h, w_h, V_h_base_h, orig_arr, lo_arr, hi_arr,
    rem_neg, rem_pos, t_star, threshold, M, N, time_limit,
) -> bool:
    """Solve a single feasibility LP. Return True iff LP feasible."""
    prob = pulp.LpProblem(f"tighten_h{h}_t{t_star}", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{p}", lowBound=float(lo_arr[p]), upBound=float(hi_arr[p]), cat="Integer")
         for p in range(N)]
    # y_{p, τ} for τ ∈ [0, t_star] — BINARY for tighter bound
    n_tau = t_star + 1
    y = [[pulp.LpVariable(f"y_{p}_{τ}", cat="Binary")
          for τ in range(n_tau)] for p in range(N)]
    d_pos = [pulp.LpVariable(f"dp_{p}", lowBound=0.0, cat="Continuous") for p in range(N)]
    d_neg = [pulp.LpVariable(f"dn_{p}", lowBound=0.0, cat="Continuous") for p in range(N)]

    # Big-M linkage
    for p in range(N):
        for τ in range(n_tau):
            prob += x[p] - τ <= M * (1 - y[p][τ])           # y=1 ⇒ x ≤ τ
            prob += x[p] - (τ + 1) >= -M * y[p][τ]          # y=0 ⇒ x ≥ τ+1
        # Monotonicity
        for τ in range(n_tau - 1):
            prob += y[p][τ] <= y[p][τ + 1]
        # Budget linearization
        prob += d_pos[p] - d_neg[p] == x[p] - float(orig_arr[p])

    # Total budget
    prob += pulp.lpSum(d_pos) <= float(rem_pos)
    prob += pulp.lpSum(d_neg) <= float(rem_neg)

    # Threshold constraints for each τ ∈ [0, t_star]
    for τ in range(n_tau):
        baseline_fire_τ = (orig_arr <= τ).astype(np.float64)
        # V_h(τ) = V_h_base(τ) + Σ_p w_h[p] · (y[p,τ] - baseline_fire[p])
        expr = float(V_h_base_h[τ])
        for p in range(N):
            expr = expr + w_h[p] * (y[p][τ] - baseline_fire_τ[p])
        prob += expr <= float(threshold)

    # Dummy objective (feasibility)
    prob += 0

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    status = prob.solve(solver)
    return status == pulp.LpStatusOptimal


# -----------------------------------------------------------------------------
# Quick standalone test: does per-h LP tighten s_h_max for sample 9144?
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights
    from utils.dictionary_mnist import threshold
    from bnb_ibp import ibp_prove_robust_budgeted, _spike_bounds_from_voltage, _knapsack_max_benefit_vec

    cfg = CFG(
        log_name="lp_test", subtype="mnist", load_data_func=load_mnist,
        n_layer_neurons=(784, 500, 10), layer_shapes=((28, 28), (500, 1), (10, 1)),
        num_steps=5, seed=42, num_samples=1, deltas=(2,),
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist(cfg)
    sample_no = 9144
    img = images[sample_no]
    w1 = weights[0]
    w2 = weights[1][:, :, 0]
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)], dtype=int)

    # Current IBP s_h bounds at root
    T = 5
    # Reuse internals of ibp_prove_robust_budgeted to get s_h_min/max
    n_hidden = 500
    all_px, all_py = np.mgrid[0:28, 0:28].reshape(2, -1)
    times = img[all_px, all_py].astype(int)
    w_all = w1[:, all_px, all_py]
    t_grid = np.arange(T + 1)
    fire_mask = times[:, None] <= t_grid[None, :]
    V_h_base = w_all @ fire_mask.astype(np.float64)

    rn, rp = 0, 2
    remaining = all_pixels
    px_arr = remaining[:, 0]; py_arr = remaining[:, 1]
    orig_arr = img[px_arr, py_arr].astype(int)
    w_rem = w1[:, px_arr, py_arr].astype(np.float64)
    V_h_max = V_h_base.copy()
    V_h_min = V_h_base.copy()
    for t in range(T + 1):
        pos_mask = orig_arr <= t; neg_mask = ~pos_mask
        pos_costs = (t + 1 - orig_arr[pos_mask]).astype(int)
        pos_delta = -w_rem[:, pos_mask]
        neg_costs = (orig_arr[neg_mask] - t).astype(int)
        neg_delta = w_rem[:, neg_mask]
        if pos_costs.size > 0 and rp > 0:
            max_pos = _knapsack_max_benefit_vec(pos_costs, np.maximum(pos_delta, 0.0).T, rp)
            min_pos = _knapsack_max_benefit_vec(pos_costs, np.maximum(-pos_delta, 0.0).T, rp)
        else:
            max_pos = np.zeros(n_hidden); min_pos = np.zeros(n_hidden)
        if neg_costs.size > 0 and rn > 0:
            max_neg = _knapsack_max_benefit_vec(neg_costs, np.maximum(neg_delta, 0.0).T, rn)
            min_neg = _knapsack_max_benefit_vec(neg_costs, np.maximum(-neg_delta, 0.0).T, rn)
        else:
            max_neg = np.zeros(n_hidden); min_neg = np.zeros(n_hidden)
        V_h_max[:, t] += max_pos + max_neg
        V_h_min[:, t] -= min_pos + min_neg

    s_h_min, s_h_max = _spike_bounds_from_voltage(V_h_max, V_h_min, threshold, T, min_time=1)

    # Find top 5 hidden neurons with widest s_h range
    width = s_h_max - s_h_min
    top_h = np.argsort(width)[::-1][:5]
    print(f"top-5 widest h: {top_h.tolist()}")
    print(f"before LP: s_h_max for these = {s_h_max[top_h].tolist()}")
    print(f"           s_h_min for these = {s_h_min[top_h].tolist()}")

    # LP tighten
    tic = time.time()
    s_h_max_tight = lp_tighten_sh_max(
        top_h, img.astype(int), remaining, rn, rp, w1, T, threshold, s_h_max, time_limit=1.0,
    )
    elapsed = time.time() - tic
    print(f"after LP ({elapsed:.2f}s): s_h_max for these = {s_h_max_tight[top_h].tolist()}")
