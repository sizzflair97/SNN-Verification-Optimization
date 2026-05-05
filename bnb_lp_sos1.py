"""
Per-hidden-neuron LP/ILP tightening via SOS1-style formulation (no big-M).

For each pixel p, its perturbed spike time is chosen from a finite set
  S_p = {max(0, orig_p - rem_neg), ..., min(T-1, orig_p + rem_pos)}
Variable z_{p, s} ∈ {0, 1} (or [0, 1] in LP relaxation) indicates selection.
Constraint: Σ_s z_{p, s} = 1  (exactly one spike time per pixel).

Contribution to V_h(τ): w_{h,p} · Σ_{s ≤ τ} z_{p, s}
Budget: Σ_p Σ_{s > orig_p} (s - orig_p) · z_{p, s} ≤ rem_pos  (positive shifts)
        Σ_p Σ_{s < orig_p} (orig_p - s) · z_{p, s} ≤ rem_neg  (negative shifts)

For each hidden h and candidate delay t_cand, check feasibility of
  V_h(τ) ≤ threshold  for all τ ∈ [0, t_cand - 2]

If infeasible → s_h < t_cand → tighten s_h_max[h] to t_cand - 1.

LP relaxation (z continuous): infeasibility still gives a sound upper bound on
s_h_max since it's a relaxation of the ILP.
ILP gives tight bound but is slower.
"""
from __future__ import annotations
import numpy as np
import pulp


def lp_tighten_sh_max_sos1(
    h_indices: np.ndarray,
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    num_steps: int,
    threshold: float,
    current_s_h_max: np.ndarray,
    use_ilp: bool = False,           # False → LP, True → ILP (slower, tighter)
    time_limit: float = 1.0,
) -> np.ndarray:
    """Returns refined s_h_max for the given h_indices."""
    T = num_steps
    out = current_s_h_max.copy()

    # Baseline V_h(τ) using ALL pixels at current values (fixed + remaining-at-orig)
    all_px, all_py = np.mgrid[0:28, 0:28].reshape(2, -1)
    times = pixel_img[all_px, all_py].astype(int)
    w_all = w1[:, all_px, all_py]
    t_grid = np.arange(T + 1)
    fire_mask = times[:, None] <= t_grid[None, :]
    V_h_base = w_all @ fire_mask.astype(np.float64)

    px_arr = remaining_pixels[:, 0]
    py_arr = remaining_pixels[:, 1]
    orig_arr = pixel_img[px_arr, py_arr].astype(int)
    lo_arr = np.maximum(0, orig_arr - rem_neg)
    hi_arr = np.minimum(T - 1, orig_arr + rem_pos)
    N = len(remaining_pixels)

    # Precompute per-pixel allowed spike times (as integer lists)
    # AND the z-index map.
    allowed = [list(range(int(lo_arr[p]), int(hi_arr[p]) + 1)) for p in range(N)]

    # Iterate hidden neurons
    for h in h_indices:
        h = int(h)
        w_h = w1[h, px_arr, py_arr].astype(np.float64)
        cur_max = int(out[h])

        # Try tightening from cur_max down to 2
        for t_cand in range(cur_max, 1, -1):
            t_star = t_cand - 2
            if t_star < 0:
                break
            feasible = _solve_h_lp_sos1(
                h, w_h, V_h_base[h], orig_arr, allowed,
                rem_neg, rem_pos, t_star, threshold, N, use_ilp, time_limit,
            )
            if not feasible:
                out[h] = t_cand - 1
                # Continue descending — maybe even smaller t_cand is also infeasible
            else:
                break

    return out


def _solve_h_lp_sos1(
    h, w_h, V_h_base_h, orig_arr, allowed,
    rem_neg, rem_pos, t_star, threshold, N, use_ilp, time_limit,
) -> bool:
    """Feasibility of "∃ π with V_h(τ) ≤ threshold for τ ∈ [0, t_star]" using SOS1 LP/ILP."""
    prob = pulp.LpProblem(f"sos1_h{h}_ts{t_star}", pulp.LpMinimize)
    cat = "Binary" if use_ilp else "Continuous"

    # Variables: z[p][s] for s in allowed[p]
    z = []
    for p in range(N):
        zp = {}
        for s in allowed[p]:
            zp[s] = pulp.LpVariable(f"z_{p}_{s}", lowBound=0.0, upBound=1.0, cat=cat)
        z.append(zp)

    # SOS1: each pixel chooses exactly one spike time
    for p in range(N):
        prob += pulp.lpSum(z[p].values()) == 1

    # Budget constraints
    pos_expr = pulp.LpAffineExpression()
    neg_expr = pulp.LpAffineExpression()
    for p in range(N):
        op = int(orig_arr[p])
        for s, var in z[p].items():
            if s > op:
                pos_expr += (s - op) * var
            elif s < op:
                neg_expr += (op - s) * var
    prob += pos_expr <= float(rem_pos)
    prob += neg_expr <= float(rem_neg)

    # Threshold constraints for each τ ∈ [0, t_star]
    for tau in range(t_star + 1):
        baseline_fire_p = (orig_arr <= tau).astype(np.float64)
        # V_h(τ) = V_h_base(τ) + Σ_p w_h[p] · (Σ_{s ≤ τ} z[p][s] - baseline_fire_p)
        expr = float(V_h_base_h[tau])
        for p in range(N):
            fires_now = pulp.lpSum(var for s, var in z[p].items() if s <= tau)
            expr = expr + w_h[p] * (fires_now - baseline_fire_p[p])
        prob += expr <= float(threshold)

    prob += 0  # dummy objective

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    status = prob.solve(solver)
    return status == pulp.LpStatusOptimal


# -----------------------------------------------------------------------------
# Standalone test on sample 9144
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights
    from utils.dictionary_mnist import threshold
    from bnb_ibp import _spike_bounds_from_voltage, _knapsack_max_benefit_vec

    cfg = CFG(
        log_name="sos1_test", subtype="mnist", load_data_func=load_mnist,
        n_layer_neurons=(784, 500, 10), layer_shapes=((28, 28), (500, 1), (10, 1)),
        num_steps=5, seed=42, num_samples=1, deltas=(2,),
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, *_ = load_mnist(cfg)
    img = images[9144]
    w1 = weights[0]
    w2 = weights[1][:, :, 0]
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)], dtype=int)
    T = 5
    n_hidden = 500

    # Compute IBP bounds at root for rn=0, rp=2 (one of the failing splits)
    all_px, all_py = np.mgrid[0:28, 0:28].reshape(2, -1)
    times = img[all_px, all_py].astype(int)
    w_all = w1[:, all_px, all_py]
    t_grid = np.arange(T + 1)
    fire_mask = times[:, None] <= t_grid[None, :]
    V_h_base = w_all @ fire_mask.astype(np.float64)

    rn, rp = 0, 2
    px_arr = all_pixels[:, 0]; py_arr = all_pixels[:, 1]
    orig_arr = img[px_arr, py_arr].astype(int)
    w_rem = w1[:, px_arr, py_arr].astype(np.float64)
    V_h_max = V_h_base.copy(); V_h_min = V_h_base.copy()
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
    width = s_h_max - s_h_min
    top_h = np.argsort(width)[::-1][:10]
    print(f"(sample 9144, rn={rn}, rp={rp})")
    print(f"top-10 widest h: {top_h.tolist()}")
    print(f"before:  s_h_min = {s_h_min[top_h].tolist()}")
    print(f"         s_h_max = {s_h_max[top_h].tolist()}")

    # Full pipeline: ILP tighten ALL hidden (with cached top-k first, then others)
    # to test how much overall robustness improves.
    print()
    print("=== Full pipeline: ILP-tighten all n_h=500 hidden then check robustness ===")
    tic = time.time()
    # Try tightening all hidden, but limit to those with s_h_max >= 2 (room to improve)
    all_targets = np.arange(n_hidden)
    s_h_max_tight = lp_tighten_sh_max_sos1(
        all_targets, img.astype(int), all_pixels, rn, rp, w1, T, threshold,
        s_h_max, use_ilp=True, time_limit=0.3,
    )
    ilp_all_elapsed = time.time() - tic
    n_tightened = (s_h_max_tight < s_h_max).sum()
    print(f"ILP-tightened {n_tightened}/{n_hidden} hidden neurons in {ilp_all_elapsed:.1f}s")

    # Recompute output bounds using tightened s_h_max
    from bnb_ibp import _output_voltage_bounds
    V_o_max_orig, V_o_min_orig = _output_voltage_bounds(s_h_min, s_h_max, w2, T, 10)
    V_o_max_tight, V_o_min_tight = _output_voltage_bounds(s_h_min, s_h_max_tight, w2, T, 10)

    s_o_min_orig, s_o_max_orig = _spike_bounds_from_voltage(V_o_max_orig, V_o_min_orig, threshold, T, min_time=2)
    s_o_min_tight, s_o_max_tight = _spike_bounds_from_voltage(V_o_max_tight, V_o_min_tight, threshold, T, min_time=2)

    orig_pred = forward(cfg, weights, img)
    print(f"target (o={orig_pred}) s_o_max: orig={s_o_max_orig[orig_pred]}, tight={s_o_max_tight[orig_pred]}")
    offenders_orig = sorted([(o, s_o_min_orig[o]) for o in range(10) if o != orig_pred], key=lambda x: x[1])[:3]
    offenders_tight = sorted([(o, s_o_min_tight[o]) for o in range(10) if o != orig_pred], key=lambda x: x[1])[:3]
    print(f"top-3 non-target s_o_min (orig):  {offenders_orig}")
    print(f"top-3 non-target s_o_min (tight): {offenders_tight}")

    # Check if tight bounds prove robust
    t_max = s_o_max_tight[orig_pred]
    prove = True
    for o in range(10):
        if o == orig_pred:
            continue
        if o < orig_pred:
            if s_o_min_tight[o] <= t_max:
                prove = False; break
        else:
            if s_o_min_tight[o] < t_max:
                prove = False; break
    print(f"tight bounds prove robust? {prove}")
