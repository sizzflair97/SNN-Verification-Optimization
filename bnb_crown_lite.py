"""
α,β-CROWN-lite for SNN BnB pruning (Level 1).

Integrates:
  - SOS1 encoding on pixel spike times (no big-M on pixel layer)
  - CROWN-style linear relaxation of hidden & output threshold activations
  - Joint budget constraints
  - Single LP solve at root that checks cross-layer robustness

CROWN activation linear bounds for `a = I[V > threshold]` on `V ∈ [V_min, V_max]`:
  If V_min > threshold: a = 1 (constant)
  If V_max <= threshold: a = 0 (constant)
  Otherwise (V_min <= threshold < V_max):
    Upper: a ≤ (V - V_min) / (threshold - V_min)
    Lower: a ≥ (V - threshold) / (V_max - threshold)
  Both are valid linear over/under-envelopes of the step function.

Spike-time-indicator `cond = I[s_n ≤ τ] = I[∃ τ' < τ with activation]`:
  cond_τ ≥ a_{τ'}   for every τ' < τ   (lower: if any activated, cond = 1)
  cond_τ ≤ Σ_{τ' < τ} a_{τ'}           (upper: cond can't exceed total activation mass)
  cond_τ ≥ cond_{τ-1}                  (monotonicity)
  cond_τ ∈ [0, 1]

Adversarial gap: for each non-target o and each τ, robustness needs
  cond_out[target, τ] ≥ cond_out[o, τ]  (target fires no later than non-target)
LP objective: maximize  max over (o, τ) of (cond_out[o, τ] - cond_out[target, τ]).
If the LP optimum ≤ 0 (with tolerance), robustness is proven.

This is a SOUND upper bound because LP relaxes binary variables; any actual
perturbation satisfies stricter constraints than the LP's fractional solutions.
"""
from __future__ import annotations
import numpy as np
import pulp

from bnb_ibp import _knapsack_max_benefit_vec, _spike_bounds_from_voltage, _output_voltage_bounds


def _compute_v_bounds(img, rem_neg, rem_pos, w1, w2_flat, num_steps, threshold, num_classes):
    """Reuse IBP budget-coupled logic to get V_h and V_o bounds."""
    T = num_steps
    n_hidden = w1.shape[0]

    all_px, all_py = np.mgrid[0:28, 0:28].reshape(2, -1)
    times = img[all_px, all_py].astype(int)
    w_all = w1[:, all_px, all_py]
    t_grid = np.arange(T + 1)
    fire_mask = times[:, None] <= t_grid[None, :]
    V_h_base = w_all @ fire_mask.astype(np.float64)

    orig_arr = img[all_px, all_py].astype(int)
    w_rem = w1[:, all_px, all_py].astype(np.float64)
    V_h_max = V_h_base.copy()
    V_h_min = V_h_base.copy()
    for t in range(T + 1):
        pos_mask = orig_arr <= t
        neg_mask = ~pos_mask
        pos_costs = (t + 1 - orig_arr[pos_mask]).astype(int)
        pos_delta = -w_rem[:, pos_mask]
        neg_costs = (orig_arr[neg_mask] - t).astype(int)
        neg_delta = w_rem[:, neg_mask]
        if pos_costs.size > 0 and rem_pos > 0:
            max_pos = _knapsack_max_benefit_vec(pos_costs, np.maximum(pos_delta, 0.0).T, rem_pos)
            min_pos = _knapsack_max_benefit_vec(pos_costs, np.maximum(-pos_delta, 0.0).T, rem_pos)
        else:
            max_pos = np.zeros(n_hidden); min_pos = np.zeros(n_hidden)
        if neg_costs.size > 0 and rem_neg > 0:
            max_neg = _knapsack_max_benefit_vec(neg_costs, np.maximum(neg_delta, 0.0).T, rem_neg)
            min_neg = _knapsack_max_benefit_vec(neg_costs, np.maximum(-neg_delta, 0.0).T, rem_neg)
        else:
            max_neg = np.zeros(n_hidden); min_neg = np.zeros(n_hidden)
        V_h_max[:, t] += max_pos + max_neg
        V_h_min[:, t] -= min_pos + min_neg

    s_h_min, s_h_max = _spike_bounds_from_voltage(V_h_max, V_h_min, threshold, T, min_time=1)
    V_o_max, V_o_min = _output_voltage_bounds(s_h_min, s_h_max, w2_flat, T, num_classes)
    return V_h_min, V_h_max, V_o_min, V_o_max


def crown_lite_prove_robust(
    img: np.ndarray,
    orig_pred: int,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    num_steps: int,
    threshold: float,
    num_classes: int,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> bool:
    """Returns True iff LP-relaxed CROWN-lite proves robustness at root."""
    T = num_steps
    n_hidden = w1.shape[0]

    # Precompute V bounds (for CROWN activation relaxation)
    V_h_min, V_h_max, V_o_min, V_o_max = _compute_v_bounds(
        img, rem_neg, rem_pos, w1, w2_flat, num_steps, threshold, num_classes,
    )

    prob = pulp.LpProblem("crown_lite", pulp.LpMinimize)

    # SOS1 variables for each pixel
    all_pixels = [(i, j) for i in range(28) for j in range(28)]
    N = len(all_pixels)
    orig_arr = np.array([int(img[px, py]) for px, py in all_pixels], dtype=int)

    z = []  # z[p] = dict {s_value -> LpVar}
    pos_expr = pulp.LpAffineExpression()
    neg_expr = pulp.LpAffineExpression()
    for p, (px, py) in enumerate(all_pixels):
        op = int(orig_arr[p])
        lo = max(0, op - rem_neg)
        hi = min(T - 1, op + rem_pos)
        zp = {}
        for s in range(lo, hi + 1):
            var = pulp.LpVariable(f"z_{p}_{s}", lowBound=0.0, upBound=1.0, cat="Continuous")
            zp[s] = var
            if s > op:
                pos_expr += (s - op) * var
            elif s < op:
                neg_expr += (op - s) * var
        z.append(zp)
        prob += pulp.lpSum(zp.values()) == 1
    prob += pos_expr <= float(rem_pos)
    prob += neg_expr <= float(rem_neg)

    # y_{p, τ} = Σ_{s ≤ τ} z_{p, s}  (not a variable, just an expression cache)
    y = [[None] * (T + 1) for _ in range(N)]
    for p in range(N):
        for tau in range(T + 1):
            y[p][tau] = pulp.lpSum(var for s, var in z[p].items() if s <= tau)

    # V_h(τ) linear expression in z
    # Precompute w1 entries for efficiency
    w1_flat = np.array([[w1[h, px, py] for (px, py) in all_pixels] for h in range(n_hidden)])  # (n_h, N)

    V_h_expr = [[None] * (T + 1) for _ in range(n_hidden)]
    for h in range(n_hidden):
        for tau in range(T + 1):
            expr = pulp.LpAffineExpression()
            for p in range(N):
                w = w1_flat[h, p]
                if w != 0.0:
                    expr += float(w) * y[p][tau]
            V_h_expr[h][tau] = expr

    # Hidden activation a_{h, τ} ≈ I[V_h(τ) > threshold]
    # (CROWN relaxation). τ = T-1 is FORCED to 1 (Voltage[:, T-1] = threshold+1 in forward pass).
    a_hid = [[None] * (T + 1) for _ in range(n_hidden)]
    for h in range(n_hidden):
        for tau in range(T + 1):
            if tau == num_steps - 1:
                a_hid[h][tau] = 1.0  # forced spike at last natural timestep
                continue
            Vmin = V_h_min[h, tau]
            Vmax = V_h_max[h, tau]
            if Vmin > threshold:
                a_hid[h][tau] = 1.0
            elif Vmax <= threshold:
                a_hid[h][tau] = 0.0
            else:
                var = pulp.LpVariable(f"a_h{h}_t{tau}", lowBound=0.0, upBound=1.0, cat="Continuous")
                V = V_h_expr[h][tau]
                prob += (threshold - Vmin) * var <= V - Vmin
                prob += (Vmax - threshold) * var >= V - threshold
                a_hid[h][tau] = var

    # cond_{h, τ} = I[s_h ≤ τ] = I[∃ τ' ∈ [1, τ] with V_h(τ'-1) > threshold]
    # Link to a_hid: cond_{h, τ} = max over τ' ∈ [1, τ] of a_hid[h, τ'-1]
    # cond_{h, τ} = I[s_h ≤ τ]. Spike time in forward is clamped to [1, T-1], so
    #   cond[τ] = 1 iff argmax(V > threshold) + 1 ≤ τ
    #         = I[∃ τ' < τ with natural crossing] for τ < T-1
    #         = 1 (constant) for τ ≥ T-1 (clamp ensures s_h ≤ T-1)
    cond_hid = [[None] * (T + 1) for _ in range(n_hidden)]
    for h in range(n_hidden):
        cond_hid[h][0] = 0.0
        for tau in range(1, T + 1):
            if tau >= num_steps - 1:
                cond_hid[h][tau] = 1.0
                continue
            var = pulp.LpVariable(f"cond_h{h}_t{tau}", lowBound=0.0, upBound=1.0, cat="Continuous")
            cond_hid[h][tau] = var
            for taup in range(tau):
                val = a_hid[h][taup]
                if val is not None:
                    prob += var >= val
            total = pulp.LpAffineExpression()
            for taup in range(tau):
                val = a_hid[h][taup]
                if val is not None:
                    total += val
            prob += var <= total
            prev = cond_hid[h][tau - 1]
            if prev is not None and not isinstance(prev, (int, float)):
                prob += var >= prev
            elif isinstance(prev, (int, float)) and prev > 0:
                prob += var >= prev

    # V_o(τ) = Σ_h w2[o, h] · cond_{h, τ}
    V_o_expr = [[None] * (T + 1) for _ in range(num_classes)]
    for o in range(num_classes):
        for tau in range(T + 1):
            expr = pulp.LpAffineExpression()
            for h in range(n_hidden):
                w = w2_flat[o, h]
                if w != 0.0:
                    val = cond_hid[h][tau]
                    if val is not None:
                        expr += float(w) * val
            V_o_expr[o][tau] = expr

    # Output activation a_o (τ = T-1 forced to 1, output min-time = 2)
    a_out = [[None] * (T + 1) for _ in range(num_classes)]
    for o in range(num_classes):
        for tau in range(T + 1):
            if tau == num_steps - 1:
                a_out[o][tau] = 1.0
                continue
            Vmin = V_o_min[o, tau]
            Vmax = V_o_max[o, tau]
            if Vmin > threshold:
                a_out[o][tau] = 1.0
            elif Vmax <= threshold:
                a_out[o][tau] = 0.0
            else:
                var = pulp.LpVariable(f"a_o{o}_t{tau}", lowBound=0.0, upBound=1.0, cat="Continuous")
                V = V_o_expr[o][tau]
                prob += (threshold - Vmin) * var <= V - Vmin
                prob += (Vmax - threshold) * var >= V - threshold
                a_out[o][tau] = var

    # cond_out (clamped at τ ≥ T-1, output min-time 2 means cond_out[o, 0] = cond_out[o, 1] = 0)
    cond_out = [[None] * (T + 1) for _ in range(num_classes)]
    for o in range(num_classes):
        cond_out[o][0] = 0.0
        cond_out[o][1] = 0.0  # synaptic delay 2
        for tau in range(2, T + 1):
            if tau >= num_steps - 1:
                cond_out[o][tau] = 1.0
                continue
            var = pulp.LpVariable(f"cond_o{o}_t{tau}", lowBound=0.0, upBound=1.0, cat="Continuous")
            cond_out[o][tau] = var
            for taup in range(tau):
                val = a_out[o][taup]
                if val is not None:
                    prob += var >= val
            total = pulp.LpAffineExpression()
            for taup in range(tau):
                val = a_out[o][taup]
                if val is not None:
                    total += val
            prob += var <= total
            prev = cond_out[o][tau - 1]
            if prev is not None and not isinstance(prev, (int, float)):
                prob += var >= prev
            elif isinstance(prev, (int, float)) and prev > 0:
                prob += var >= prev

    # For each (o, τ), solve a SEPARATE LP maximizing gap = cond_out[o, τ] - cond_out[target, τ].
    # Robust iff max gap ≤ 0 for every (o, τ).
    import time
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    max_gap_overall = -1e9
    ambiguous_pairs = 0
    tic = time.time()
    for o in range(num_classes):
        if o == orig_pred:
            continue
        for tau in range(2, T + 1):  # τ = 0, 1 both constants → skip. cond_out[:, T-1] = 1 → skip.
            o_val = cond_out[o][tau]
            t_val = cond_out[orig_pred][tau]
            # Skip pairs that are both constants
            if isinstance(o_val, (int, float)) and isinstance(t_val, (int, float)):
                g = float(o_val) - float(t_val)
                if g > max_gap_overall:
                    max_gap_overall = g
                continue
            ambiguous_pairs += 1
            # Copy problem and add objective
            sub = prob.copy()
            ov = o_val if isinstance(o_val, (int, float)) else o_val
            tv = t_val if isinstance(t_val, (int, float)) else t_val
            sub += -(ov - tv)  # minimize -(o - t) == maximize (o - t)
            status = sub.solve(solver)
            if status != pulp.LpStatusOptimal:
                if verbose:
                    print(f"  (o={o}, τ={tau}) LP status={pulp.LpStatus[status]} — conservative: not-robust")
                return False
            val = -pulp.value(sub.objective)  # gap (since we minimized -gap)
            if val > max_gap_overall:
                max_gap_overall = val
                if verbose:
                    print(f"  (o={o}, τ={tau}) gap={val:.4f}")
            if val > 1e-6:
                # Found a non-target/τ that could outpace target → not proven robust
                elapsed = time.time() - tic
                if verbose:
                    print(f"LP total: {elapsed:.1f}s, max_gap={max_gap_overall:.4f}, AMBIGUOUS ({ambiguous_pairs} pairs checked)")
                return False

    elapsed = time.time() - tic
    if verbose:
        print(f"LP total: {elapsed:.1f}s, max_gap={max_gap_overall:.4f}, {ambiguous_pairs} pairs checked — ROBUST")
    return max_gap_overall <= 1e-6


# -----------------------------------------------------------------------------
# Standalone test on sample 9144
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights
    from utils.dictionary_mnist import threshold

    cfg = CFG(
        log_name="crown_test", subtype="mnist", load_data_func=load_mnist,
        n_layer_neurons=(784, 500, 10), layer_shapes=((28, 28), (500, 1), (10, 1)),
        num_steps=5, seed=42, num_samples=1, deltas=(2,),
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, *_ = load_mnist(cfg)

    for sample_no in [14628, 48598, 9144]:
        img = images[sample_no]
        orig_pred = forward(cfg, weights, img)
        w1 = weights[0]
        w2 = weights[1][:, :, 0]
        print(f"\n=== sample {sample_no}, orig_pred={orig_pred}, n_h=500, delta=2 ===")
        for rn in range(3):
            rp = 2 - rn
            tic = time.time()
            ok = crown_lite_prove_robust(
                img.astype(int), orig_pred, rn, rp, w1, w2,
                num_steps=5, threshold=threshold, num_classes=10,
                time_limit=60.0, verbose=True,
            )
            elapsed = time.time() - tic
            print(f"  rn={rn}, rp={rp}: prove_robust={ok}, total {elapsed:.1f}s")
