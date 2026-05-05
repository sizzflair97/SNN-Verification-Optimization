"""
Interval Bound Propagation (IBP) for BnB pruning in SNN adversarial verification.

Given a partial BnB state (some pixels fixed, some remaining with budget),
compute sound over-approximations of reachable hidden/output voltages and
spike times, then check if robustness is already proved.

Key property: if `ibp_prove_robust(...)` returns True, no adversarial exists
in this subtree (sound prune). False means bounds are too loose; caller must
continue branching.

Semantics match forward pass in utils/mnist_net.py:
  - Voltage[:, t] is cumulative: sum_{p: s_p <= t} w_p
  - Spike triggered at time s iff V(s-1) > threshold (strict), clamped to num_steps-1
  - Output has implicit synaptic delay 2 (earliest fire at t=2)
"""
from __future__ import annotations
import numpy as np


# =============================================================================
# Budget-coupled IBP via 0/1 knapsack over pixel-flip decisions.
# =============================================================================

def _knapsack_max_benefit_vec(costs: np.ndarray, benefits: np.ndarray, budget: int) -> np.ndarray:
    """Exact 0/1 knapsack, vectorized across a trailing axis.

    costs: (N,) positive integers
    benefits: (N, K) — per-item benefits, broadcast across K parallel problems
    budget: int, upper bound on total cost

    Returns: (K,) array of max sum-of-benefits achievable at total cost <= budget.
    Items with cost > budget or benefit <= 0 are skipped (benefits here are assumed
    non-negative by caller — they filter).
    """
    if budget <= 0 or len(costs) == 0:
        return np.zeros(benefits.shape[1] if benefits.ndim > 1 else 1, dtype=np.float64)

    K = benefits.shape[1]
    dp = np.zeros((budget + 1, K), dtype=np.float64)
    for i in range(costs.shape[0]):
        c = int(costs[i])
        if c <= 0 or c > budget:
            continue
        b_i = benefits[i]
        # Update dp in reverse (0/1 knapsack) so each item is used at most once.
        for j in range(budget, c - 1, -1):
            np.maximum(dp[j], dp[j - c] + b_i, out=dp[j])
    return dp[budget]


def _compute_budgeted_s_h_bounds(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    num_steps: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (s_h_min, s_h_max) via budget-coupled knapsack IBP.

    Factored out of ibp_prove_robust_budgeted so β-branching can reuse it
    and override s_h bounds per case without recomputing V_h_max/V_h_min.
    """
    n_hidden = w1.shape[0]
    T = num_steps

    img_h, img_w = pixel_img.shape[:2]
    all_px, all_py = np.mgrid[0:img_h, 0:img_w].reshape(2, -1)
    times = pixel_img[all_px, all_py].astype(int)
    w_all = w1[:, all_px, all_py]
    t_grid = np.arange(T + 1)
    fire_mask = times[:, None] <= t_grid[None, :]
    V_h_base = w_all @ fire_mask.astype(np.float64)

    V_h_max = V_h_base.copy()
    V_h_min = V_h_base.copy()

    if len(remaining_pixels) > 0 and (rem_neg > 0 or rem_pos > 0):
        px_arr = remaining_pixels[:, 0]
        py_arr = remaining_pixels[:, 1]
        orig_arr = pixel_img[px_arr, py_arr].astype(int)
        w_rem = w1[:, px_arr, py_arr].astype(np.float64)

        for t in range(T + 1):
            pos_mask = orig_arr <= t
            neg_mask = ~pos_mask
            pos_costs = (t + 1 - orig_arr[pos_mask]).astype(int)
            pos_delta = -w_rem[:, pos_mask]
            neg_costs = (orig_arr[neg_mask] - t).astype(int)
            neg_delta = w_rem[:, neg_mask]

            if pos_costs.size > 0 and rem_pos > 0:
                ben = np.maximum(pos_delta, 0.0).T
                max_pos = _knapsack_max_benefit_vec(pos_costs, ben, rem_pos)
            else:
                max_pos = np.zeros(n_hidden)
            if neg_costs.size > 0 and rem_neg > 0:
                ben = np.maximum(neg_delta, 0.0).T
                max_neg = _knapsack_max_benefit_vec(neg_costs, ben, rem_neg)
            else:
                max_neg = np.zeros(n_hidden)
            V_h_max[:, t] += max_pos + max_neg

            if pos_costs.size > 0 and rem_pos > 0:
                ben = np.maximum(-pos_delta, 0.0).T
                min_pos = _knapsack_max_benefit_vec(pos_costs, ben, rem_pos)
            else:
                min_pos = np.zeros(n_hidden)
            if neg_costs.size > 0 and rem_neg > 0:
                ben = np.maximum(-neg_delta, 0.0).T
                min_neg = _knapsack_max_benefit_vec(neg_costs, ben, rem_neg)
            else:
                min_neg = np.zeros(n_hidden)
            V_h_min[:, t] -= min_pos + min_neg

    return _spike_bounds_from_voltage(V_h_max, V_h_min, threshold, T, min_time=1)


def _check_robust_from_s_h(
    s_h_min: np.ndarray,
    s_h_max: np.ndarray,
    w2_flat: np.ndarray,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
) -> bool:
    """Given (possibly-tightened) s_h bounds, check if output robustness follows."""
    T = num_steps
    V_o_max, V_o_min = _output_voltage_bounds(s_h_min, s_h_max, w2_flat, T, num_classes)
    s_o_min, s_o_max = _spike_bounds_from_voltage(V_o_max, V_o_min, threshold, T, min_time=2)
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


def ibp_prove_robust_budgeted(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
) -> bool:
    """Budget-coupled IBP: per-(h,t) max/min V_h solved as 0/1 knapsack.

    Tighter than the per-pixel-independent variant `ibp_prove_robust` because
    total L1 budget constraint is enforced (at most `rem_neg` total pixel shifts
    in the negative direction, similarly `rem_pos`).

    Sound: max V_h(t) per (h,t) is an upper bound over ALL feasible perturbations
    (each (h,t) solved independently — even looser bound would be sound; this
    one is tighter than per-pixel-indep).
    """
    s_h_min, s_h_max = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos, w1, num_steps, threshold
    )
    return _check_robust_from_s_h(
        s_h_min, s_h_max, w2_flat, num_steps, threshold, orig_pred, num_classes
    )


def ibp_prove_robust_multilayer(
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
    """BC-IBP for multi-hidden-layer TTFS SNNs.

    weights_list: ordered list of per-layer weights:
      weights_list[0]: shape (n_h_1, H, W)        -- input -> hidden_1
      weights_list[k]: shape (n_h_{k+1}, n_h_k, 1) for k >= 1  -- hidden_k -> hidden_{k+1}
      weights_list[-1]: shape (num_classes, n_h_L, 1)  -- last hidden -> output

    Architecture is sound by composition:
      (a) Layer 1 uses the budget-coupled knapsack (Theorem 3.1 in main).
      (b) Layers k >= 2 use interval-bound propagation through the step
          activation: at each (h, t), the indicator
          [s_{k-1, h'} <= t-tau] is uncertain only when
          s_{k-1,h'}^min <= t-tau < s_{k-1,h'}^max; worst-case sign-aware
          summation over uncertain pre-synaptic h' bounds V_k(h, t).
      (c) Spike-time intervals at layer k follow from voltage-monotonicity
          (cf. proof of Thm 3 Step 2).
      (d) Output check uses the same argmin tie-break as single-hidden.

    NOTE: bound looseness compounds with depth (each interval propagation
    drops cross-neuron joint feasibility). BC-IBP at L hidden layers is
    sound but progressively looser; use mainly as an existence proof of
    completeness preservation in the multi-hidden regime.
    """
    s_h_min, s_h_max = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list[0], num_steps, threshold,
    )
    # Propagate through hidden_1 -> ... -> hidden_L (exclude last weight which is to output).
    n_layers_total = len(weights_list)  # = L_hidden + 1 (output)
    for layer_idx in range(1, n_layers_total - 1):
        w_layer = weights_list[layer_idx]
        if w_layer.ndim == 3:
            w_layer = w_layer[:, :, 0]  # (n_next, n_prev)
        n_next = w_layer.shape[0]
        V_max, V_min = _output_voltage_bounds(
            s_h_min, s_h_max, w_layer, num_steps, n_next,
        )
        # Layer (layer_idx+1) lies at synaptic-delay (layer_idx+1) from input.
        s_h_min, s_h_max = _spike_bounds_from_voltage(
            V_max, V_min, threshold, num_steps, min_time=layer_idx + 1,
        )
        if np.any(s_h_min > s_h_max):
            return True  # branch infeasible
    # Final output check
    w_out = weights_list[-1]
    if w_out.ndim == 3:
        w_out = w_out[:, :, 0]
    return _check_robust_from_s_h(
        s_h_min, s_h_max, w_out, num_steps, threshold, orig_pred, num_classes,
    )


def _propagate_and_check_robust(
    s_h1_min, s_h1_max, weights_list, num_steps, threshold, orig_pred, num_classes,
):
    """Propagate s_h1 bounds through layers 2+ and check output robustness.

    Returns True iff the bounds prove robustness; False otherwise.
    """
    s_min, s_max = s_h1_min, s_h1_max
    n_layers_total = len(weights_list)
    for layer_idx in range(1, n_layers_total - 1):
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


def _beta_recurse_h1(
    s_h1_min, s_h1_max, weights_list, num_steps, threshold,
    orig_pred, num_classes, depth, top_k, _blocked=frozenset(),
):
    """β-branching at the FIRST hidden layer, repropagating downstream per branch.

    Picks the widest free h1, splits on midpoint, refines (s_h1_min, s_h1_max)
    accordingly, and re-propagates through layers 2+ for each branch. Both
    branches must individually prove robust for the parent to be robust.
    """
    if np.any(s_h1_min > s_h1_max):
        return True
    if _propagate_and_check_robust(
        s_h1_min, s_h1_max, weights_list, num_steps, threshold, orig_pred, num_classes,
    ):
        return True
    if depth <= 0:
        return False

    ranges = s_h1_max - s_h1_min
    candidates = [h for h in range(len(ranges)) if h not in _blocked and ranges[h] > 0]
    if not candidates:
        return False
    candidates.sort(key=lambda h: ranges[h], reverse=True)
    candidates = candidates[:top_k]

    for h in candidates:
        lo = int(s_h1_min[h])
        hi = int(s_h1_max[h])
        tau = (lo + hi) // 2
        if tau < lo or tau >= hi:
            continue

        # Case A: s_h1[h] ≤ τ
        s_max_A = s_h1_max.copy()
        s_max_A[h] = min(s_max_A[h], tau)
        ok_A = _beta_recurse_h1(
            s_h1_min, s_max_A, weights_list, num_steps, threshold,
            orig_pred, num_classes, depth - 1, top_k, _blocked | {h},
        )
        if not ok_A:
            continue

        # Case B: s_h1[h] ≥ τ+1
        s_min_B = s_h1_min.copy()
        s_min_B[h] = max(s_min_B[h], tau + 1)
        ok_B = _beta_recurse_h1(
            s_min_B, s_h1_max, weights_list, num_steps, threshold,
            orig_pred, num_classes, depth - 1, top_k, _blocked | {h},
        )
        if ok_B:
            return True

    return False


def ibp_prove_robust_multilayer_beta_h1(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    weights_list: list,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    beta_depth: int = 2,
    beta_top_k: int = 4,
) -> bool:
    """Multi-hidden BC-IBP with β-branching at the FIRST hidden layer.

    For each split, the constrained s_h1 bounds are re-propagated through all
    downstream layers. This is closer to a "cross-layer coupling" attempt
    because tightening h1's firing affects the entire chain.

    Sound by composition: BC-IBP at layer 1 is sound; the β split exhausts
    h1's firing-status binary; downstream propagation is sound interval
    propagation; output check is sound (Theorem~thm:bcibp). Completeness:
    inconclusive returns trigger BnB descent in the caller.
    """
    s_h1_min, s_h1_max = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list[0], num_steps, threshold,
    )
    return _beta_recurse_h1(
        s_h1_min, s_h1_max, weights_list, num_steps, threshold,
        orig_pred, num_classes, depth=beta_depth, top_k=beta_top_k,
    )


def ibp_prove_robust_multilayer_beta(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    weights_list: list,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    beta_depth: int = 2,
    beta_top_k: int = 4,
) -> bool:
    """Multi-hidden BC-IBP with β-branching on last-hidden firings.

    Pipeline:
      1. Layer 1 (input → hidden_1): budget-coupled knapsack for s_h1 bounds.
      2. Hidden layers 2..L: sound interval propagation (no β here -- the
         intermediate looseness is bounded by the s_{l-1} bound widths from
         the previous step, not by the hidden-output indicator coupling).
      3. Last-hidden → output: β-branching on the widest s_{h_last} ranges,
         exactly mirroring the single-hidden β code (`_beta_recurse`).

    Soundness: each β split exhausts the firing-status binary at the chosen
    h_last neuron; both branches must individually prove robust under the
    refined bounds. This is a standard β-CROWN-style argument applied at the
    last hidden layer; the input-budget knapsack at layer 1 is preserved
    unchanged.
    """
    s_h_min, s_h_max = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos,
        weights_list[0], num_steps, threshold,
    )
    n_layers_total = len(weights_list)
    # Propagate hidden_1 → hidden_2 → ... → hidden_L (last hidden).
    for layer_idx in range(1, n_layers_total - 1):
        w_layer = weights_list[layer_idx]
        if w_layer.ndim == 3:
            w_layer = w_layer[:, :, 0]
        n_next = w_layer.shape[0]
        V_max, V_min = _output_voltage_bounds(
            s_h_min, s_h_max, w_layer, num_steps, n_next,
        )
        s_h_min, s_h_max = _spike_bounds_from_voltage(
            V_max, V_min, threshold, num_steps, min_time=layer_idx + 1,
        )
        if np.any(s_h_min > s_h_max):
            return True
    # Now (s_h_min, s_h_max) are the bounds at the LAST hidden layer.
    # Apply β-branching on output via the existing single-hidden recursion.
    w_out = weights_list[-1]
    if w_out.ndim == 3:
        w_out = w_out[:, :, 0]
    return _beta_recurse(
        s_h_min, s_h_max, w_out, num_steps, threshold,
        orig_pred, num_classes, depth=beta_depth, top_k=beta_top_k,
    )


def ibp_prove_robust_beta(
    pixel_img: np.ndarray,
    remaining_pixels: np.ndarray,
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,
    w2_flat: np.ndarray,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    beta_depth: int = 2,
    beta_top_k: int = 1,
) -> bool:
    """β-branching on top of BC-IBP.

    If the plain BC-IBP bound fails to prove robustness, recursively case-split
    on `s_h ≤ τ` vs `s_h > τ` for the hidden neurons with the widest spike-time
    uncertainty. Both branches must be proved robust (possibly via deeper β
    splits) for the overall result. Sound because:
      - s_h_max ← min(s_h_max, τ) upper-bounds s_h under the assumption s_h ≤ τ.
      - s_h_min ← max(s_h_min, τ+1) lower-bounds s_h under s_h > τ.
      - Every feasible perturbation falls into exactly one of the two branches.
    If a branch is infeasible (s_h_min > s_h_max for some h after tightening),
    that branch is trivially robust (vacuously true).
    """
    # Base bounds first (shared across any β split).
    s_h_min0, s_h_max0 = _compute_budgeted_s_h_bounds(
        pixel_img, remaining_pixels, rem_neg, rem_pos, w1, num_steps, threshold
    )
    return _beta_recurse(
        s_h_min0, s_h_max0, w2_flat, num_steps, threshold,
        orig_pred, num_classes, depth=beta_depth, top_k=beta_top_k,
    )


def _beta_recurse(
    s_h_min: np.ndarray,
    s_h_max: np.ndarray,
    w2_flat: np.ndarray,
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
    depth: int,
    top_k: int,
    _blocked: frozenset = frozenset(),
) -> bool:
    """Recursive helper. Tries robust check at current bounds; if it fails and
    depth > 0, splits on widest-range hidden not already blocked.
    `_blocked` tracks hiddens already split on to avoid infinite recursion.
    """
    # 1. Infeasibility check (any branch-induced tightening made s_h_min > s_h_max).
    if np.any(s_h_min > s_h_max):
        return True

    # 2. Try to prove at current bounds.
    if _check_robust_from_s_h(
        s_h_min, s_h_max, w2_flat, num_steps, threshold, orig_pred, num_classes
    ):
        return True

    if depth <= 0:
        return False

    # 3. Pick candidate hiddens to split on — widest [s_h_min, s_h_max] range,
    #    skipping already-split and already-singleton neurons.
    ranges = s_h_max - s_h_min
    candidates = [h for h in range(len(ranges)) if h not in _blocked and ranges[h] > 0]
    if not candidates:
        return False
    candidates.sort(key=lambda h: ranges[h], reverse=True)
    candidates = candidates[:top_k]

    # 4. For each candidate, split on midpoint τ. BOTH branches must prove robust.
    for h in candidates:
        lo = int(s_h_min[h])
        hi = int(s_h_max[h])
        tau = (lo + hi) // 2  # split so that s_h ≤ tau | s_h ≥ tau+1 both non-empty
        if tau < lo or tau >= hi:
            continue

        # Case A: s_h ≤ tau (tighten s_h_max[h]).
        s_h_max_A = s_h_max.copy()
        s_h_max_A[h] = min(s_h_max_A[h], tau)
        ok_A = _beta_recurse(
            s_h_min, s_h_max_A, w2_flat, num_steps, threshold,
            orig_pred, num_classes, depth - 1, top_k, _blocked | {h},
        )
        if not ok_A:
            continue  # This split didn't help; try next candidate.

        # Case B: s_h ≥ tau+1 (tighten s_h_min[h]).
        s_h_min_B = s_h_min.copy()
        s_h_min_B[h] = max(s_h_min_B[h], tau + 1)
        ok_B = _beta_recurse(
            s_h_min_B, s_h_max, w2_flat, num_steps, threshold,
            orig_pred, num_classes, depth - 1, top_k, _blocked | {h},
        )
        if ok_B:
            return True

    return False


def ibp_prove_robust(
    pixel_img: np.ndarray,           # current pixel assignment, shape (H, W), int
    remaining_pixels: np.ndarray,     # shape (N_rem, 2), dtype int (px, py for each remaining)
    rem_neg: int,
    rem_pos: int,
    w1: np.ndarray,                  # (n_hidden, H, W)
    w2_flat: np.ndarray,             # (num_classes, n_hidden)
    num_steps: int,
    threshold: float,
    orig_pred: int,
    num_classes: int,
) -> bool:
    """Returns True iff IBP proves robustness in the current subtree.

    Per-pixel IBP (NOT coupled through L1 budget) — sound but loose.
    """
    n_hidden = w1.shape[0]
    T = num_steps

    # --- Step 1. Fixed-pixel contribution to V_h at each time ---
    # For all pixels whose value is fully determined (either filtered-out at root
    # or already branched in ancestors), accumulate their contribution.
    V_h_fixed = _fixed_pixel_voltage(pixel_img, remaining_pixels, w1, T, n_hidden)

    # --- Step 2. Remaining-pixel IBP for V_h bounds ---
    V_h_max, V_h_min = _remaining_pixel_bounds(
        V_h_fixed, pixel_img, remaining_pixels, rem_neg, rem_pos, w1, T, n_hidden,
    )

    # --- Step 3. Hidden spike time bounds ---
    s_h_min, s_h_max = _spike_bounds_from_voltage(V_h_max, V_h_min, threshold, T, min_time=1)

    # --- Step 4. Output voltage bounds given hidden spike ranges ---
    V_o_max, V_o_min = _output_voltage_bounds(s_h_min, s_h_max, w2_flat, T, num_classes)

    # --- Step 5. Output spike time bounds (clamped to >= 2 for synaptic delay) ---
    s_o_min, s_o_max = _spike_bounds_from_voltage(V_o_max, V_o_min, threshold, T, min_time=2)

    # --- Step 6. Robustness check (argmin tie-breaking: lower index wins) ---
    target_max = s_o_max[orig_pred]
    for o in range(num_classes):
        if o == orig_pred:
            continue
        if o < orig_pred:
            # tie with lower index o → non-robust, so must strictly have s_o > target
            if s_o_min[o] <= target_max:
                return False
        else:
            # tie with higher index o → target wins; need s_o >= target
            if s_o_min[o] < target_max:
                return False
    return True


def _fixed_pixel_voltage(pixel_img, remaining_pixels, w1, T, n_hidden):
    """V_h_fixed[:, t] = sum of w1[h, px, py] for every pixel (px,py) NOT in remaining
    whose current value s = pixel_img[px, py] satisfies s <= t.
    """
    # Build a mask of "remaining" positions
    rem_mask = np.zeros(pixel_img.shape[:2], dtype=bool)
    if len(remaining_pixels) > 0:
        rem_mask[remaining_pixels[:, 0], remaining_pixels[:, 1]] = True
    fixed_mask = ~rem_mask  # (28,28)

    # For efficiency, sort fixed pixels by their time value and accumulate
    V = np.zeros((n_hidden, T + 1), dtype=np.float64)
    fx, fy = np.nonzero(fixed_mask)
    if len(fx) == 0:
        return V
    ftimes = pixel_img[fx, fy].astype(int)
    fw = w1[:, fx, fy]  # (n_hidden, N_fixed)

    # For each time t, contribution = sum of fw[:, i] where ftimes[i] <= t
    # Vectorize: for each t, mask = (ftimes <= t), V[:, t] = fw @ mask
    t_grid = np.arange(T + 1)                         # (T+1,)
    fire_mask = ftimes[:, None] <= t_grid[None, :]    # (N_fixed, T+1)
    V = fw @ fire_mask.astype(np.float64)             # (n_hidden, T+1)
    return V


def _remaining_pixel_bounds(V_h_fixed, pixel_img, remaining_pixels, rem_neg, rem_pos, w1, T, n_hidden):
    """Add remaining-pixel contributions with per-pixel IBP.

    Each remaining pixel p has spike time s_p in [lo_p, hi_p] where
        lo_p = max(0, orig_p - rem_neg), hi_p = min(T-1, orig_p + rem_pos)
    Contribution to V_h(t): w1[h,p] * [s_p <= t].
    Per-pixel classification at time t:
        forced_true : hi_p <= t    → contribution fixed at w
        forced_false: lo_p > t     → contribution fixed at 0
        free        : lo_p <= t < hi_p → contribution in {0, w}
    V_h_max takes max over free (w or 0), V_h_min takes min.
    """
    V_max = V_h_fixed.copy()
    V_min = V_h_fixed.copy()
    if len(remaining_pixels) == 0:
        return V_max, V_min

    px_arr = remaining_pixels[:, 0]
    py_arr = remaining_pixels[:, 1]
    orig = pixel_img[px_arr, py_arr].astype(int)
    lo = np.maximum(0, orig - rem_neg)
    hi = np.minimum(T - 1, orig + rem_pos)
    w_rem = w1[:, px_arr, py_arr].T.astype(np.float64)  # (N_rem, n_hidden)
    w_pos = np.maximum(w_rem, 0.0)
    w_neg = np.minimum(w_rem, 0.0)

    t_grid = np.arange(T + 1)
    forced_true = hi[:, None] <= t_grid[None, :]      # (N_rem, T+1)
    forced_false = lo[:, None] > t_grid[None, :]
    free = ~forced_true & ~forced_false

    # Add forced_true: both V_max and V_min gain w_rem at those t's
    V_max += (forced_true.T.astype(np.float64) @ w_rem).T
    V_min += (forced_true.T.astype(np.float64) @ w_rem).T
    # Add free: V_max gains positive-only, V_min gains negative-only
    V_max += (free.T.astype(np.float64) @ w_pos).T
    V_min += (free.T.astype(np.float64) @ w_neg).T
    return V_max, V_min


def _spike_bounds_from_voltage(V_max, V_min, threshold, T, min_time):
    """Given V_max, V_min: (n, T+1), return (s_min, s_max): (n,).

    s_min = earliest possible spike time: first t' in [0, T-2] where V_max(t') > threshold;
            spike at t = t' + 1; if none, spike forced at T-1.
    s_max = latest possible spike time : first t' in [0, T-2] where V_min(t') > threshold;
            spike at t = t' + 1; if none, spike forced at T-1.

    Result clamped down to at least `min_time` (synaptic delay from preceding layers).
    """
    if T < 2:
        return np.full(V_max.shape[0], T - 1), np.full(V_max.shape[0], T - 1)

    slice_max = V_max[:, : T - 1] > threshold         # (n, T-1)
    slice_min = V_min[:, : T - 1] > threshold

    any_max = slice_max.any(axis=1)
    any_min = slice_min.any(axis=1)

    # argmax returns 0 when all False; masked via np.where
    s_min = np.where(any_max, slice_max.argmax(axis=1) + 1, T - 1)
    s_max = np.where(any_min, slice_min.argmax(axis=1) + 1, T - 1)

    s_min = np.maximum(s_min, min_time)
    s_max = np.maximum(s_max, min_time)
    return s_min, s_max


def _output_voltage_bounds(s_h_min, s_h_max, w2_flat, T, num_classes):
    """V_o(t) = sum_h w2[o, h] * [s_h <= t], with s_h in [s_h_min[h], s_h_max[h]]."""
    n_hidden = s_h_min.shape[0]
    t_grid = np.arange(T + 1)
    forced_true_h = s_h_max[:, None] <= t_grid[None, :]   # (n_hidden, T+1)
    forced_false_h = s_h_min[:, None] > t_grid[None, :]
    free_h = ~forced_true_h & ~forced_false_h

    w2_T = w2_flat.T.astype(np.float64)                   # (n_hidden, num_classes)
    w2_pos = np.maximum(w2_T, 0.0)
    w2_neg = np.minimum(w2_T, 0.0)

    # V_o_max[t, o] = sum over h of (forced_true? w : 0) + (free? max(w,0) : 0)
    V_o_max = (forced_true_h.T.astype(np.float64) @ w2_T) + (free_h.T.astype(np.float64) @ w2_pos)
    V_o_min = (forced_true_h.T.astype(np.float64) @ w2_T) + (free_h.T.astype(np.float64) @ w2_neg)
    return V_o_max.T, V_o_min.T
