"""
Drill-down on hard case: n_h=500, delta=2, sample=9144 (robust).

Goal: why does budget-coupled IBP fail to prove at the root?
Steps:
  1. Compute baseline V_h(t), s_h, V_o(t), s_o.
  2. For each (rn, rp) split in {(0,2),(1,1),(2,0)}:
     - Run ibp_prove_robust_budgeted with verbose bounds.
     - Report target_s_o_max and each non-target's s_o_min.
     - Identify which output neurons tie/overtake the target under the bound.
  3. If root fails, simulate one level of BnB: pick top-sensitivity pixel,
     branch it, and re-check IBP at each child. Report which branch prunes.
"""
from __future__ import annotations
import numpy as np
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.dictionary_mnist import threshold
from bnb_ibp import (
    ibp_prove_robust_budgeted,
    _spike_bounds_from_voltage,
    _output_voltage_bounds,
    _knapsack_max_benefit_vec,
)


def compute_bounds_verbose(
    pixel_img, remaining_pixels, rem_neg, rem_pos,
    w1, w2_flat, num_steps, threshold, orig_pred, num_classes,
):
    """Replicate ibp_prove_robust_budgeted but return intermediate bounds."""
    n_hidden = w1.shape[0]
    T = num_steps

    all_px, all_py = np.mgrid[0:28, 0:28].reshape(2, -1)
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
                max_pos = _knapsack_max_benefit_vec(pos_costs, np.maximum(pos_delta, 0.0).T, rem_pos)
            else:
                max_pos = np.zeros(n_hidden)
            if neg_costs.size > 0 and rem_neg > 0:
                max_neg = _knapsack_max_benefit_vec(neg_costs, np.maximum(neg_delta, 0.0).T, rem_neg)
            else:
                max_neg = np.zeros(n_hidden)
            V_h_max[:, t] += max_pos + max_neg

            if pos_costs.size > 0 and rem_pos > 0:
                min_pos = _knapsack_max_benefit_vec(pos_costs, np.maximum(-pos_delta, 0.0).T, rem_pos)
            else:
                min_pos = np.zeros(n_hidden)
            if neg_costs.size > 0 and rem_neg > 0:
                min_neg = _knapsack_max_benefit_vec(neg_costs, np.maximum(-neg_delta, 0.0).T, rem_neg)
            else:
                min_neg = np.zeros(n_hidden)
            V_h_min[:, t] -= min_pos + min_neg

    s_h_min, s_h_max = _spike_bounds_from_voltage(V_h_max, V_h_min, threshold, T, min_time=1)
    V_o_max, V_o_min = _output_voltage_bounds(s_h_min, s_h_max, w2_flat, T, num_classes)
    s_o_min, s_o_max = _spike_bounds_from_voltage(V_o_max, V_o_min, threshold, T, min_time=2)
    return V_h_max, V_h_min, s_h_min, s_h_max, V_o_max, V_o_min, s_o_min, s_o_max


def main():
    cfg = CFG(
        log_name="dbg9144",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=42,
        num_samples=1,
        deltas=(2,),
        n_layer_neurons=(784, 500, 10),
        layer_shapes=((28, 28), (500, 1), (10, 1)),
        num_steps=5,
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist(cfg)
    sample_no = 9144
    img = images[sample_no]
    orig_pred = forward(cfg, weights, img)
    print(f"Sample {sample_no}: orig_pred = {orig_pred}, label = {labels[sample_no]}")

    w1 = weights[0]
    w2_flat = weights[1][:, :, 0]

    # Baseline
    base_ft = []
    forward(cfg, weights, img, base_ft)
    s_o_base = base_ft[-1]
    print(f"Baseline s_o = {s_o_base.tolist()}")
    print(f"Target (o={orig_pred}) baseline spike = {s_o_base[orig_pred]}")
    non_targets = sorted(
        [(o, s_o_base[o]) for o in range(10) if o != orig_pred],
        key=lambda x: x[1],
    )
    print("Non-target baseline spikes (sorted):")
    for o, s in non_targets[:5]:
        print(f"  class {o}: s_o_base = {s}")
    print()

    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)], dtype=int)

    # Check each budget split
    T = 5
    delta = 2
    for rn in range(delta + 1):
        rp = delta - rn
        ok = ibp_prove_robust_budgeted(
            pixel_img=img.astype(int),
            remaining_pixels=all_pixels,
            rem_neg=rn, rem_pos=rp,
            w1=w1, w2_flat=w2_flat,
            num_steps=5, threshold=threshold,
            orig_pred=orig_pred, num_classes=10,
        )
        print(f"[rn={rn}, rp={rp}] ibp_prove_robust_budgeted → {ok}")
        if ok:
            continue

        # Drill down: compute bounds
        _, _, s_h_min, s_h_max, V_o_max, V_o_min, s_o_min, s_o_max = compute_bounds_verbose(
            img.astype(int), all_pixels, rn, rp, w1, w2_flat, 5, threshold, orig_pred, 10,
        )
        target_max = s_o_max[orig_pred]
        print(f"  target s_o_max = {target_max}")
        print(f"  target V_o range at t∈[0..{T}]:")
        for t in range(T + 1):
            print(f"    t={t}: V_o_min={V_o_min[orig_pred, t]:.1f}, V_o_max={V_o_max[orig_pred, t]:.1f} (thresh={threshold})")
        print(f"  non-target s_o_min (sorted, top offenders first):")
        offenders = sorted(
            [(o, s_o_min[o]) for o in range(10) if o != orig_pred],
            key=lambda x: x[1],
        )
        for o, s_min in offenders[:5]:
            if o < orig_pred:
                req = target_max + 1  # need s_o_min > target_max
                ok_o = s_min >= req
            else:
                req = target_max
                ok_o = s_min >= req
            marker = " ✓" if ok_o else " ✗ (fails)"
            print(f"    class {o}: s_o_min={s_min}  (need >={req}){marker}")
        # Hidden neurons with widest interval (candidate IBP tightening targets)
        widest = np.argsort(s_h_max - s_h_min)[::-1][:5]
        print(f"  top-5 widest hidden spike intervals:")
        for h in widest:
            print(f"    h={h}: [{s_h_min[h]}, {s_h_max[h]}]")
        print()


if __name__ == "__main__":
    main()
