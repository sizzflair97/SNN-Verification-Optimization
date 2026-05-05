"""
Test LP relaxation of existing MILP formulation on sample 9144.
Sanity check: does LP solve fast enough to be useful at BnB root?
If LP proves INFEASIBLE → target dominates → robust proven.
"""
import time
import pulp
import numpy as np
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.dictionary_mnist import threshold
from adv_rob_mnist_module import run_milp_single


def run_lp_relaxed(cfg, weights_list, s0_orig, pred_orig, delta=1, time_limit=60.0, keep_spike_int=False):
    """Like run_milp_single but with all integer/binary vars relaxed to continuous.
    If keep_spike_int=True, keep spike_times variables as Integer (mixed LP).
    Returns (prob, wall_time)."""
    from pulp import LpVariable, LpAffineExpression, lpSum
    from utils.encoding_mnist import get_layer_neurons_iter

    n_layer_neurons = cfg.n_layer_neurons
    num_steps = cfg.num_steps
    tau = 1

    model = pulp.LpProblem("LP_relaxation", pulp.LpMinimize)
    EPS = 1e-4
    M_time = float(num_steps + 1)
    M_act = {}
    for layer in range(1, len(n_layer_neurons)):
        prev_layer = layer - 1
        w = weights_list[prev_layer]
        if prev_layer == 0:
            abs_sums = np.sum(np.abs(w.reshape(w.shape[0], -1)), axis=1)
        else:
            abs_sums = np.sum(np.abs(w[:, :, 0]), axis=1)
        M_act[layer] = float(abs_sums.max() + threshold + 10)

    # Continuous relaxations
    spike_cat = "Integer" if keep_spike_int else "Continuous"
    spike_times = {}
    neuron_perturbation = {}
    for neuron in get_layer_neurons_iter(cfg, 0):
        spike_times[neuron, 0] = LpVariable(f"s_0_{neuron}", 0, num_steps - 1, cat=spike_cat)
        neuron_perturbation[neuron] = LpVariable(f"d_{neuron}", 0, num_steps - 1, cat=spike_cat)
        model += spike_times[neuron, 0] - s0_orig[neuron] <= neuron_perturbation[neuron]
        model += s0_orig[neuron] - spike_times[neuron, 0] <= neuron_perturbation[neuron]
    model += lpSum(neuron_perturbation.values()) <= delta

    from utils.dictionary_mnist import Neuron_Layer_Time
    p = {}
    flag = {}
    activated = {}
    cond = {}
    for post_layer in range(1, len(n_layer_neurons)):
        for post_neuron in get_layer_neurons_iter(cfg, post_layer):
            spike_times[post_neuron, post_layer] = LpVariable(
                f"s_{post_neuron}_{post_layer}", tau * post_layer, num_steps - 1, cat=spike_cat
            )
            for t in range(num_steps):
                flag[post_neuron, post_layer, t] = LpVariable(
                    f"a_{post_neuron}_{post_layer}_{t}", lowBound=0.0, upBound=1.0, cat="Continuous"
                )
                activated[post_neuron, post_layer, t] = LpVariable(
                    f"(p>=theta)_{post_neuron}_{post_layer}_{t}", lowBound=0.0, upBound=1.0, cat="Continuous"
                )

    for prev_layer in range(len(n_layer_neurons) - 1):
        for prev_neuron in get_layer_neurons_iter(cfg, prev_layer):
            _spike_time = spike_times[prev_neuron, prev_layer]
            for t in range(num_steps):
                _cond = cond[prev_neuron, prev_layer, t] = LpVariable(
                    f"If_{prev_neuron}_{prev_layer}_{t}", lowBound=0.0, upBound=1.0, cat="Continuous"
                )
                model += t + (1 - _cond) * M_time >= _spike_time
                model += t + 1 - _cond * M_time <= _spike_time

    for post_layer in range(1, len(n_layer_neurons)):
        prev_layer = post_layer - 1
        for post_neuron in get_layer_neurons_iter(cfg, post_layer):
            p[post_neuron, post_layer, 0] = lpSum([])
            for t in range(1, num_steps):
                expr = LpAffineExpression()
                for prev_neuron in get_layer_neurons_iter(cfg, prev_layer):
                    weight = weights_list[prev_layer][post_neuron[0], prev_neuron[0], prev_neuron[1]]
                    expr += weight * cond[prev_neuron, prev_layer, t - tau]
                p[post_neuron, post_layer, t] = expr

            for t_prev in range(num_steps - 1):
                _p = p[post_neuron, post_layer, t_prev]
                _activated = activated[post_neuron, post_layer, t_prev]
                model += _p <= threshold + _activated * M_act[post_layer]
                model += _p >= threshold + EPS - (1 - _activated) * M_act[post_layer]
            model += activated[post_neuron, post_layer, num_steps - 1] == 1

            model += flag[post_neuron, post_layer, 0] == 0
            for t in range(1, num_steps):
                _flag = flag[post_neuron, post_layer, t]
                expr = LpAffineExpression()
                for t_prev in range(t):
                    _activated = activated[post_neuron, post_layer, t_prev]
                    model += _flag >= _activated
                    expr += _activated
                model += _flag <= expr

            one_hot = LpAffineExpression()
            xi_5_term = LpAffineExpression()
            for t in range(tau * post_layer, num_steps - 1):
                _spike_cond = LpVariable(
                    f"spike_{post_neuron}_{post_layer}_{t}", lowBound=0.0, upBound=1.0, cat="Continuous"
                )
                _flag = flag[post_neuron, post_layer, t]
                _activated = activated[post_neuron, post_layer, t]
                model += _spike_cond <= 1 - _flag
                model += _spike_cond <= _activated
                model += _spike_cond >= (1 - _flag) + _activated - 1
                one_hot += _spike_cond
                xi_5_term += t * _spike_cond
            xi_6_term = (num_steps - 1) * (1 - flag[post_neuron, post_layer, num_steps - 1])
            model += one_hot + (1 - flag[post_neuron, post_layer, num_steps - 1]) == 1
            model += spike_times[post_neuron, post_layer] == xi_5_term + xi_6_term

    target_spike_time = spike_times[(pred_orig, 0), len(n_layer_neurons) - 1]
    not_robust = []
    for out_neuron in get_layer_neurons_iter(cfg, len(n_layer_neurons) - 1):
        if out_neuron[0] == pred_orig:
            continue
        _other_spike_time = spike_times[out_neuron, len(n_layer_neurons) - 1]
        _not_robust = LpVariable(f"not_robust_{out_neuron}", lowBound=0.0, upBound=1.0, cat="Continuous")
        if out_neuron[0] < pred_orig:
            model += _other_spike_time <= target_spike_time + (1 - _not_robust) * M_time
            model += _other_spike_time >= target_spike_time + EPS - _not_robust * M_time
        else:
            model += _other_spike_time <= (target_spike_time - 1) + (1 - _not_robust) * M_time
            model += _other_spike_time >= (target_spike_time - 1) + EPS - _not_robust * M_time
        not_robust.append(_not_robust)
    model += lpSum(not_robust) >= 1
    model += target_spike_time  # dummy objective

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    tic = time.time()
    status = model.solve(solver)
    return model, time.time() - tic


if __name__ == "__main__":
    cfg = CFG(
        log_name="lp_full_test", subtype="mnist", load_data_func=load_mnist,
        n_layer_neurons=(784, 500, 10), layer_shapes=((28, 28), (500, 1), (10, 1)),
        num_steps=5, seed=42, num_samples=1, deltas=(2,),
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, *_ = load_mnist(cfg)
    img = images[9144]
    orig_pred = forward(cfg, weights, img)
    print(f"Sample 9144: orig_pred={orig_pred}")

    print("=== Pure LP relaxation (all vars continuous, big-M) ===")
    prob, elapsed = run_lp_relaxed(cfg, weights, img, orig_pred, delta=2, time_limit=60.0, keep_spike_int=False)
    print(f"status={pulp.LpStatus[prob.status]}, time={elapsed:.1f}s")

    print()
    print("=== Mixed: spike_times Integer, binary flags Continuous ===")
    prob2, elapsed2 = run_lp_relaxed(cfg, weights, img, orig_pred, delta=2, time_limit=120.0, keep_spike_int=True)
    print(f"status={pulp.LpStatus[prob2.status]}, time={elapsed2:.1f}s")
