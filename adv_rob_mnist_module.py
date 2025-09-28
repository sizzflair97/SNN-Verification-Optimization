from copy import deepcopy
from multiprocessing import Pool
from random import sample as random_sample
from random import seed
from typing import Any
from collections.abc import Generator
import time, logging, pdb
import numpy as np
import pulp
from pulp import LpVariable, LpAffineExpression, lpSum
from torch import mode
from z3 import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import CFG
from utils.debug import info
from utils.mnist_net import forward, backward, prepare_weights

def run_z3(cfg: CFG, *, weights_list: TWeightList, images: TImageBatch):
    n_layer_neurons = cfg.n_layer_neurons
    S = Solver()
    spike_times = gen_spike_times(cfg)
    weights = gen_weights(cfg, weights_list)

    # Load equations.
    eqn_path = (
        f"eqn/eqn_{num_steps}_{'_'.join([str(i) for i in n_layer_neurons])}.txt"
    )
    if not load_expr or not os.path.isfile(eqn_path):
        node_eqns = gen_node_eqns(cfg, weights, spike_times)
        S.add(node_eqns)
        if save_expr:
            try:
                with open(eqn_path, "w") as f:
                    f.write(S.sexpr())
                    info("Node equations are saved.")
            except:
                pdb.set_trace(header="Failed to save node eqns.")
    else:
        S.from_file(eqn_path)
    info("Solver is loaded.")

    samples_no_list, sampled_imgs, orig_preds = sample_images_and_predictions(cfg, weights_list, images)

    # For each delta
    for delta in cfg.deltas:
        global check_sample

        def check_sample(sample: tuple[int, TImage, int]):
            sample_no, img, orig_pred = sample
            orig_neuron = (orig_pred, 0)
            tx = time.time()

            # Input property terms
            prop: list[BoolRef] = []
            input_layer = 0
            delta_pos = IntVal(0)
            delta_neg = IntVal(0)

            def relu(x: Any):
                return If(x > 0, x, 0)

            for in_neuron in get_layer_neurons_iter(cfg, input_layer):
                # Try to avoid using abs, as it makes z3 extremely slow.
                delta_pos += relu(
                    spike_times[in_neuron, input_layer] - int(img[in_neuron])
                )
                delta_neg += relu(
                    int(img[in_neuron]) - spike_times[in_neuron, input_layer]
                )
            prop.append((delta_pos + delta_neg) <= delta)
            info(f"Inputs Property Done in {time.time() - tx} sec")

            # Output property
            tx = time.time()
            op = []
            last_layer = len(n_layer_neurons) - 1
            for out_neuron in get_layer_neurons_iter(cfg, last_layer):
                if out_neuron != orig_neuron:
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    op.append(
                        spike_times[out_neuron, last_layer]
                        <= spike_times[orig_neuron, last_layer]
                    )
            op = Or(op)
            info(f"Output Property Done in {time.time() - tx} sec")

            tx = time.time()
            S_instance = deepcopy(S)
            info(f"Network Encoding read in {time.time() - tx} sec")
            S_instance.add(op)  # type: ignore
            S_instance.add(prop)  # type: ignore
            info(f"Total model ready in {time.time() - tx}")

            info("Query processing starts")
            # set_param(verbose=2)
            # set_param("parallel.enable", True)
            tx = time.time()
            result = S_instance.check()  # type: ignore
            info(f"Checking done in time {time.time() - tx}")
            if result == sat:
                info(f"Not robust for sample {sample_no} and delta={delta}")
            elif result == unsat:
                info(f"Robust for sample {sample_no} and delta={delta}")
            else:
                info(
                    f"Unknown at sample {sample_no} for reason {S_instance.reason_unknown()}"
                )
            info("")
            return result

        samples = zip(samples_no_list, sampled_imgs, orig_preds)
        if mp:
            with Pool(num_procs) as pool:
                pool.map(check_sample, samples)
                pool.close()
                pool.join()
        else:
            for sample in samples:
                check_sample(sample)

    info("")

def sample_images_and_predictions(cfg:CFG, weights_list:TWeightList, images: TImageBatch):
    samples_no_list = list[int]()
    sampled_imgs = list[TImage]()
    orig_preds = list[int]()
    for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
        info(f"sample {sample_no} is drawn.")
        samples_no_list.append(sample_no)
        img = images[sample_no]
        sampled_imgs.append(img)  # type: ignore
        orig_preds.append(forward(cfg, weights_list, img))
    info(f"Sampling is completed with {num_procs} samples.")
    return samples_no_list, sampled_imgs, orig_preds

def run_milp(cfg: CFG, *, weights_list:TWeightList, images: TImageBatch, MAP = {pulp.LpStatusOptimal: "Not Robust", pulp.LpStatusInfeasible: "Robust"}):
    samples_no_list, sampled_imgs, orig_preds = sample_images_and_predictions(cfg, weights_list, images)
    for delta in cfg.deltas:
        info(f"Delta: {delta}")
        for sample_no, img, orig_pred in zip(samples_no_list, sampled_imgs, orig_preds):
            _model, _tx = run_milp_single(cfg, weights_list, img, orig_pred, delta=delta)
            info(f"Sample {sample_no}\t|\ttime: {_tx:.13f}\t|\tstatus: {MAP[_model.status]}")

def run_milp_single(cfg:CFG, weights_list:TWeightList, s0_orig:TImage, pred_orig:int, delta:int = 1) -> tuple[pulp.LpProblem, float]:
    n_layer_neurons = cfg.n_layer_neurons
    num_steps = cfg.num_steps
    tau = 1  # synaptic delay

    model = pulp.LpProblem("MultiLayer_SNN_Verification", pulp.LpMinimize)
    M = 1000000  # Large constant for MILP
    EPS = 1e-7

    # Variables: spike time and perturbation (input layer)
    spike_times = dict[tuple[NodeIdx, LayerIdx], LpVariable]()  # s[l,n] = spike time
    neuron_perturbation = dict[NodeIdx, LpVariable]()  # d_n = perturbation for neuron n
    for neuron in get_layer_neurons_iter(cfg, 0):
        spike_times[neuron, 0] = LpVariable(f"s_0_{neuron}", 0, num_steps - 1, cat=pulp.LpInteger) # Xi_1
        # Begin Xi_7
        neuron_perturbation[neuron] = LpVariable(f"d_{neuron}", 0, num_steps - 1, cat=pulp.LpInteger)
        model += spike_times[neuron, 0] - s0_orig[neuron] <= neuron_perturbation[neuron]
        model += s0_orig[neuron] - spike_times[neuron, 0] <= neuron_perturbation[neuron]
        # End Xi_7
    model += lpSum(neuron_perturbation.values()) <= delta

    # Intermediate variables
    p = dict[Neuron_Layer_Time, LpAffineExpression]()  # p[l,t,n] = potential at layer l, time t, neuron n
    flag = dict[Neuron_Layer_Time, LpVariable]() # a[l,t,n] = activation flag for neuron n at layer l, time t
    activated = dict[Neuron_Layer_Time, LpVariable]()  # (p[l,t,n] >= threshold)
    cond = dict[Neuron_Layer_Time, LpVariable]()  # cond[l,t,n] = If (s_{l,n} ≤ t, 1, 0)

    # Variables and constraints for each layer ≥ 1
    for post_layer in range(1, len(n_layer_neurons)):
        for post_neuron in get_layer_neurons_iter(cfg, post_layer):
            assert tau * post_layer <= num_steps - 1, "Too high synaptic delay."
            spike_times[post_neuron, post_layer] = LpVariable(f"s_{post_neuron}_{post_layer}", tau * post_layer, num_steps - 1, cat=pulp.LpInteger) # Xi_1
            for t in range(num_steps):
                flag[post_neuron, post_layer, t] = LpVariable(f"a_{post_neuron}_{post_layer}_{t}", cat=pulp.LpBinary)
                activated[post_neuron, post_layer, t] = LpVariable(f"(p>=theta)_{post_neuron}_{post_layer}_{t}", cat=pulp.LpBinary)
    
    # Condition variables for previous layers, used in Xi_3
    for prev_layer in range(len(n_layer_neurons) - 1):
        for prev_neuron in get_layer_neurons_iter(cfg, prev_layer):
            _spike_time = spike_times[prev_neuron, prev_layer]
            for t in range(num_steps):
                _cond = cond[prev_neuron, prev_layer, t] = LpVariable(f"If_{prev_neuron}_{prev_layer}_{t}", cat=pulp.LpBinary)
                model += t + (1 - _cond) * M >= _spike_time
                model += t + 1 - _cond * M <= _spike_time
                # model += _spike_time >= t - tau + EPS - _cond * M

    # Potential accumulation and spike decision
    for post_layer in range(1, len(n_layer_neurons)):
        prev_layer = post_layer - 1
        for post_neuron in get_layer_neurons_iter(cfg, post_layer):
            p[post_neuron, post_layer, 0] = lpSum([])  # Xi_2
            for t in range(1, num_steps):
                ### Begin Xi_3
                expr = LpAffineExpression()
                for prev_neuron in get_layer_neurons_iter(cfg, prev_layer):
                    weight = weights_list[prev_layer][post_neuron[0], prev_neuron[0], prev_neuron[1]]
                    expr += weight * cond[prev_neuron, prev_layer, t-tau]
                p[post_neuron, post_layer, t] = expr
                ### End Xi_3
                
            ### Begin Xi_4
            # Big-M method for spike condition
            
            for t_prev in range(num_steps - 1):
                _p = p[post_neuron, post_layer, t_prev]
                _activated = activated[post_neuron, post_layer, t_prev]
                model += _p <= threshold - EPS + _activated * M # Not active
                model += _p >= threshold - (1 - _activated) * M # Active
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
            ### End Xi_4

            ### Begin Xi_5, Xi_6
            one_hot = LpAffineExpression()
            xi_5_term = LpAffineExpression()
            for t in range(tau * post_layer, num_steps - 1):
                _spike_cond = LpVariable(f"spike_{post_neuron}_{post_layer}_{t}", cat=pulp.LpBinary)
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
            ### End Xi_5, Xi_6

    target_spike_time = spike_times[(pred_orig, 0), len(n_layer_neurons) - 1]
    
    ### Begin Xi_8
    # Robustness constraint: output neuron 1 should not spike earlier than neuron 0
    not_robust = list[LpVariable]()
    for out_neuron in get_layer_neurons_iter(cfg, len(n_layer_neurons) - 1):
        if out_neuron[0] == pred_orig: continue
        
        _other_spike_time = spike_times[out_neuron, len(n_layer_neurons) - 1]
        # Ensure that output neuron 1 spikes at least 1 time step after output neuron 0
        _not_robust = LpVariable(f"not_robust_{out_neuron}", cat=pulp.LpBinary)
        model += _other_spike_time <= target_spike_time + (1 - _not_robust) * M
        model += _other_spike_time >= target_spike_time + EPS - _not_robust * M
        not_robust.append(_not_robust)
    model += lpSum(not_robust) >= 1  # Xi_8
    ### End Xi_8

    # Dummy objective
    model += target_spike_time

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True, logPath="log/milp.log")
    
    tx = time.time()
    model.solve(solver)
    total_time = time.time() - tx
    
    if None: # For debug
        forward(weights_list, s0_orig, original_result := list(), voltage_return := list())
        milp_result = list()
        milp_result.append([spike_times[neuron, 1].varValue for neuron in get_layer_neurons_iter(1)])
        milp_result.append([spike_times[neuron, 2].varValue for neuron in get_layer_neurons_iter(2)])
        print("Orig result:\t", [_array.tolist() for _array in original_result])
        print("MILP result:\t", milp_result)

    return model, total_time

def run_test(cfg: CFG):
    n_layer_neurons = cfg.n_layer_neurons
    num_steps = cfg.num_steps
    log_name = f"{cfg.log_name}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)

    weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
    images, labels, *_ = cfg.load_data_func(cfg)

    info("Data is loaded")

    if cfg.z3:
        run_z3(cfg, weights_list=weights_list, images=images)
    elif cfg.milp:
        run_milp(cfg, weights_list=weights_list, images=images)
    else:
        # Recursively find available adversarial attacks.
        def search_perts(
            img: TImage, delta: int, priority: np.ndarray, idx: int = 0, pert: TImage | None = None
        ) -> Generator[TImage, None, None]:
            # Initial case
            if pert is None:
                pert = np.zeros_like(img, dtype=img.dtype)

            # Last case
            if delta == 0:
                yield img + pert
            # Search must be terminated at the end of image.
            elif idx < len(priority):
                loc_2d = priority[idx]
                orig_time = int(img[loc_2d[0], loc_2d[1]])
                # Clamp delta at current location
                available_deltas = [*range(
                    -min(orig_time, delta), min((num_steps - 1) - orig_time, delta) + 1
                )]
                available_deltas.sort(key=abs, reverse=True)  # Search large perturbations first
                for delta_at_neuron in available_deltas:
                    new_pert = pert.copy()
                    new_pert[loc_2d[0], loc_2d[1]] += delta_at_neuron
                    yield from search_perts(
                        img, delta - abs(delta_at_neuron), priority, idx + 1, new_pert
                    )

        samples_no_list = list[int]()
        sampled_imgs = list[TImage]()
        sampled_labels = list[int]()
        orig_preds = list[int]()
        search_priority = list[np.ndarray[Any, np.dtype[np.int_]]]()
        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img: TImage = images[sample_no]
            label = labels[sample_no]
            sampled_imgs.append(img)
            sampled_labels.append(label)
            orig_preds.append(forward(cfg, weights_list, img, spike_times := []))
            input_grad = backward(cfg, weights_list, spike_times, img, label)[1]
            priority = np.dstack(np.unravel_index(np.abs(input_grad).ravel().argsort()[::-1], input_grad.shape))[0]
            search_priority.append(priority)
        info(f"Sampling is completed with {num_procs} samples.")

        # For each delta
        for delta in cfg.deltas:
            global check_sample_direct

            def check_sample_direct(
                sample: tuple[int, TImage, int, int, np.ndarray],
                weights_list: TWeightList = weights_list,
            ):
                sample_no, img, label, orig_pred, priority = sample

                info("Query processing starts")
                tx = time.time()
                sat_flag: bool = False
                adv_spk_times: list[list[np.ndarray[Any, np.dtype[np.float_]]]] = []
                n_counterexamples = 0
                
                for pertd_img in search_perts(img, delta, priority):
                    pert_pred = forward(cfg, weights_list, pertd_img, spk_times := [])
                    
                    adv_spk_times.append(spk_times)
                    last_layer_spk_times = spk_times[-1]
                    not_orig_mask = [
                        x for x in range(n_layer_neurons[-1]) if x != pert_pred
                    ]
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    if np.any(
                        last_layer_spk_times[not_orig_mask]
                        <= last_layer_spk_times[orig_pred]
                    ):
                        sat_flag = True
                        n_counterexamples += 1
                info(f"Checking done in time {time.time() - tx}")
                if sat_flag:
                    info(f"Not robust for sample {sample_no} and delta={delta}")
                elif sat_flag == False:
                    info(f"Robust for sample {sample_no} and delta={delta}.")
                info("")
                return sat_flag

            samples = zip(samples_no_list, sampled_imgs, sampled_labels, orig_preds, search_priority)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_direct, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_direct(sample)

        info("")


# %%
