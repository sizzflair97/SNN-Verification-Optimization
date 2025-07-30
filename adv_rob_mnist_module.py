from copy import deepcopy
from multiprocessing import Pool
from random import sample as random_sample
from random import seed
from typing import Any, TypeAlias, TypeVar
from collections.abc import Generator
import time, logging, typing, pdb
import numpy as np
import pulp
from pulp import LpVariable, LpAffineExpression, lpSum
from torch import mode
from z3 import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import CFG
from utils.debug import info
from utils.mnist_net import forward, backward, test_weights, prepare_weights
from mnist import MNIST


def load_mnist() -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    # Parameter setting
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading MNIST dataset
    mndata = MNIST("data/mnist/MNIST/raw/")

    Images, Labels = mndata.load_training()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(
                np.floor(
                    (GrayLevels - Images[i].reshape(28, 28))
                    * (num_steps - 1)
                    / GrayLevels
                ).astype(int)
            )
            labels.append(cats.index(Labels[i]))
    Images, Labels = mndata.load_testing()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images_test.append(
                np.floor(
                    (GrayLevels - Images[i].reshape(28, 28))
                    * (num_steps - 1)
                    / GrayLevels
                ).astype(int)
            )
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    # images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))

    return images, labels, images_test, labels_test

def run_z3(cfg: CFG, *, weights_list: TWeightList, images: TImageBatch):
    S = Solver()
    spike_times = gen_spike_times()
    weights = gen_weights(weights_list)

    # Load equations.
    eqn_path = (
        f"eqn/eqn_{num_steps}_{'_'.join([str(i) for i in n_layer_neurons])}.txt"
    )
    if not load_expr or not os.path.isfile(eqn_path):
        node_eqns = gen_node_eqns(weights, spike_times)
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

            for in_neuron in get_layer_neurons_iter(input_layer):
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
            for out_neuron in get_layer_neurons_iter(last_layer):
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
        orig_preds.append(forward(weights_list, img))
    info(f"Sampling is completed with {num_procs} samples.")
    return samples_no_list, sampled_imgs, orig_preds

def run_milp(cfg: CFG, *, weights_list:TWeightList, images: TImageBatch):
    samples_no_list, sampled_imgs, orig_preds = sample_images_and_predictions(cfg, weights_list, images)
    for sample_no, img, orig_pred in zip(samples_no_list, sampled_imgs, orig_preds):
        info(f"Sample {sample_no} with original prediction {orig_pred} is processed.")
        _model = run_milp_single(weights_list, img, orig_pred)
        print("Status:", pulp.LpStatus[_model.status])
    
def run_milp_single(weights_list:TWeightList, s0_orig:TImage, pred_orig:int) -> pulp.LpProblem:
    T = 5  # time steps
    theta = 1.0  # spike threshold
    tau = 1  # synaptic delay
    delta = 1  # perturbation budget

    model = pulp.LpProblem("MultiLayer_SNN_Verification", pulp.LpMinimize)
    M = 1000  # Large constant for MILP
    epsilon = LpVariable(f"epsilon", lowBound = 0)
    model += epsilon == 1e-12

    # Variables: spike time and perturbation (input layer)
    spike_times = dict[tuple[NodeIdx, LayerIdx], LpVariable]()  # s[l,n] = spike time
    neuron_perturbation = dict[NodeIdx, LpVariable]()  # d_n = perturbation for neuron n
    for neuron in get_layer_neurons_iter(0):
        spike_times[neuron, 0] = LpVariable(f"s_0_{neuron}", 0, T - 1, cat="Integer") # Xi_1
        ## Begin Xi_7
        neuron_perturbation[neuron] = LpVariable(f"d_{neuron}", 0, T - 1, cat="Integer")
        model += spike_times[neuron, 0] - s0_orig[neuron] <= neuron_perturbation[neuron]
        model += s0_orig[neuron] - spike_times[neuron, 0] <= neuron_perturbation[neuron]
        ## End Xi_7
    model += lpSum(neuron_perturbation.values()) <= delta

    # Intermediate variables
    p = dict[Neuron_Layer_Time, LpVariable]()  # p[l,t,n] = potential at layer l, time t, neuron n
    flag = dict[Neuron_Layer_Time, LpVariable]() # a[l,t,n] = activation flag for neuron n at layer l, time t
    activated = dict[Neuron_Layer_Time, LpVariable]()  # a[l,t,n] = activation flag for neuron n at layer l, time t
    cond = dict[Neuron_Layer_Time, LpVariable]()  # cond[l,t,n] = condition variable for neuron n at layer l, time t

    # Variables and constraints for each layer ≥ 1
    for post_layer in range(1, len(n_layer_neurons)):
        for post_neuron in get_layer_neurons_iter(post_layer):
            spike_times[post_neuron, post_layer] = LpVariable(f"s_{post_layer}_{post_neuron}", tau * post_layer, T - 1, cat="Integer") # Xi_1
            for t in range(T):
                p[post_neuron, post_layer, t] = LpVariable(f"p_{post_layer}_{t}_{post_neuron}")
                flag[post_neuron, post_layer, t] = LpVariable(f"a_{post_layer}_{t}_{post_neuron}", 0, 1, cat="Binary")
                activated[post_neuron, post_layer, t] = LpVariable(f"active_{post_layer}_{t}_{post_neuron}", 0, 1, cat="Binary")
            model += p[post_neuron, post_layer, 0] == 0  # Xi_2
    
    # Condition variables for previous layers, used in Xi_3
    for prev_layer in range(len(n_layer_neurons) - 1):
        for prev_neuron in get_layer_neurons_iter(prev_layer):
            for t in range(1, T):
                cond[prev_neuron, t, prev_layer] = LpVariable(f"cond_{prev_layer}_{t-tau}_{prev_neuron}", 0, 1, cat="Binary")

    # Potential accumulation and spike decision
    for post_layer in range(1, len(n_layer_neurons)):
        prev_layer = post_layer - 1
        for post_neuron in get_layer_neurons_iter(post_layer):
            model += flag[post_neuron, post_layer, 0] == 0  # Initial activation flag
            for t in range(1, T):
                ### Begin Xi_3
                expr = list[LpAffineExpression]()
                for prev_neuron in get_layer_neurons_iter(prev_layer):
                    _spike_time = spike_times[prev_neuron, prev_layer]
                    model += _spike_time <= t - tau + (1 - cond[prev_neuron, t, prev_layer]) * M
                    model += _spike_time >= t - tau + epsilon - cond[prev_neuron, t, prev_layer] * M
                    weight = weights_list[prev_layer][post_neuron[0], prev_neuron[0], prev_neuron[1]]
                    expr.append(weight * cond[prev_neuron, t, prev_layer])
                model += p[post_neuron, post_layer, t] == lpSum(expr)
                ### End Xi_3
                
                ### Begin Xi_4
                # Big-M method for spike condition
                _activated = activated[post_neuron, post_layer, t]
                model += p[post_neuron, post_layer, t] <= theta - epsilon + _activated * M # Not active
                model += p[post_neuron, post_layer, t] >= theta - (1 - _activated) * M # Active

                _flag = flag[post_neuron, post_layer, t]
                for t_prev in range(t):
                    model += _flag >= activated[post_neuron, post_layer, t_prev]  # Activation flag should be true if any previous time was active
                model += _flag <= lpSum([activated[post_neuron, post_layer, t_prev] for t_prev in range(t)]) # activation is false if no previous time was active
                ### End Xi_4

            ### Begin Xi_5, Xi_6
            spike_cond = dict[int, LpVariable]()   
            for t in range(tau * post_layer, T - 1):
                _spike_cond = spike_cond[t] = LpVariable(f"xi_5_lhs_{post_layer}_{t}_{post_neuron}", 0, 1, cat="Binary")
                model += _spike_cond <= 1 - flag[post_neuron, post_layer, t]
                model += _spike_cond <= activated[post_neuron, post_layer, t]
                model += _spike_cond >= (1 - flag[post_neuron, post_layer, t]) + activated[post_neuron, post_layer, t] - 1
            xi_5_term = lpSum([t * spike_cond[t] for t in range(tau * post_layer, T - 1)])
            xi_6_term = (T - 1) * (1 - flag[post_neuron, post_layer, T - 1])
            model += spike_times[post_neuron, post_layer] ==  xi_5_term + xi_6_term
            ### End Xi_5, Xi_6

    # Robustness constraint: output neuron 1 should not spike earlier than neuron 0
    target_spike_time = spike_times[(pred_orig, 0), len(n_layer_neurons) - 1]
    not_robust = list[LpVariable]()
    for out_neuron in get_layer_neurons_iter(len(n_layer_neurons) - 1):
        _other_spike_time = spike_times[out_neuron, len(n_layer_neurons) - 1]
        # Ensure that output neuron 1 spikes at least 1 time step after output neuron 0
        if out_neuron[0] != pred_orig:
            _not_robust = LpVariable(f"not_robust_{out_neuron}", 0, 1, cat="Binary")
            model += _other_spike_time <= target_spike_time + (1 - _not_robust) * M
            model += _other_spike_time >= target_spike_time + epsilon - _not_robust * M
            not_robust.append(_not_robust)
    model += lpSum(not_robust) >= 1  # Xi_8

    # Dummy objective
    model += target_spike_time

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=True))
    return model

def run_test(cfg: CFG):
    log_name = f"{cfg.log_name}_{num_steps}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)

    weights_list = prepare_weights(subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist()

    info("Data is loaded")

    if cfg.z3:
        run_z3(cfg, weights_list=weights_list, images=images)
    elif cfg.milp:
        run_milp(cfg, weights_list=weights_list, images=images)
    else:
        # Recursively find available adversarial attacks.
        def search_perts(
            img: TImage, delta: int, loc: int = 0, pert: TImage | None = None
        ) -> Generator[TImage, None, None]:
            # Initial case
            if pert is None:
                pert = np.zeros_like(img, dtype=img.dtype)

            # Last case
            if delta == 0:
                yield img + pert
            # Search must be terminated at the end of image.
            elif loc < n_layer_neurons[0]:
                loc_2d = (loc // layer_shapes[0][1], loc % layer_shapes[0][1])
                orig_time = int(img[loc_2d])
                # Clamp delta at current location
                available_deltas = range(
                    -min(orig_time, delta), min((num_steps - 1) - orig_time, delta) + 1
                )
                for delta_at_neuron in available_deltas:
                    new_pert = pert.copy()
                    new_pert[loc_2d] += delta_at_neuron
                    yield from search_perts(
                        img, delta - abs(delta_at_neuron), loc + 1, new_pert
                    )

        samples_no_list: list[int] = []
        sampled_imgs: list[TImage] = []
        sampled_labels: list[int] = []
        orig_preds: list[int] = []
        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img: TImage = images[sample_no]
            label = labels[sample_no]
            sampled_imgs.append(img)
            sampled_labels.append(label)
            orig_preds.append(forward(weights_list, img))
        info(f"Sampling is completed with {num_procs} samples.")

        # For each delta
        for delta in cfg.deltas:
            global check_sample_non_smt

            def check_sample_non_smt(
                sample: tuple[int, TImage, int, int],
                adv_train: bool = False,
                weights_list: TWeightList = weights_list,
            ):
                sample_no, img, label, orig_pred = sample

                info("Query processing starts")
                tx = time.time()
                sat_flag: bool = False
                adv_spk_times: list[list[np.ndarray[Any, np.dtype[np.float_]]]] = []
                n_counterexamples = 0
                for pertd_img in search_perts(img, delta):
                    pert_pred = forward(weights_list, pertd_img, spk_times := [])
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
                        if not adv_train:
                            break
                        n_counterexamples += 1
                info(f"Checking done in time {time.time() - tx}")
                if sat_flag:
                    if adv_train:
                        info(
                            f"Not robust for sample {sample_no} and delta={delta} with {n_counterexamples} counterexamples."
                        )
                        info(f"Start adversarial training.")
                        updated_weights_list = weights_list
                        for spk_times in adv_spk_times:
                            updated_weights_list = backward(
                                updated_weights_list, spk_times, img, label
                            )
                        test_weights(updated_weights_list, load_mnist)
                        new_orig_pred = forward(updated_weights_list, img)
                        new_sample = (*sample[:3], new_orig_pred)
                        info(
                            f"Completed adversarial training. Checking robustness again."
                        )
                        check_sample_non_smt(
                            new_sample,
                            adv_train=False,
                            weights_list=updated_weights_list,
                        )
                    else:
                        info(f"Not robust for sample {sample_no} and delta={delta}")
                elif sat_flag == False:
                    info(f"Robust for sample {sample_no} and delta={delta}.")
                info("")
                return sat_flag

            samples = zip(samples_no_list, sampled_imgs, sampled_labels, orig_preds)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_non_smt, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_non_smt(sample)

        info("")


# %%
