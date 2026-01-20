from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from random import sample as random_sample
from random import seed
from typing import Any
from collections.abc import Generator
import time, logging, pdb
import numpy as np
import pulp, torch
from pulp import LpVariable, LpAffineExpression, lpSum
from z3 import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import CFG
from utils.debug import info
from utils.mnist_net import forward, backward, prepare_weights

import sys

sys.setrecursionlimit(3000)

# from utils.ann import SimpleANN, get_gradient, load_ann


def run_z3(cfg: CFG, *, weights_list: TWeightList, images: TImageBatch):
    n_layer_neurons = cfg.n_layer_neurons
    S = Solver()
    spike_times = gen_spike_times(cfg)
    weights = gen_weights(cfg, weights_list)

    # Load equations.
    eqn_path = f"eqn/eqn_{cfg.num_steps}_{'_'.join([str(i) for i in n_layer_neurons])}.txt"
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
                delta_pos += relu(spike_times[in_neuron, input_layer] - int(img[in_neuron]))
                delta_neg += relu(int(img[in_neuron]) - spike_times[in_neuron, input_layer])
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
                    op.append(spike_times[out_neuron, last_layer] <= spike_times[orig_neuron, last_layer])
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
                info(f"Unknown at sample {sample_no} for reason {S_instance.reason_unknown()}")
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


def sample_images_and_predictions(cfg: CFG, weights_list: TWeightList, images: TImageBatch):
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


def run_milp(
    cfg: CFG,
    *,
    weights_list: TWeightList,
    images: TImageBatch,
    MAP={pulp.LpStatusOptimal: "Not Robust", pulp.LpStatusInfeasible: "Robust"},
):
    samples_no_list, sampled_imgs, orig_preds = sample_images_and_predictions(cfg, weights_list, images)
    for delta in cfg.deltas:
        info(f"Delta: {delta}")
        for sample_no, img, orig_pred in zip(samples_no_list, sampled_imgs, orig_preds):
            _model, _tx = run_milp_single(cfg, weights_list, img, orig_pred, delta=delta)
            info(f"Sample {sample_no}\t|\ttime: {_tx:.13f}\t|\tstatus: {MAP[_model.status]}")


def run_milp_single(
    cfg: CFG, weights_list: TWeightList, s0_orig: TImage, pred_orig: int, delta: int = 1
) -> tuple[pulp.LpProblem, float]:
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
        spike_times[neuron, 0] = LpVariable(f"s_0_{neuron}", 0, num_steps - 1, cat=pulp.LpInteger)  # Xi_1
        # Begin Xi_7
        neuron_perturbation[neuron] = LpVariable(f"d_{neuron}", 0, num_steps - 1, cat=pulp.LpInteger)
        model += spike_times[neuron, 0] - s0_orig[neuron] <= neuron_perturbation[neuron]
        model += s0_orig[neuron] - spike_times[neuron, 0] <= neuron_perturbation[neuron]
        # End Xi_7
    model += lpSum(neuron_perturbation.values()) <= delta

    # Intermediate variables
    p = dict[Neuron_Layer_Time, LpAffineExpression]()  # p[l,t,n] = potential at layer l, time t, neuron n
    flag = dict[Neuron_Layer_Time, LpVariable]()  # a[l,t,n] = activation flag for neuron n at layer l, time t
    activated = dict[Neuron_Layer_Time, LpVariable]()  # (p[l,t,n] >= threshold)
    cond = dict[Neuron_Layer_Time, LpVariable]()  # cond[l,t,n] = If (s_{l,n} ≤ t, 1, 0)

    # Variables and constraints for each layer ≥ 1
    for post_layer in range(1, len(n_layer_neurons)):
        for post_neuron in get_layer_neurons_iter(cfg, post_layer):
            assert tau * post_layer <= num_steps - 1, "Too high synaptic delay."
            spike_times[post_neuron, post_layer] = LpVariable(
                f"s_{post_neuron}_{post_layer}", tau * post_layer, num_steps - 1, cat=pulp.LpInteger
            )  # Xi_1
            for t in range(num_steps):
                flag[post_neuron, post_layer, t] = LpVariable(f"a_{post_neuron}_{post_layer}_{t}", cat=pulp.LpBinary)
                activated[post_neuron, post_layer, t] = LpVariable(
                    f"(p>=theta)_{post_neuron}_{post_layer}_{t}", cat=pulp.LpBinary
                )

    # Condition variables for previous layers, used in Xi_3
    for prev_layer in range(len(n_layer_neurons) - 1):
        for prev_neuron in get_layer_neurons_iter(cfg, prev_layer):
            _spike_time = spike_times[prev_neuron, prev_layer]
            for t in range(num_steps):
                _cond = cond[prev_neuron, prev_layer, t] = LpVariable(
                    f"If_{prev_neuron}_{prev_layer}_{t}", cat=pulp.LpBinary
                )
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
                    expr += weight * cond[prev_neuron, prev_layer, t - tau]
                p[post_neuron, post_layer, t] = expr
                ### End Xi_3

            ### Begin Xi_4
            # Big-M method for spike condition

            for t_prev in range(num_steps - 1):
                _p = p[post_neuron, post_layer, t_prev]
                _activated = activated[post_neuron, post_layer, t_prev]
                model += _p <= threshold - EPS + _activated * M  # Not active
                model += _p >= threshold - (1 - _activated) * M  # Active
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
        if out_neuron[0] == pred_orig:
            continue

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

    if True:  # For debug
        for v in model.variables():
            print(v.name, v.varValue)

        forward(cfg, weights_list, s0_orig, original_result := list(), voltage_return := list())
        milp_result = list()
        milp_result.append([spike_times[neuron, 1].varValue for neuron in get_layer_neurons_iter(cfg, 1)])
        milp_result.append([spike_times[neuron, 2].varValue for neuron in get_layer_neurons_iter(cfg, 2)])
        print("Orig result:\t", [_array.tolist() for _array in original_result])
        print("MILP result:\t", milp_result)

    return model, total_time


# Recursively find available adversarial attacks.
def search_perts(
    cfg: CFG,
    img: TImage,
    delta: int,
    priority: np.ndarray,
    grad_sign: np.ndarray,
    prefix_set: set[frozenset[tuple[int, int]]],
    prefix_lengths: set[int],
    idx: int = 0,
    pert: TImage | None = None,
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
        available_deltas = [*range(-min(orig_time, delta), min((cfg.num_steps - 1) - orig_time, delta) + 1)]
        if grad_sign[loc_2d[0], loc_2d[1]] > 0:
            available_deltas.reverse()  # If gradient is negative, try negative perturbation first:
            # to find adversarial examples faster.
        for delta_at_neuron in available_deltas:
            new_pert = pert.copy()
            new_pert[loc_2d[0], loc_2d[1]] += delta_at_neuron
            yield from search_perts(
                cfg,
                img,
                delta - abs(delta_at_neuron),
                priority,
                grad_sign,
                prefix_set,
                prefix_lengths,
                idx + 1,
                new_pert,
            )


# Recursively find available adversarial attacks.
def search_perts_psm(
    cfg: CFG,
    img: TImage,
    delta: int,
    priority: np.ndarray,
    grad_sign: np.ndarray,
    prefix_set: set[
        tuple[
            frozenset[tuple[int, int]],
            frozenset[tuple[int, int]],
        ]
    ],
    prefix_lengths: set[int],
    idx: int = 0,
    pert: TImage | None = None,
) -> Generator[TImage, None, None]:
    # Initial case
    if pert is None:
        pert = np.zeros_like(img, dtype=img.dtype)

    # Last case
    if delta == 0:
        img_pert = img + pert
        prefix = frozenset()
        t = 0
        while len(prefix) < max(prefix_lengths):
            rows, cols = np.nonzero(img_pert <= t)
            prefix = frozenset(zip(rows, cols))
            # prefix = frozenset(
            #     (i, j)
            #     for i in range(28)
            #     for j in range(28)
            #     if img_pert[i, j] <= t
            # )
            if prefix in prefix_set:
                info(f"Prefix {prefix} is in prefix_set, pruning search.")
                return
            t += 1
        else:
            yield img_pert
    # Search must be terminated at the end of image.
    elif idx < len(priority):
        loc_2d = priority[idx]
        orig_time = int(img[loc_2d[0], loc_2d[1]])
        # Clamp delta at current location
        available_deltas = [*range(-min(orig_time, delta), min((cfg.num_steps - 1) - orig_time, delta) + 1)]
        if grad_sign[loc_2d[0], loc_2d[1]] > 0:
            available_deltas.reverse()  # If gradient is negative, try negative perturbation first:
            # to find adversarial examples faster.
        for delta_at_neuron in available_deltas:
            new_pert = pert.copy()
            new_pert[loc_2d[0], loc_2d[1]] += delta_at_neuron
            yield from search_perts_psm(
                cfg,
                img,
                delta - abs(delta_at_neuron),
                priority,
                grad_sign,
                prefix_set,
                prefix_lengths,
                idx + 1,
                new_pert,
            )


def get_bottom_two_diff(nums):
    # 요소가 2개 미만인 경우 처리
    if len(nums) < 2:
        return None

    # 초기값을 무한대(infinity)로 설정
    first_min = second_min = float("inf")

    for n in nums:
        if n < first_min:
            # 새로운 최솟값을 찾으면 기존 최솟값은 두 번째가 됨
            second_min = first_min
            first_min = n
        elif n < second_min:
            # 최솟값보다는 크지만 두 번째보다는 작은 경우
            second_min = n

    # 두 번째 작은 값에서 가장 작은 값을 뺌 (양수 결과)
    return second_min - first_min


def run_test(cfg: CFG):
    n_layer_neurons = cfg.n_layer_neurons
    log_name = f"{cfg.log_name}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)

    weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
    images, labels, *_ = cfg.load_data_func(cfg)
    if cfg.manual_indices is not None:
        images = images[cfg.manual_indices]
        labels = labels[cfg.manual_indices]

    info("Data is loaded")

    if cfg.z3:
        run_z3(cfg, weights_list=weights_list, images=images)
    elif cfg.milp:
        run_milp(cfg, weights_list=weights_list, images=images)
    else:
        # ann_path = Path("models") / "ann" / f"{cfg.subtype}_mlp_{n_layer_neurons[1]}.pth"
        # ann = load_ann(ann_path, n_hidden_neurons=n_layer_neurons[1])
        samples_no_list = list[int]()
        sampled_imgs = list[TImage]()
        sampled_labels = list[int]()
        orig_preds = list[int]()
        search_schedule = list[tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]]()

        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            img: TImage = images[sample_no]
            label = labels[sample_no]
            orig_pred = forward(cfg, weights_list, img, layers_firing_time := [])
            if len(np.argwhere(layers_firing_time[-1] == np.min(layers_firing_time[-1]))[0]) != 1:
                info(f"Multiple output neurons fired first for sample {sample_no}, skipping this sample.")
                continue

            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            orig_preds.append(orig_pred)
            sampled_imgs.append(img)
            sampled_labels.append(label)
            if cfg.adv_attack:
                input_grad = backward(cfg, weights_list, layers_firing_time, img, label, relative_target_offset=-1)[1]
                # input_grad = get_gradient(ann, torch.tensor(img, dtype=torch.float32).view(1, 28*28), torch.tensor([label], dtype=torch.long)).view(28,28).numpy()
                priority = np.dstack(np.unravel_index((-np.abs(input_grad)).ravel().argsort(), input_grad.shape))[0]
            else:
                input_grad = np.ones_like(img, dtype=np.float32)
                priority = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].reshape(2, -1).T
            search_schedule.append((priority, np.sign(input_grad)))
        info(f"Sampling is completed with {len(samples_no_list)} samples.")

        # For each delta
        for delta in cfg.deltas:
            global check_sample_direct

            def check_sample_direct(
                sample: tuple[int, TImage, int, int, tuple[np.ndarray, np.ndarray]],
                weights_list: TWeightList = weights_list,
            ):
                sample_no, img, label, orig_pred, (priority, sign) = sample
                info("BnB-based Query processing (Dual-side Perturbation)")
                tx = time.time()

                # [STEP 1] Baseline Forward 실행
                base_spks = []
                forward(cfg, weights_list, img, base_spks)
                base_times = base_spks[-1]

                num_classes = weights_list[-1].shape[0]
                found_adversarial = [False]

                synaptic_delay = 2

                target_time = base_times[orig_pred]
                # Non-target 중 가장 빨리 터지는 놈 혹은 target_time 중 더 빠른 것을 기준점으로 잡음
                min_non_target = np.min([base_times[i] for i in range(num_classes) if i != orig_pred])
                # T_ref: 이 시간 이후에 도착하는 입력 스파이크는 결과를 뒤집기에 너무 늦음
                t_ref = min(target_time, min_non_target)

                # 메모이제이션 테이블: (pos, eps, allow_pos) -> min_d
                memo = {}

                def bnb_dfs(current_img, pixel_pos, rem_neg, rem_pos):

                    # print(f"At pixel_pos {pixel_pos}, rem_neg {rem_neg}, rem_pos {rem_pos}")

                    if found_adversarial[0]:
                        return

                    # [STEP 1] 현재 상태의 출력 시간 확인
                    current_spks = []
                    forward(cfg, weights_list, current_img, current_spks)
                    current_last_layer = current_spks[-1]

                    target_time = current_last_layer[orig_pred]
                    min_non_target_time = np.min([current_last_layer[i] for i in range(num_classes) if i != orig_pred])

                    if min_non_target_time <= target_time:
                        print("Adversarial found at leaf node.")
                        found_adversarial[0] = True
                        return

                    if pixel_pos == len(active_priority):
                        # print("Reached leaf node without finding adversarial.")
                        return

                    if rem_neg == 0 and rem_pos == 0:
                        return

                    idx_x, idx_y = active_priority[pixel_pos]

                    orig_val = current_img[idx_x, idx_y]
                    max_t = cfg.num_steps  # SNN 시뮬레이션의 최대 타임스텝

                    # ---------------------------------------------------------
                    # [경우의 수 나누기 - Branching]
                    # ---------------------------------------------------------

                    # 1. 음수 섭동 (모든 중간 값 v < orig_val 시도)
                    if rem_neg >= 1:
                        for v in range(int(orig_val)):
                            cost = int(orig_val - v)
                            if rem_neg >= cost:  # and (orig_val - cost + synaptic_delay <= target_time):
                                next_img = current_img.copy()
                                next_img[idx_x, idx_y] = v

                                # 음수 섭동 후에도 양수 권한을 열어둘지 닫을지 결정
                                bnb_dfs(next_img.copy(), pixel_pos + 1, rem_neg - cost, rem_pos)

                                if found_adversarial[0]:
                                    return

                    # 2. 양수 섭동 (모든 중간 값 v > orig_val 시도)
                    if rem_pos >= 1:  # and (orig_val + synaptic_delay <= target_time):
                        for v in range(int(orig_val) + 1, max_t):
                            cost = int(v - orig_val)
                            if rem_pos >= cost:
                                next_img = current_img.copy()
                                next_img[idx_x, idx_y] = v

                                # 양수 섭동을 했으므로, 이후 단계에서도 권한을 유지하거나 여기서 닫음
                                bnb_dfs(next_img.copy(), pixel_pos + 1, rem_neg, rem_pos - cost)

                                if found_adversarial[0]:
                                    return

                    # 3. 섭동 없음 (No Perturbation)
                    # 미래에 양수 섭동 권한을 유지할지, 여기서 닫을지 선택
                    bnb_dfs(current_img.copy(), pixel_pos + 1, rem_neg, rem_pos)

                    if found_adversarial[0]:
                        return

                for i in range(delta + 1):

                    rem_neg = i
                    rem_pos = delta - i
                    # [STEP 2] 인과율 필터링 (Active Set 구성)
                    # 입력 스파이크 시간(orig_val)이 (기준 시간 + 예산)보다 크면 절대 개입 불가
                    active_priority = []

                    for px, py in priority:
                        orig_val = img[px, py]
                        # 입력 스파이크를 최대로 당겨도(orig_val - delta) 기준 시간(t_ref)보다 늦으면 배제

                        if (orig_val - rem_neg + synaptic_delay <= t_ref) or (
                            orig_val + synaptic_delay <= t_ref and rem_pos > 0
                        ):
                            active_priority.append((px, py))

                    info(
                        f"Filtered pixels: {len(priority)} -> {len(active_priority)} (Reduced by {len(priority)-len(active_priority)})"
                    )

                    bnb_dfs(img.copy(), 0, rem_neg, rem_pos)

                # 5. 결과 로깅 및 반환
                info(f"Checking done in time {time.time() - tx}")
                if found_adversarial[0]:
                    info(f"Not robust for sample {sample_no} and delta={delta}")
                else:
                    info(f"Robust for sample {sample_no} and delta={delta}.")
                info("")

                return found_adversarial[0]

            samples = zip(samples_no_list, sampled_imgs, sampled_labels, orig_preds, search_schedule)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_direct, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_direct(sample)

        info("")
