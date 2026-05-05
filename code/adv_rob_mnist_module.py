from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from random import sample as random_sample
from random import seed, Random
import hashlib
from typing import Any
from collections.abc import Generator
import time, logging, pdb

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


import numpy as np

from threadpoolctl import threadpool_limits

threadpool_limits(1)

np.show_config()

import pulp, torch
from pulp import LpVariable, LpAffineExpression, lpSum
from z3 import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import CFG
from utils.debug import info
from utils.mnist_net import forward, backward, prepare_weights
from bnb_ibp import (
    ibp_prove_robust,
    ibp_prove_robust_budgeted,
    ibp_prove_robust_beta,
)

import sys

sys.setrecursionlimit(10000)

# from utils.ann import SimpleANN, get_gradient, load_ann

debug = False


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

            # Output property: at least one non-target fires before target.
            # argmin tie-breaking: 동일 spike time일 때 낮은 인덱스가 승리.
            tx = time.time()
            op = []
            last_layer = len(n_layer_neurons) - 1
            for out_neuron in get_layer_neurons_iter(cfg, last_layer):
                if out_neuron != orig_neuron:
                    if out_neuron[0] < orig_neuron[0]:
                        # 낮은 인덱스: tie시 이 뉴런이 argmin 승리
                        op.append(spike_times[out_neuron, last_layer] <= spike_times[orig_neuron, last_layer])
                    else:
                        # 높은 인덱스: strict less-than 필요
                        op.append(spike_times[out_neuron, last_layer] < spike_times[orig_neuron, last_layer])
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
    # ==================================================================
    # [수정 Step 3] run_milp 이 전체 샘플을 순회하도록 변경
    # ------------------------------------------------------------------
    # 기존: `if sample_no == 1639` 조건으로 한 샘플만 풀었음.
    # 변경: cfg 에 지정된 모든 샘플에 대해 MILP 를 풀고 결과를 dict 로
    #       반환한다. 이렇게 하면 BnB 결과와 동일 샘플에서 1:1 비교가
    #       가능해지며, 검증 harness 가 soundness / completeness 를
    #       직접 체크할 수 있다.
    # 반환값:
    #   verdicts: {sample_no -> bool}  (True = Not Robust, False = Robust)
    # ==================================================================
    samples_no_list, sampled_imgs, orig_preds = sample_images_and_predictions(cfg, weights_list, images)
    verdicts: dict[tuple[int, int], bool] = {}
    for delta in cfg.deltas:
        info(f"Delta: {delta}")
        for sample_no, img, orig_pred in zip(samples_no_list, sampled_imgs, orig_preds):
            _model, _tx = run_milp_single(cfg, weights_list, img, orig_pred, delta=delta, verbose=False)
            status_str = MAP.get(_model.status, f"Unknown({_model.status})")
            info(f"Sample {sample_no}\t|\tdelta: {delta}\t|\ttime: {_tx:.6f}\t|\tstatus: {status_str}")
            # MILP 가 feasible (Optimal) → adversarial 존재 → Not Robust (True)
            # MILP 가 infeasible → adversarial 없음 → Robust (False)
            if _model.status == pulp.LpStatusOptimal:
                verdicts[(sample_no, delta)] = True
            elif _model.status == pulp.LpStatusInfeasible:
                verdicts[(sample_no, delta)] = False
            else:
                # Unknown 상태는 검증 harness 에서 건너뛰도록 None 을 기록.
                verdicts[(sample_no, delta)] = None  # type: ignore
    return verdicts


def run_milp_single(
    cfg: CFG,
    weights_list: TWeightList,
    s0_orig: TImage,
    pred_orig: int,
    delta: int = 1,
    verbose: bool = False,  # [수정 Step 3-보조] 검증 harness 에서 대량 호출 시 로그 억제 옵션
    # ------------------------------------------------------------------
    # [수정 Step 4-보강] perturbable_pixels: 섭동을 허용할 픽셀 집합
    # None  → 모든 픽셀 허용 (기존 거동과 동일)
    # set[(x,y)] → 포함된 픽셀만 변수로, 나머지는 s0_orig 값에 고정.
    # 검증 harness 에서 축소된 sub-problem 을 풀기 위한 용도.
    # ------------------------------------------------------------------
    perturbable_pixels: set[tuple[int, int]] | None = None,
) -> tuple[pulp.LpProblem, float]:
    n_layer_neurons = cfg.n_layer_neurons
    num_steps = cfg.num_steps
    tau = 1  # synaptic delay

    model = pulp.LpProblem("MultiLayer_SNN_Verification", pulp.LpMinimize)
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
        # ------------------------------------------------------------------
        # [수정 Step 4-보강] perturbable_pixels 제약
        # 해당 픽셀이 perturbable 집합에 없으면 s0 를 원본값에 고정 (d=0)
        # ------------------------------------------------------------------
        if perturbable_pixels is not None and neuron not in perturbable_pixels:
            model += spike_times[neuron, 0] == int(s0_orig[neuron])
            model += neuron_perturbation[neuron] == 0
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
                model += t + (1 - _cond) * M_time >= _spike_time
                model += t + 1 - _cond * M_time <= _spike_time
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
                model += _p <= threshold + _activated * M_act[post_layer]  # activated=0 → p <= threshold
                model += _p >= threshold + EPS - (1 - _activated) * M_act[post_layer]  # activated=1 → p > threshold
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
    # Robustness constraint: at least one non-target neuron must fire before the target.
    # argmin tie-breaking: 동일 spike time일 때 낮은 인덱스가 승리.
    not_robust = list[LpVariable]()
    for out_neuron in get_layer_neurons_iter(cfg, len(n_layer_neurons) - 1):
        if out_neuron[0] == pred_orig:
            continue

        _other_spike_time = spike_times[out_neuron, len(n_layer_neurons) - 1]
        _not_robust = LpVariable(f"not_robust_{out_neuron}", cat=pulp.LpBinary)
        if out_neuron[0] < pred_orig:
            # 낮은 인덱스: tie시 argmin이 이 뉴런을 선택 → <= 유지
            model += _other_spike_time <= target_spike_time + (1 - _not_robust) * M_time
            model += _other_spike_time >= target_spike_time + EPS - _not_robust * M_time
        else:
            # 높은 인덱스: tie시 target이 승리 → strict less-than (정수이므로 <= target-1)
            model += _other_spike_time <= (target_spike_time - 1) + (1 - _not_robust) * M_time
            model += _other_spike_time >= (target_spike_time - 1) + EPS - _not_robust * M_time
        not_robust.append(_not_robust)
    model += lpSum(not_robust) >= 1  # Xi_8
    ### End Xi_8

    # Dummy objective
    model += target_spike_time

    # Solve
    # [수정 Step 3-보조] verbose 플래그에 따라 solver 로그 억제
    solver = pulp.PULP_CBC_CMD(msg=verbose, logPath="log/milp.log")

    tx = time.time()
    model.solve(solver)
    total_time = time.time() - tx

    if verbose:  # For debug — verbose 일 때만 dump
        for v in model.variables():
            print(v.name, v.varValue)

        forward(cfg, weights_list, s0_orig, original_result := list(), voltage_return := list())
        milp_result = list()
        # 중간 레이어 결과는 레이어 개수 의존이므로 n_layer_neurons 에 맞게 수집.
        for layer_idx in range(1, len(cfg.n_layer_neurons)):
            milp_result.append(
                [spike_times[neuron, layer_idx].varValue for neuron in get_layer_neurons_iter(cfg, layer_idx)]
            )
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

                num_steps = cfg.num_steps
                num_classes = weights_list[-1].shape[0]
                n_hidden = cfg.n_layer_neurons[1]
                w1 = weights_list[0]  # (n_hidden, 28, 28)
                w2 = weights_list[1]  # (n_output, n_hidden, 1)
                w2_flat = w2[:, :, 0]  # (n_output, n_hidden)
                found_adversarial = [False]
                synaptic_delay = len(cfg.n_layer_neurons) - 1

                # ============================================================
                # Baseline forward: 초기 voltage 상태 계산
                # ============================================================
                base_spks = []
                base_voltages = []
                forward(cfg, weights_list, img, base_spks, base_voltages)
                base_times = base_spks[-1]

                # ============================================================
                # Active set filter (legacy/efficient/sound 모드)
                # ============================================================
                _legacy_active = os.environ.get("SNN_BNB_LEGACY_ACTIVE_SET", "0") == "1"
                _efficient = os.environ.get("SNN_BNB_EFFICIENT", "1") == "1"
                _use_ibp = os.environ.get("SNN_BNB_IBP", "0") == "1"
                _ibp_min_depth = int(os.environ.get("SNN_BNB_IBP_MIN_DEPTH", "0"))
                _ibp_every = int(os.environ.get("SNN_BNB_IBP_EVERY", "100"))  # call IBP every K branching levels (empirical sweet spot)
                _ibp_coupled = os.environ.get("SNN_BNB_IBP_COUPLED", "0") == "1"  # budget-coupled knapsack IBP
                _use_beta = os.environ.get("SNN_BNB_BETA", "0") == "1"  # β-branching on BC-IBP
                _beta_depth = int(os.environ.get("SNN_BNB_BETA_DEPTH", "2"))
                _beta_top_k = int(os.environ.get("SNN_BNB_BETA_TOPK", "1"))
                if _use_beta:
                    def _ibp_fn(**kwargs):
                        return ibp_prove_robust_beta(
                            beta_depth=_beta_depth, beta_top_k=_beta_top_k, **kwargs
                        )
                elif _ibp_coupled:
                    _ibp_fn = ibp_prove_robust_budgeted
                else:
                    _ibp_fn = ibp_prove_robust

                target_time_base = base_times[orig_pred]
                min_non_target_base = np.min([base_times[i] for i in range(num_classes) if i != orig_pred])
                t_ref = min(target_time_base, min_non_target_base)

                if _legacy_active:
                    active_priority_base = []
                    for px, py in priority:
                        orig_val = img[px, py]
                        if orig_val - delta + synaptic_delay <= t_ref + delta:
                            active_priority_base.append((px, py))
                elif _efficient:
                    pixel_sensitivity = []
                    for px, py in priority:
                        orig_val = int(img[px, py])
                        max_change = 0.0
                        for dv in [-delta, delta]:
                            nv = max(0, min(num_steps - 1, orig_val + dv))
                            if nv == orig_val:
                                continue
                            test_img = img.copy()
                            test_img[px, py] = nv
                            test_spks: list = []
                            forward(cfg, weights_list, test_img, test_spks)
                            test_last = test_spks[-1]
                            change = np.max(np.abs(test_last - base_times))
                            if change > max_change:
                                max_change = change
                        if max_change > 0:
                            pixel_sensitivity.append(((px, py), max_change))
                    pixel_sensitivity.sort(key=lambda x: x[1], reverse=True)
                    active_priority_base = [p for p, _ in pixel_sensitivity]
                else:
                    active_priority_base = [tuple(p) for p in priority]

                info(
                    f"Filtered pixels: {len(priority)} -> {len(active_priority_base)} "
                    f"(Reduced by {len(priority)-len(active_priority_base)})"
                )

                # ============================================================
                # PSM (Prefix-Set Matching) 설정
                # ============================================================
                _use_psm = os.environ.get("SNN_BNB_PSM", "0") == "1" or getattr(cfg, 'prefix_set_match', False)

                if _use_psm:
                    # 픽셀을 원래 스파이크 시간 순으로 정렬
                    active_priority_base.sort(key=lambda p: int(img[p[0], p[1]]))

                    # Time-boundary 위치 계산: 각 원래 시간 T에서 활성 픽셀이 마지막으로 나타나는 위치
                    time_boundary_pos = {}
                    for idx, (px, py) in enumerate(active_priority_base):
                        t = int(img[px, py])
                        time_boundary_pos[t] = idx + 1  # 다음 픽셀의 위치

                    # Prefix check positions: 시간 T-delta까지의 prefix가 stable해지는 위치
                    prefix_check_positions = set()
                    for T, pos in time_boundary_pos.items():
                        stable_t = T - delta
                        if stable_t >= 0:
                            prefix_check_positions.add(pos)

                    info(f"PSM enabled: {len(prefix_check_positions)} prefix check positions")
                else:
                    time_boundary_pos = {}
                    prefix_check_positions = set()
                    info("PSM disabled")

                # ============================================================
                # V_h[h, t]: hidden neuron h의 시간 t에서의 누적 voltage
                # voltage array shape from forward: (n_hidden, num_steps+1)
                # 마지막 timestep (num_steps-1)은 forced spike (threshold+1)
                V_h_base = base_voltages[0].copy()  # (n_hidden, num_steps+1)
                V_o_base = base_voltages[1].copy()  # (n_output, num_steps+1)

                # ============================================================
                # PSM: Zobrist 해시 테이블 초기화
                # ============================================================
                if _use_psm:
                    zobrist_table = {}
                    rng = Random(42)
                    for p_idx in range(len(active_priority_base)):
                        for t in range(num_steps):
                            zobrist_table[(p_idx, t)] = rng.getrandbits(64)

                    # 초기 해시: 모든 픽셀이 원래 시간에 있을 때의 prefix
                    initial_hash = 0
                    for p_idx, (px, py) in enumerate(active_priority_base):
                        initial_hash ^= zobrist_table[(p_idx, int(img[px, py]))]
                else:
                    zobrist_table = {}
                    initial_hash = 0

                def _compute_spike_times(V, force_idx=None):
                    """Voltage 배열에서 spike time 계산. forward()와 동일한 로직."""
                    # argmax(V > threshold) + 1, clamped to num_steps-1
                    st = np.argmax(V > threshold, axis=1).astype(float) + 1
                    st[st > num_steps - 1] = num_steps - 1
                    return st

                def _compute_state_hash(V_h, V_o, spike_h, spike_o, pixel_pos, rem_neg, rem_pos):
                    """상태를 hash로 변환. state caching용."""
                    # numpy array를 bytes로 변환 후 hash
                    state_bytes = (
                        V_h.tobytes()
                        + V_o.tobytes()
                        + spike_h.tobytes()
                        + spike_o.tobytes()
                        + pixel_pos.to_bytes(4, 'little')
                        + rem_neg.to_bytes(4, 'little')
                        + rem_pos.to_bytes(4, 'little')
                    )
                    # hashlib.sha256보다 빠른 hash 사용
                    return hash(state_bytes)

                # ============================================================
                # Incremental BnB DFS
                # ============================================================
                def bnb_dfs_incremental(
                    pixel_img,  # 현재 각 픽셀의 값 (mutable, backtrack으로 복원)
                    V_h,
                    V_o,  # 현재 voltage state (mutable)
                    spike_h,  # hidden spike times (mutable float array)
                    spike_o,  # output spike times (mutable float array)
                    pixel_pos,
                    rem_neg,
                    rem_pos,
                    max_incr,  # max_incr[h, t]: remaining 픽셀이 V_h[h,t]를 증가시킬 수 있는 relaxed 상한
                    max_decr,  # max_decr[h, t]: remaining 픽셀이 V_h[h,t]를 감소시킬 수 있는 relaxed 상한
                    prefix_hash=0,  # PSM: 현재 prefix의 zobrist hash
                    psm_cache=None,  # PSM: {(prefix_hash, rem_neg, rem_pos)} 캐시
                    state_cache=None,  # State caching: {state_hash: found_adv} 캐시
                ):
                    # Iterative Branch 3 (no perturbation) to bound stack depth.
                    # Recursion via Branches 1/2 is bounded by delta; the no-perturb path
                    # would otherwise blow up to len(active_priority) (3072 for CIFAR).
                    pending_restore = []  # (idx_x, idx_y, orig_val, ppos)
                    try:
                        while True:
                            if found_adversarial[0]:
                                return

                            # ---- Adversarial check ----
                            target_time = spike_o[orig_pred]
                            min_nt_time = float("inf")
                            for i in range(num_classes):
                                if i != orig_pred and spike_o[i] < min_nt_time:
                                    min_nt_time = spike_o[i]

                            is_adversarial = (min_nt_time < target_time) or (
                                min_nt_time == target_time and any(spike_o[i] == target_time for i in range(orig_pred))
                            )

                            if is_adversarial:
                                l1_cost = int(np.sum(np.abs(pixel_img.astype(int) - img.astype(int))))
                                if l1_cost <= delta:
                                    witness_pred = forward(cfg, weights_list, pixel_img)
                                    assert (
                                        witness_pred != orig_pred
                                    ), f"Incremental mismatch: witness_pred={witness_pred}, orig_pred={orig_pred}"
                                    print(f"Adversarial found: pred={witness_pred}, L1_cost={l1_cost}.")
                                    found_adversarial[0] = True
                                return

                            if pixel_pos == len(active_priority):
                                return
                            if rem_neg == 0 and rem_pos == 0:
                                return

                            # ---- State cache lookup ----
                            if state_cache is not None:
                                state_hash = _compute_state_hash(V_h, V_o, spike_h, spike_o, pixel_pos, rem_neg, rem_pos)
                                if state_hash in state_cache:
                                    psm_stats["state_hits"] += 1
                                    return
                                psm_stats["state_checks"] += 1

                            # ---- Pruning: check if all hidden spike times are fixed ----
                            if n_hidden <= 32:
                                all_fixed = True
                                for h in range(n_hidden):
                                    s_h = int(spike_h[h])
                                    si = s_h - 1
                                    if si < 0:
                                        si = 0
                                    if V_h[h, si] - max_decr[h, si] <= threshold:
                                        all_fixed = False
                                        break
                                    can_advance = False
                                    for t in range(si):
                                        if V_h[h, t] + max_incr[h, t] > threshold:
                                            can_advance = True
                                            break
                                    if can_advance:
                                        all_fixed = False
                                        break
                                if all_fixed:
                                    return

                            # ---- IBP bound pruning ----
                            if (
                                _use_ibp
                                and pixel_pos >= _ibp_min_depth
                                and (pixel_pos - _ibp_min_depth) % _ibp_every == 0
                            ):
                                remaining_arr = np.asarray(active_priority[pixel_pos:], dtype=int)
                                if remaining_arr.ndim == 1:
                                    remaining_arr = remaining_arr.reshape(0, 2)
                                psm_stats["ibp_calls"] = psm_stats.get("ibp_calls", 0) + 1
                                if _ibp_fn(
                                    pixel_img=pixel_img,
                                    remaining_pixels=remaining_arr,
                                    rem_neg=rem_neg,
                                    rem_pos=rem_pos,
                                    w1=w1,
                                    w2_flat=w2_flat,
                                    num_steps=num_steps,
                                    threshold=threshold,
                                    orig_pred=orig_pred,
                                    num_classes=num_classes,
                                ):
                                    psm_stats["ibp_prunes"] = psm_stats.get("ibp_prunes", 0) + 1
                                    return

                            # ---- PSM: Cache lookup at prefix check points ----
                            if psm_cache is not None and pixel_pos in prefix_check_positions:
                                cache_key = (prefix_hash, rem_neg, rem_pos)
                                psm_stats["checks"] += 1
                                if cache_key in psm_cache:
                                    psm_stats["hits"] += 1
                                    return

                            # ---- Branching ----
                            idx_x, idx_y = active_priority[pixel_pos]
                            orig_val = int(pixel_img[idx_x, idx_y])
                            max_t = num_steps

                            _update_bounds(idx_x, idx_y, orig_val, max_incr, max_decr, w1, num_steps, -1)
                            pending_restore.append((idx_x, idx_y, orig_val, pixel_pos))

                            # Branch 1: positive perturbation (recurse, bounded by delta)
                            if rem_pos >= 1:
                                for v in range(orig_val + 1, max_t):
                                    cost = v - orig_val
                                    if rem_pos < cost:
                                        break
                                    _apply_pixel_change(
                                        idx_x, idx_y, orig_val, v,
                                        w1, w2_flat, V_h, V_o, spike_h, spike_o,
                                        num_steps, n_hidden, num_classes,
                                    )
                                    pixel_img[idx_x, idx_y] = v

                                    new_hash = prefix_hash
                                    if zobrist_table:
                                        new_hash ^= zobrist_table[(pixel_pos, orig_val)]
                                        new_hash ^= zobrist_table[(pixel_pos, v)]

                                    bnb_dfs_incremental(
                                        pixel_img, V_h, V_o, spike_h, spike_o,
                                        pixel_pos + 1, rem_neg, rem_pos - cost,
                                        max_incr, max_decr, new_hash, psm_cache, state_cache,
                                    )

                                    if found_adversarial[0]:
                                        return

                                    _apply_pixel_change(
                                        idx_x, idx_y, v, orig_val,
                                        w1, w2_flat, V_h, V_o, spike_h, spike_o,
                                        num_steps, n_hidden, num_classes,
                                    )
                                    pixel_img[idx_x, idx_y] = orig_val

                            # Branch 2: negative perturbation (recurse, bounded by delta)
                            if rem_neg >= 1:
                                for v in range(orig_val - 1, -1, -1):
                                    cost = orig_val - v
                                    if rem_neg < cost:
                                        break
                                    _apply_pixel_change(
                                        idx_x, idx_y, orig_val, v,
                                        w1, w2_flat, V_h, V_o, spike_h, spike_o,
                                        num_steps, n_hidden, num_classes,
                                    )
                                    pixel_img[idx_x, idx_y] = v

                                    new_hash = prefix_hash
                                    if zobrist_table:
                                        new_hash ^= zobrist_table[(pixel_pos, orig_val)]
                                        new_hash ^= zobrist_table[(pixel_pos, v)]

                                    bnb_dfs_incremental(
                                        pixel_img, V_h, V_o, spike_h, spike_o,
                                        pixel_pos + 1, rem_neg - cost, rem_pos,
                                        max_incr, max_decr, new_hash, psm_cache, state_cache,
                                    )

                                    if found_adversarial[0]:
                                        return

                                    _apply_pixel_change(
                                        idx_x, idx_y, v, orig_val,
                                        w1, w2_flat, V_h, V_o, spike_h, spike_o,
                                        num_steps, n_hidden, num_classes,
                                    )
                                    pixel_img[idx_x, idx_y] = orig_val

                            # Branch 3: no perturbation -> iterate (no recursion)
                            pixel_pos += 1
                    finally:
                        # Restore bounds + insert cache entries for visited pixels (reverse order).
                        # Branches 1/2 are state-neutral on backtrack, so V_h/V_o/spike_*/pixel_img
                        # are at function-entry state here (unless found_adversarial, in which case
                        # we skip cache inserts but still must restore max_incr/max_decr).
                        for idx_x_r, idx_y_r, orig_val_r, ppos_r in reversed(pending_restore):
                            _update_bounds(idx_x_r, idx_y_r, orig_val_r, max_incr, max_decr, w1, num_steps, +1)
                            if not found_adversarial[0]:
                                if psm_cache is not None and ppos_r in prefix_check_positions:
                                    psm_cache.add((prefix_hash, rem_neg, rem_pos))
                                if state_cache is not None:
                                    state_cache.add(_compute_state_hash(V_h, V_o, spike_h, spike_o, ppos_r, rem_neg, rem_pos))

                # ============================================================
                # Helper: incremental voltage update for single pixel change
                # ============================================================
                def _apply_pixel_change(
                    px, py, old_val, new_val, w1, w2_flat, V_h, V_o, spike_h, spike_o, num_steps, n_hidden, num_classes
                ):
                    if old_val == new_val:
                        return

                    w_h = w1[:, px, py]  # (n_hidden,)

                    t_lo = min(old_val, new_val)
                    t_hi = max(old_val, new_val)
                    if new_val < old_val:
                        V_h[:, t_lo:t_hi] += w_h[:, np.newaxis]
                    else:
                        V_h[:, t_lo:t_hi] -= w_h[:, np.newaxis]

                    # Vectorized hidden spike time 재계산
                    affected = w_h != 0  # (n_hidden,) bool mask
                    old_spike_h = spike_h.copy()

                    # 영향받는 뉴런 중 spike에 변화 가능성 있는 것만 재계산
                    exceeded_all = V_h[:, :num_steps] > threshold  # (n_hidden, num_steps)
                    any_exceeded = np.any(exceeded_all, axis=1)  # (n_hidden,)
                    first_exceeded = np.argmax(exceeded_all, axis=1).astype(float) + 1  # (n_hidden,)
                    first_exceeded = np.minimum(first_exceeded, num_steps - 1)
                    no_exceed_mask = ~any_exceeded
                    first_exceeded[no_exceed_mask] = num_steps - 1

                    # 영향받은 뉴런만 업데이트
                    spike_h[affected] = first_exceeded[affected]

                    # Output voltage 업데이트: spike time이 바뀐 hidden neuron만
                    changed = spike_h != old_spike_h  # (n_hidden,)
                    changed_idx = np.where(changed)[0]
                    for h in changed_idx:
                        s_old = int(old_spike_h[h])
                        s_new = int(spike_h[h])
                        w_o = w2_flat[:, h]  # (n_output,)
                        if s_new < s_old:
                            V_o[:, s_new:s_old] += w_o[:, np.newaxis]
                        else:
                            V_o[:, s_old:s_new] -= w_o[:, np.newaxis]

                    # Output spike time 재계산 (vectorized)
                    if len(changed_idx) > 0:
                        exceeded_o = V_o[:, :num_steps] > threshold  # (n_output, num_steps)
                        any_exc_o = np.any(exceeded_o, axis=1)
                        spike_o[:] = np.where(
                            any_exc_o,
                            np.minimum(np.argmax(exceeded_o, axis=1).astype(float) + 1, num_steps - 1),
                            num_steps - 1,
                        )

                # ============================================================
                # Helper: Voltage margin bound 관리 (vectorized)
                # ============================================================
                def _compute_initial_bounds(active_priority, pixel_img, w1, num_steps, n_hidden):
                    max_incr = np.zeros((n_hidden, num_steps + 1))
                    max_decr = np.zeros((n_hidden, num_steps + 1))
                    for px, py in active_priority:
                        v_p = int(pixel_img[px, py])
                        w_h = w1[:, px, py]  # (n_hidden,)
                        w_pos = np.maximum(w_h, 0)  # (n_hidden,)
                        w_neg = np.maximum(-w_h, 0)  # (n_hidden,) = |negative weights|
                        # wh > 0: incr at t<v_p, decr at t>=v_p
                        # wh < 0: incr at t>=v_p (|wh|), decr at t<v_p (|wh|)
                        if v_p > 0:
                            max_incr[:, :v_p] += w_pos[:, np.newaxis]
                            max_decr[:, :v_p] += w_neg[:, np.newaxis]
                        if v_p < num_steps:
                            max_decr[:, v_p:num_steps] += w_pos[:, np.newaxis]
                            max_incr[:, v_p:num_steps] += w_neg[:, np.newaxis]
                    return max_incr, max_decr

                def _update_bounds(px, py, v_p, max_incr, max_decr, w1, num_steps, sign):
                    """sign=+1: restore (add back), sign=-1: remove (subtract)."""
                    w_h = w1[:, px, py]
                    w_pos = np.maximum(w_h, 0)
                    w_neg = np.maximum(-w_h, 0)
                    if v_p > 0:
                        max_incr[:, :v_p] += sign * w_pos[:, np.newaxis]
                        max_decr[:, :v_p] += sign * w_neg[:, np.newaxis]
                    if v_p < num_steps:
                        max_decr[:, v_p:num_steps] += sign * w_pos[:, np.newaxis]
                        max_incr[:, v_p:num_steps] += sign * w_neg[:, np.newaxis]

                # ============================================================
                # 메인 루프: delta budget split
                # ============================================================
                psm_stats = {"checks": 0, "hits": 0, "state_checks": 0, "state_hits": 0}  # Cache statistics

                for i in range(delta + 1):
                    rem_neg = i
                    rem_pos = delta - i

                    active_priority = active_priority_base

                    # Incremental state 초기화 (원본 이미지 기준)
                    V_h = V_h_base.copy()
                    V_o = V_o_base.copy()
                    spike_h = _compute_spike_times(V_h)
                    spike_o = _compute_spike_times(V_o)
                    pixel_img = img.copy()

                    # Voltage margin bound 초기화
                    max_incr, max_decr = _compute_initial_bounds(active_priority, pixel_img, w1, num_steps, n_hidden)

                    # PSM: Cache 초기화 (모든 budget split에서 공유 가능)
                    psm_cache = set() if _use_psm else None

                    # State cache 초기화
                    state_cache = set()

                    bnb_dfs_incremental(
                        pixel_img,
                        V_h,
                        V_o,
                        spike_h,
                        spike_o,
                        0,
                        rem_neg,
                        rem_pos,
                        max_incr,
                        max_decr,
                        initial_hash,
                        psm_cache,
                        state_cache,
                    )

                    if found_adversarial[0]:
                        break

                # 결과 로깅
                info(f"Checking done in time {time.time() - tx}")
                if _use_psm:
                    hit_rate = (psm_stats["hits"] / psm_stats["checks"] * 100) if psm_stats["checks"] > 0 else 0
                    info(f"PSM cache: {psm_stats['checks']} checks, {psm_stats['hits']} hits ({hit_rate:.1f}%)")
                state_hit_rate = (psm_stats["state_hits"] / psm_stats["state_checks"] * 100) if psm_stats["state_checks"] > 0 else 0
                info(f"State cache: {psm_stats['state_checks']} checks, {psm_stats['state_hits']} hits ({state_hit_rate:.1f}%)")
                ibp_calls = psm_stats.get("ibp_calls", 0)
                ibp_prunes = psm_stats.get("ibp_prunes", 0)
                if ibp_calls:
                    info(f"IBP: {ibp_calls} calls, {ibp_prunes} prunes ({100*ibp_prunes/ibp_calls:.1f}%)")
                if found_adversarial[0]:
                    info(f"Not robust for sample {sample_no} and delta={delta}")
                else:
                    info(f"Robust for sample {sample_no} and delta={delta}.")
                info("")

                return found_adversarial[0]

            # ==================================================================
            # [수정 Step 1] Sound & Complete 한 Exhaustive DFS Oracle 함수
            # ------------------------------------------------------------------
            # 목적:
            #   가지치기(pruning)를 전혀 적용하지 않은 단순 DFS 로, L1 예산 delta
            #   이내의 모든 perturbation 을 탐색한다. 브랜칭은 각 픽셀에 대해
            #   v ∈ [0, num_steps-1] 를 전부 시도하고 cost = |v - orig_val|,
            #   누적 cost 가 delta 를 넘으면 중단.
            #   이 함수는 soundness / completeness 가 자명하므로
            #   검증 결과의 ground truth 오라클로 사용한다.
            #
            # 주의:
            #   - 정답성 확인 전용. 실제 규모 입력에서는 매우 느리다.
            #   - check_sample_direct 와 동일한 시그니처 / 반환값 (bool) 을 사용.
            #   - 별도의 priority / sign / active_set 필터 없음.
            #   - adversarial 발견 시 witness(실제 forward 재분류) 로 재검증.
            # ==================================================================
            def check_sample_exhaustive(
                sample: tuple[int, TImage, int, int, tuple[np.ndarray, np.ndarray]],
                weights_list: TWeightList = weights_list,
            ):
                sample_no, img, label, orig_pred, (priority, sign) = sample
                info("Exhaustive DFS (oracle, no pruning) Query processing")
                tx = time.time()

                num_classes = weights_list[-1].shape[0]
                max_t = cfg.num_steps  # 입력 픽셀 spike time 의 유효 범위는 [0, num_steps-1]
                found_adversarial = [False]
                witness_img_holder: list[TImage] = []

                # 픽셀 순회 순서는 정답성에 영향을 주지 않으므로 priority 그대로 사용.
                pixels = [tuple(p) for p in priority]

                def exhaustive_dfs(current_img, pixel_pos, rem_budget):
                    if found_adversarial[0]:
                        return

                    # [판정 단계]
                    # 매 노드에서 forward 를 돌려 adversarial 여부를 확인.
                    # (이른 발견 시 즉시 종료; 정답성에는 무해)
                    spks: list = []
                    forward(cfg, weights_list, current_img, spks)
                    last = spks[-1]
                    target_time = last[orig_pred]
                    min_non_target_time = np.min([last[i] for i in range(num_classes) if i != orig_pred])

                    is_adversarial = (min_non_target_time < target_time) or (
                        min_non_target_time == target_time and any(last[i] == target_time for i in range(orig_pred))
                    )
                    if is_adversarial:
                        l1_cost = int(np.sum(np.abs(current_img.astype(int) - img.astype(int))))
                        if l1_cost <= delta:
                            found_adversarial[0] = True
                            witness_img_holder.append(current_img.copy())
                            return

                    # [종결 조건]
                    if pixel_pos == len(pixels) or rem_budget == 0:
                        return

                    idx_x, idx_y = pixels[pixel_pos]
                    orig_val = int(current_img[idx_x, idx_y])

                    # [브랜칭]
                    # 이 픽셀에 대해 가능한 모든 값 v ∈ [0, max_t-1] 을 시도한다.
                    # cost = |v - orig_val| ≤ rem_budget 인 경우에만 재귀.
                    for v in range(max_t):
                        cost = abs(v - orig_val)
                        if cost > rem_budget:
                            continue
                        next_img = current_img.copy()
                        next_img[idx_x, idx_y] = v
                        exhaustive_dfs(next_img, pixel_pos + 1, rem_budget - cost)
                        if found_adversarial[0]:
                            return

                exhaustive_dfs(img.copy(), 0, delta)

                info(f"[Exhaustive] done in {time.time() - tx:.3f}s")
                if found_adversarial[0]:
                    info(f"[Exhaustive] Not robust for sample {sample_no} and delta={delta}")
                else:
                    info(f"[Exhaustive] Robust for sample {sample_no} and delta={delta}")
                return found_adversarial[0]

            # Exhaustive oracle 을 전역으로 노출 (multiprocessing / 외부 검증 harness 에서 접근 가능하도록).
            global check_sample_exhaustive_global
            check_sample_exhaustive_global = check_sample_exhaustive

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
