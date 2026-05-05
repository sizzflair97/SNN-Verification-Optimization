#!/usr/bin/env python3
"""
Exhaustive DFS Benchmark vs BnB/MILP
====================================
"""

import numpy as np
import time
import logging
from random import seed, sample as random_sample
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, backward, prepare_weights
from utils.debug import info
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Setup logging
logging.basicConfig(filename="log/exhaustive_benchmark.log", level=logging.INFO, format="%(message)s")


def benchmark_exhaustive(num_samples=5):
    """벤치마크: Exhaustive DFS"""

    print("=" * 80)
    print("EXHAUSTIVE DFS BENCHMARK")
    print("=" * 80)
    print()

    # Config
    cfg = CFG(
        log_name="exhaustive_bench",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=42,
        num_samples=num_samples,
        deltas=(2,),
        z3=False,
        milp=False,
        prefix_set_match=False,
        adv_attack=False,
        n_layer_neurons=(784, 10, 10),
        layer_shapes=((28, 28), (10, 1), (10, 1)),
        num_steps=5,
    )

    seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load weights and data
    weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
    images, labels, *_ = cfg.load_data_func(cfg)

    print(f"Network: 784 → 10 → 10")
    print(f"num_steps: {cfg.num_steps}")
    print(f"delta: 2")
    print()

    # Sample images
    samples_no_list = []
    sampled_imgs = []
    sampled_labels = []
    orig_preds = []

    for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
        img = images[sample_no]
        label = labels[sample_no]
        orig_pred = forward(cfg, weights_list, img, layers_firing_time := [])

        if len(np.argwhere(layers_firing_time[-1] == np.min(layers_firing_time[-1]))[0]) != 1:
            continue

        samples_no_list.append(sample_no)
        orig_preds.append(orig_pred)
        sampled_imgs.append(img)
        sampled_labels.append(label)

        print(f"Sample {sample_no}: orig_pred={orig_pred}")

    print()
    print("=" * 80)
    print("RUNNING EXHAUSTIVE DFS")
    print("=" * 80)
    print()

    # Exhaustive DFS function
    def exhaustive_dfs_search(img, orig_pred, delta, cfg, weights_list):
        """Pure Exhaustive DFS (no pruning)"""
        num_classes = weights_list[-1].shape[0]
        max_t = cfg.num_steps
        found_adversarial = [False]

        pixels_list = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])]

        def exhaustive_dfs(current_img, pixel_pos, rem_budget):
            if found_adversarial[0]:
                return

            # Forward pass to check adversarial
            spks = []
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
                    return

            # Termination
            if pixel_pos == len(pixels_list) or rem_budget == 0:
                return

            idx_x, idx_y = pixels_list[pixel_pos]
            orig_val = int(current_img[idx_x, idx_y])

            # Branching: try all possible values
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
        return found_adversarial[0]

    # Run exhaustive
    times = []
    delta = 2

    print("Sample │ Time (sec) │ Result")
    print("────────┼────────────┼─────────────")

    for i, (sample_no, img, orig_pred) in enumerate(zip(samples_no_list, sampled_imgs, orig_preds)):
        start = time.time()
        is_adv = exhaustive_dfs_search(img, orig_pred, delta, cfg, weights_list)
        elapsed = time.time() - start
        times.append(elapsed)

        result = "Not Robust" if is_adv else "Robust"
        print(f"{sample_no:<6} │ {elapsed:>9.2f} │ {result}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"Average time:   {avg_time:.2f}s")
    print(f"Min time:       {min_time:.2f}s")
    print(f"Max time:       {max_time:.2f}s")
    print(f"Samples tested: {len(times)}")
    print()

    return avg_time


if __name__ == "__main__":
    try:
        avg_time = benchmark_exhaustive(num_samples=5)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
