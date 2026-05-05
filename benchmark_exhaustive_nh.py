#!/usr/bin/env python3
"""
Exhaustive DFS Benchmark for n_h=100, 200, 500
비교: BnB vs Exhaustive
"""

import numpy as np
import time
import sys
from random import seed, sample as random_sample
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, backward, prepare_weights


def exhaustive_dfs_benchmark(n_hidden, num_samples=10, delta=2, num_steps=5):
    """Exhaustive DFS 벤치마크"""

    print("=" * 80)
    print(f"EXHAUSTIVE DFS BENCHMARK: n_h={n_hidden}, delta={delta}")
    print("=" * 80)
    print()

    # Config
    cfg = CFG(
        log_name=f"exhaustive_nh{n_hidden}",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=42,
        num_samples=num_samples,
        deltas=(delta,),
        z3=False,
        milp=False,
        prefix_set_match=False,
        adv_attack=False,
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=num_steps,
    )

    seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load model and data
    weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
    images, labels, *_ = cfg.load_data_func(cfg)

    # Sample selection
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

    print(f"Network: 784 → {n_hidden} → 10")
    print(f"num_steps: {num_steps}, delta: {delta}")
    print(f"Samples: {len(samples_no_list)}")
    print()

    # Exhaustive DFS function
    def exhaustive_search(img, orig_pred):
        """Pure exhaustive DFS (no pruning)"""
        num_classes = weights_list[-1].shape[0]
        max_t = num_steps
        found_adversarial = [False]

        pixels_list = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])]

        def dfs(current_img, pixel_pos, rem_budget):
            if found_adversarial[0]:
                return

            # Forward pass
            spks = []
            forward(cfg, weights_list, current_img, spks)
            last = spks[-1]
            target_time = last[orig_pred]
            min_non_target = np.min([last[i] for i in range(num_classes) if i != orig_pred])

            # Check adversarial
            is_adv = (min_non_target < target_time) or (
                min_non_target == target_time and any(last[i] == target_time for i in range(orig_pred))
            )

            if is_adv:
                l1_cost = int(np.sum(np.abs(current_img.astype(int) - img.astype(int))))
                if l1_cost <= delta:
                    found_adversarial[0] = True
                    return

            # Termination
            if pixel_pos == len(pixels_list) or rem_budget == 0:
                return

            # Branching
            idx_x, idx_y = pixels_list[pixel_pos]
            orig_val = int(current_img[idx_x, idx_y])

            for v in range(max_t):
                cost = abs(v - orig_val)
                if cost > rem_budget:
                    continue
                next_img = current_img.copy()
                next_img[idx_x, idx_y] = v
                dfs(next_img, pixel_pos + 1, rem_budget - cost)
                if found_adversarial[0]:
                    return

        dfs(img.copy(), 0, delta)
        return found_adversarial[0]

    # Benchmark
    times_robust = []
    times_not_robust = []

    print("Sample │ Time (sec) │ Result | Type")
    print("────────┼────────────┼────────┼──────────────")

    for sample_no, img, orig_pred in zip(samples_no_list, sampled_imgs, orig_preds):
        start = time.time()
        is_adv = exhaustive_search(img, orig_pred)
        elapsed = time.time() - start

        result = "Not Robust" if is_adv else "Robust"
        case_type = "adversarial" if is_adv else "robust"

        if is_adv:
            times_not_robust.append(elapsed)
        else:
            times_robust.append(elapsed)

        print(f"{sample_no:<6} │ {elapsed:>9.3f} │ {result:<6} │ {case_type}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if times_robust:
        print(f"Robust (adversarial NOT found):")
        print(f"  Samples: {len(times_robust)}")
        print(f"  Average: {np.mean(times_robust):.3f}s")
        print(f"  Min:     {np.min(times_robust):.3f}s")
        print(f"  Max:     {np.max(times_robust):.3f}s")
        print()

    if times_not_robust:
        print(f"Not Robust (adversarial found):")
        print(f"  Samples: {len(times_not_robust)}")
        print(f"  Average: {np.mean(times_not_robust):.3f}s")
        print(f"  Min:     {np.min(times_not_robust):.3f}s")
        print(f"  Max:     {np.max(times_not_robust):.3f}s")
        print()

    overall_avg = np.mean(times_robust + times_not_robust)
    print(f"Overall Average: {overall_avg:.3f}s")
    print()

    return {
        "n_hidden": n_hidden,
        "robust_avg": np.mean(times_robust) if times_robust else None,
        "not_robust_avg": np.mean(times_not_robust) if times_not_robust else None,
        "overall_avg": overall_avg,
        "num_robust": len(times_robust),
        "num_not_robust": len(times_not_robust),
    }


if __name__ == "__main__":
    results = []

    # Test n_h=100
    print("\n" + "🔄 " * 20)
    results.append(exhaustive_dfs_benchmark(n_hidden=100, num_samples=10, delta=2))

    # Test n_h=200
    print("\n" + "🔄 " * 20)
    results.append(exhaustive_dfs_benchmark(n_hidden=200, num_samples=10, delta=2))

    # Test n_h=500
    print("\n" + "🔄 " * 20)
    results.append(exhaustive_dfs_benchmark(n_hidden=500, num_samples=10, delta=2))

    # Summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: Exhaustive DFS")
    print("=" * 80)
    print()

    print("n_h   │ Robust Avg │ Not Robust Avg │ Overall Avg")
    print("──────┼────────────┼────────────────┼────────────")
    for r in results:
        robust_str = f"{r['robust_avg']:.3f}s" if r["robust_avg"] else "N/A"
        not_robust_str = f"{r['not_robust_avg']:.3f}s" if r["not_robust_avg"] else "N/A"
        print(f"{r['n_hidden']:<4} │ {robust_str:>10} │ {not_robust_str:>14} │ {r['overall_avg']:.3f}s")
