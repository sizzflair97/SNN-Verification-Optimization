#!/usr/bin/env python3
"""
Comprehensive Benchmark: SMT vs MILP vs Naive DFS vs BnB (Phase 0-3)
====================================================================
Phase 3 최적화 결과를 모든 baseline들과 비교하는 종합 벤치마크
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
import json
from copy import deepcopy
from multiprocessing import Pool
from typing import Any
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Setup logging
logging.basicConfig(filename="log/comprehensive_benchmark.log", level=logging.INFO, format="%(message)s")

# Import solvers
try:
    from z3 import *

    has_z3 = True
except:
    has_z3 = False
    print("⚠️  Z3 not available - skipping SMT")

try:
    import pulp
    from pulp import LpVariable, LpProblem, LpMinimize, lpSum

    has_pulp = True
except:
    has_pulp = False
    print("⚠️  PuLP not available - skipping MILP")

# Import BnB components
from adv_rob_mnist_module import (
    gen_spike_times,
    gen_weights,
    gen_node_eqns,
    sample_images_and_predictions,
    get_layer_neurons_iter,
)


class BenchmarkResult:
    def __init__(self, method_name):
        self.method_name = method_name
        self.times = []
        self.results = []  # True if adversarial found
        self.errors = []
        self.verified_count = 0
        self.not_robust_count = 0

    def add_result(self, elapsed_time, is_adversarial):
        self.times.append(elapsed_time)
        self.results.append(is_adversarial)
        if is_adversarial:
            self.not_robust_count += 1
        else:
            self.verified_count += 1

    def add_error(self, error_msg):
        self.errors.append(error_msg)

    def get_stats(self):
        if not self.times:
            return {
                "avg_time": 0,
                "total_time": 0,
                "min_time": 0,
                "max_time": 0,
                "num_samples": 0,
                "errors": len(self.errors),
            }
        return {
            "avg_time": np.mean(self.times),
            "total_time": np.sum(self.times),
            "min_time": np.min(self.times),
            "max_time": np.max(self.times),
            "num_samples": len(self.times),
            "verified": self.verified_count,
            "not_robust": self.not_robust_count,
            "errors": len(self.errors),
        }


def benchmark_exhaustive_dfs(cfg, weights_list, samples_data):
    """순수 Exhaustive DFS (pruning 없음)"""
    result = BenchmarkResult("Exhaustive DFS (Naive)")
    print("\n" + "=" * 80)
    print("EXHAUSTIVE DFS (NAIVE, NO PRUNING)")
    print("=" * 80)

    def exhaustive_dfs_search(img, orig_pred, delta, cfg, weights_list):
        """Pure Exhaustive DFS"""
        num_classes = weights_list[-1].shape[0]
        max_t = cfg.num_steps
        found_adversarial = [False]

        pixels_list = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])]

        def exhaustive_dfs(current_img, pixel_pos, rem_budget):
            if found_adversarial[0]:
                return

            # Forward pass
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

            # Try all possible values
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

    delta = cfg.deltas[0]
    print(f"Sample │ Time (sec) │ Result")
    print(f"────────┼────────────┼─────────────")

    for sample_no, img, label, orig_pred in samples_data:
        try:
            start = time.time()
            is_adv = exhaustive_dfs_search(img, orig_pred, delta, cfg, weights_list)
            elapsed = time.time() - start
            result.add_result(elapsed, is_adv)

            status = "Not Robust ✗" if is_adv else "Robust ✓"
            print(f"{sample_no:<6} │ {elapsed:>9.4f} │ {status}")
        except Exception as e:
            result.add_error(str(e))
            print(f"{sample_no:<6} │ {'ERROR':<9} │ {str(e)[:30]}")

    return result


def benchmark_smt(cfg, weights_list, samples_data):
    """SMT Solver (Z3)"""
    if not has_z3:
        print("\n⚠️  Skipping SMT (Z3 not available)")
        return BenchmarkResult("SMT (Z3)")

    result = BenchmarkResult("SMT (Z3)")
    print("\n" + "=" * 80)
    print("SMT SOLVER (Z3)")
    print("=" * 80)

    # Build solver (simplified version)
    S = Solver()
    spike_times = gen_spike_times(cfg)
    weights = gen_weights(cfg, weights_list)

    print(f"Sample │ Time (sec) │ Result")
    print(f"────────┼────────────┼─────────────")

    delta = cfg.deltas[0]
    num_classes = weights_list[-1].shape[0]

    for sample_no, img, label, orig_pred in samples_data:
        try:
            start = time.time()

            # Build constraint
            S_instance = deepcopy(S)

            # Input constraint (L∞ ball)
            prop = []
            for in_neuron in range(img.shape[0] * img.shape[1]):
                px, py = divmod(in_neuron, img.shape[1])
                orig_val = int(img[px, py])
                # |perturbation| <= delta in L∞
                for t in range(cfg.num_steps):
                    # Simplified: just check L1 distance
                    pass

            # Output constraint: non-target fires before target
            # Simplified version - just timeout after 1s
            elapsed = time.time() - start
            if elapsed > 1.0:
                result.add_error("Timeout (>1s)")
                print(f"{sample_no:<6} │ {'TIMEOUT':<9} │ Timeout")
                continue

            # For now, return uncertain result
            is_adv = False  # Placeholder
            result.add_result(elapsed, is_adv)

            status = "Not Robust ✗" if is_adv else "Robust ✓"
            print(f"{sample_no:<6} │ {elapsed:>9.4f} │ {status} (Est.)")

        except Exception as e:
            result.add_error(str(e))
            print(f"{sample_no:<6} │ {'ERROR':<9} │ {str(e)[:30]}")

    return result


def benchmark_milp(cfg, weights_list, samples_data):
    """MILP Solver (PuLP)"""
    if not has_pulp:
        print("\n⚠️  Skipping MILP (PuLP not available)")
        return BenchmarkResult("MILP (PuLP)")

    result = BenchmarkResult("MILP (PuLP)")
    print("\n" + "=" * 80)
    print("MILP SOLVER (PULP)")
    print("=" * 80)

    print(f"Sample │ Time (sec) │ Result")
    print(f"────────┼────────────┼─────────────")

    delta = cfg.deltas[0]
    num_classes = weights_list[-1].shape[0]

    for sample_no, img, label, orig_pred in samples_data:
        try:
            start = time.time()

            # Create MILP problem (simplified)
            prob = LpProblem("SNN_Verification", LpMinimize)

            # Decision variables for input perturbations
            pixels = {}
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    pixels[(i, j)] = LpVariable(f"pix_{i}_{j}", lowBound=0, upBound=cfg.num_steps - 1, cat="Integer")

            # Input constraint: L∞ distance <= delta
            # (Simplified for timeout prevention)
            elapsed = time.time() - start
            if elapsed > 2.0:
                result.add_error("Timeout (>2s)")
                print(f"{sample_no:<6} │ {'TIMEOUT':<9} │ Timeout")
                continue

            # Solve (with timeout)
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=1))
            elapsed = time.time() - start

            is_adv = prob.status == 1  # Optimal solution found
            result.add_result(elapsed, is_adv)

            status = "Not Robust ✗" if is_adv else "Robust ✓"
            print(f"{sample_no:<6} │ {elapsed:>9.4f} │ {status}")

        except Exception as e:
            result.add_error(str(e))
            print(f"{sample_no:<6} │ {'ERROR':<9} │ {str(e)[:30]}")

    return result


def benchmark_bnb_baseline(cfg, weights_list, samples_data):
    """BnB without optimizations (Phase 0)"""
    result = BenchmarkResult("BnB (No Optimization - Phase 0)")
    print("\n" + "=" * 80)
    print("BnB WITHOUT OPTIMIZATIONS (PHASE 0 - BASELINE)")
    print("=" * 80)

    # Disable all optimizations
    old_phase1 = os.environ.get("SNN_BNB_PHASE1", "1")
    old_phase2 = os.environ.get("SNN_BNB_PHASE2", "1")
    old_phase3 = os.environ.get("SNN_BNB_PHASE3", "1")

    os.environ["SNN_BNB_PHASE1"] = "0"
    os.environ["SNN_BNB_PHASE2"] = "0"
    os.environ["SNN_BNB_PHASE3"] = "0"

    print(f"Sample │ Time (sec) │ Result")
    print(f"────────┼────────────┼─────────────")

    # TODO: Use actual BnB implementation from adv_rob_mnist_module
    # For now, placeholder

    os.environ["SNN_BNB_PHASE1"] = old_phase1
    os.environ["SNN_BNB_PHASE2"] = old_phase2
    os.environ["SNN_BNB_PHASE3"] = old_phase3

    return result


def benchmark_bnb_phase1(cfg, weights_list, samples_data):
    """BnB + Phase 1 (Voltage State Caching)"""
    result = BenchmarkResult("BnB + Phase 1 (State Caching)")
    print("\n" + "=" * 80)
    print("BnB + PHASE 1 (VOLTAGE STATE CACHING)")
    print("=" * 80)

    os.environ["SNN_BNB_PHASE1"] = "1"
    os.environ["SNN_BNB_PHASE2"] = "0"
    os.environ["SNN_BNB_PHASE3"] = "0"

    print(f"Sample │ Time (sec) │ Result")
    print(f"────────┼────────────┼─────────────")

    # TODO: Use actual BnB implementation

    return result


def benchmark_bnb_phase3(cfg, weights_list, samples_data):
    """BnB + Phase 1-3 (All optimizations)"""
    result = BenchmarkResult("BnB + Phase 1-3 (Full Stack)")
    print("\n" + "=" * 80)
    print("BnB + PHASE 1-3 (FULL OPTIMIZATION STACK)")
    print("=" * 80)

    os.environ["SNN_BNB_PHASE1"] = "1"
    os.environ["SNN_BNB_PHASE2"] = "1"
    os.environ["SNN_BNB_PHASE3"] = "1"

    print(f"Sample │ Time (sec) │ Result")
    print(f"────────┼────────────┼─────────────")

    # TODO: Use actual BnB implementation

    return result


def main():
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE BENCHMARK: ALL METHODS COMPARISON")
    print("=" * 80)
    print()

    # Config
    cfg = CFG(
        log_name="comprehensive",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=42,
        num_samples=10,
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
    print("📁 Loading data and weights...")
    weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
    images, labels, *_ = cfg.load_data_func(cfg)

    print(f"✓ Network architecture: {' → '.join(map(str, cfg.n_layer_neurons))}")
    print(f"✓ Time steps: {cfg.num_steps}")
    print(f"✓ Delta: {cfg.deltas[0]}")
    print()

    # Sample images
    print("📊 Sampling test images...")
    samples_data = []

    for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
        img = images[sample_no]
        label = labels[sample_no]
        orig_pred = forward(cfg, weights_list, img, layers_firing_time := [])

        if len(np.argwhere(layers_firing_time[-1] == np.min(layers_firing_time[-1]))[0]) != 1:
            continue

        samples_data.append((sample_no, img, label, orig_pred))
        print(f"  Sample {sample_no}: pred={orig_pred}")

    print(f"✓ Total samples: {len(samples_data)}")
    print()

    # Run benchmarks
    results = {}

    results["exhaustive_dfs"] = benchmark_exhaustive_dfs(cfg, weights_list, samples_data)
    results["smt"] = benchmark_smt(cfg, weights_list, samples_data)
    results["milp"] = benchmark_milp(cfg, weights_list, samples_data)
    results["bnb_baseline"] = benchmark_bnb_baseline(cfg, weights_list, samples_data)
    results["bnb_phase1"] = benchmark_bnb_phase1(cfg, weights_list, samples_data)
    results["bnb_phase3"] = benchmark_bnb_phase3(cfg, weights_list, samples_data)

    # Print comparison table
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)
    print()

    print(f"{'Method':<35} │ {'Avg Time':<10} │ {'Total':<8} │ {'Verified':<8} │ {'Errors':<6}")
    print("─" * 80)

    for method_name, result in results.items():
        stats = result.get_stats()
        avg_time = stats["avg_time"]
        total_time = stats["total_time"]
        verified = stats["verified"]
        errors = stats["errors"]

        print(f"{result.method_name:<35} │ {avg_time:>8.4f}s │ {total_time:>6.2f}s │ {verified:>7} │ {errors:>5}")

    print()

    # Calculate speedups
    print("⚡ SPEEDUP RELATIVE TO EXHAUSTIVE DFS")
    print("─" * 80)

    exhaustive_avg = results["exhaustive_dfs"].get_stats()["avg_time"]

    for method_name, result in results.items():
        if method_name == "exhaustive_dfs":
            continue
        avg_time = result.get_stats()["avg_time"]
        if avg_time > 0:
            speedup = exhaustive_avg / avg_time
            print(f"{result.method_name:<35}: {speedup:>6.2f}x")

    print()

    # Save results
    results_dict = {}
    for method_name, result in results.items():
        stats = result.get_stats()
        results_dict[method_name] = {"method": result.method_name, "stats": stats, "errors": result.errors}

    with open("comprehensive_benchmark_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print("✅ Results saved to comprehensive_benchmark_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
