#!/usr/bin/env python3
"""
실제 Delta 확장성 테스트 (Real Verification Time)
==============================================
실제 BnB 검증 시간을 측정:
- benchmark_exhaustive.py 기반
- Exhaustive DFS vs 우리 BnB 비교
"""

import numpy as np
import time
import json
from random import seed, sample as random_sample
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.debug import info
import os
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.WARNING)


class RealDeltaScalabilityTester:
    def __init__(self):
        self.results = []

    def exhaustive_dfs_search(self, img, orig_pred, delta, cfg, weights_list):
        """순수 Exhaustive DFS - 실제 탐색 시간 측정"""
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

    def test_delta_configuration(self, n_hidden, delta, num_samples=10):
        """Test single delta with real verification"""

        print(f"\n{'='*70}")
        print(f"🧪 Testing: n_h={n_hidden}, delta={delta}")
        print(f"{'='*70}")

        cfg = CFG(
            log_name=f"delta_nh{n_hidden}_d{delta}",
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
            num_steps=5,
        )

        seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Load model and data
        print("  📁 Loading...", end=" ", flush=True)
        weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
        images, labels, *_ = cfg.load_data_func(cfg)
        print("✓")

        # Sample images
        print("  📊 Sampling...", end=" ", flush=True)
        samples_no_list = []
        sampled_imgs = []
        orig_preds = []

        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            img = images[sample_no]
            orig_pred = forward(cfg, weights_list, img, layers_firing_time := [])

            if len(np.argwhere(layers_firing_time[-1] == np.min(layers_firing_time[-1]))[0]) != 1:
                continue

            samples_no_list.append(sample_no)
            sampled_imgs.append(img)
            orig_preds.append(orig_pred)

        print(f"✓ ({len(samples_no_list)} samples)")

        # Run verification
        times = []
        robust_count = 0
        not_robust_count = 0

        print(f"  🚀 Verification (Exhaustive DFS):")
        print(f"     Sample │ Time (sec) │ Status")
        print(f"     ────────┼────────────┼──────────")

        for i, (sample_no, img, orig_pred) in enumerate(zip(samples_no_list, sampled_imgs, orig_preds)):
            start = time.time()
            is_adv = self.exhaustive_dfs_search(img, orig_pred, delta, cfg, weights_list)
            elapsed = time.time() - start

            times.append(elapsed)

            if is_adv:
                not_robust_count += 1
                status = "Not Robust ✗"
            else:
                robust_count += 1
                status = "Robust ✓"

            print(f"     {sample_no:<6} │ {elapsed:>9.4f} │ {status}")

            # Early exit if too slow
            if elapsed > 10:
                print(f"     ⚠️  Timeout threshold approaching")

        # Calculate statistics
        avg_time = np.mean(times) if times else 0
        total_time = np.sum(times) if times else 0
        min_time = np.min(times) if times else 0
        max_time = np.max(times) if times else 0

        result = {
            "n_hidden": n_hidden,
            "delta": delta,
            "num_samples": len(samples_no_list),
            "avg_time": avg_time,
            "total_time": total_time,
            "min_time": min_time,
            "max_time": max_time,
            "robust_count": robust_count,
            "not_robust_count": not_robust_count,
            "robust_ratio": robust_count / len(samples_no_list) if samples_no_list else 0,
        }

        # Print summary
        print(f"\n  📈 Summary:")
        print(f"     Average:    {avg_time:.4f} sec")
        print(f"     Range:      {min_time:.4f} - {max_time:.4f} sec")
        print(f"     Robust:     {robust_count}/{len(samples_no_list)}")

        return result

    def run_test_suite(self):
        """Run curated test suite"""

        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║  🚀 REAL DELTA SCALABILITY TEST: Verification Time vs Perturbation Size     ║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")

        # Conservative test suite (avoid timeouts)
        test_configs = [
            # n_h=100 (smallest, fastest)
            (100, 2, "Baseline"),
            (100, 3, "Small delta"),
            (100, 5, "Medium delta"),
            (100, 10, "Large delta"),
            # n_h=200 (medium)
            (200, 2, "Medium network"),
            (200, 3, "Medium + small delta"),
            (200, 5, "Medium + medium delta"),
            # n_h=500 (largest, slowest)
            (500, 2, "Large network"),
            (500, 3, "Large + small delta"),
        ]

        for n_h, delta, desc in test_configs:
            try:
                result = self.test_delta_configuration(n_h, delta, num_samples=8)
                result["description"] = desc
                self.results.append(result)

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)[:80]}")
                continue

        return self.results

    def print_comprehensive_report(self):
        """Print comprehensive analysis"""

        print("\n\n" + "=" * 90)
        print("📊 DELTA SCALABILITY COMPREHENSIVE REPORT")
        print("=" * 90)
        print()

        # Group by network size
        by_network = {}
        for r in self.results:
            n_h = r["n_hidden"]
            if n_h not in by_network:
                by_network[n_h] = []
            by_network[n_h].append(r)

        for n_h in sorted(by_network.keys()):
            results = sorted(by_network[n_h], key=lambda x: x["delta"])

            print(f"\n📌 Network: n_h={n_h}")
            print("─" * 90)
            print(f"{'Delta':<6} │ {'Avg (sec)':<12} │ {'Min-Max':<15} │ {'Robust %':<10} │ {'vs δ=2':<10}")
            print("─" * 90)

            baseline_time = results[0]["avg_time"]

            for r in results:
                delta = r["delta"]
                avg = r["avg_time"]
                min_t = r["min_time"]
                max_t = r["max_time"]
                robust_pct = r["robust_ratio"] * 100
                speedup = baseline_time / avg if avg > 0 else 0

                range_str = f"{min_t:.4f}-{max_t:.4f}"
                speedup_str = f"{speedup:.1f}x"

                print(f"{delta:<6} │ {avg:>10.4f}  │ {range_str:<15} │ {robust_pct:>8.0f}% │ {speedup_str:>10}")

        print()

    def print_delta_impact_analysis(self):
        """Analyze delta impact"""

        print("\n" + "=" * 90)
        print("⚡ DELTA IMPACT ANALYSIS: How Robustness Changes with Perturbation Size")
        print("=" * 90)
        print()

        by_network = {}
        for r in self.results:
            n_h = r["n_hidden"]
            if n_h not in by_network:
                by_network[n_h] = []
            by_network[n_h].append(r)

        for n_h in sorted(by_network.keys()):
            results = sorted(by_network[n_h], key=lambda x: x["delta"])

            print(f"\n🔍 n_h={n_h}")
            print("─" * 90)

            if len(results) < 2:
                continue

            baseline_time = results[0]["avg_time"]
            baseline_robust = results[0]["robust_ratio"] * 100

            print(f"Baseline (delta=2): {baseline_time*1000:.1f}ms, {baseline_robust:.0f}% robust")
            print()
            print(f"{'Delta':<6} │ {'Time (ms)':<12} │ {'Growth vs Δ=2':<15} │ {'Robust %':<12} │ {'Change':<12}")
            print("─" * 90)

            for i, r in enumerate(results):
                delta = r["delta"]
                time_ms = r["avg_time"] * 1000
                robust_pct = r["robust_ratio"] * 100

                if i == 0:
                    growth = "—"
                    change = "—"
                else:
                    prev_time = results[i - 1]["avg_time"]
                    growth = f"{(time_ms / (baseline_time*1000) - 1) * 100:+.0f}%"
                    change = f"{robust_pct - baseline_robust:+.0f}pp"

                print(f"{delta:<6} │ {time_ms:>10.1f}  │ {growth:>15} │ {robust_pct:>10.0f}% │ {change:>12}")

        print()

    def save_results(self):
        """Save to JSON"""
        output_file = "real_delta_scalability_results.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "test_type": "real_delta_scalability",
                    "timestamp": str(__import__("datetime").datetime.now()),
                    "note": "Measured actual Exhaustive DFS verification time",
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\n✅ Results saved to: {output_file}")


def main():
    tester = RealDeltaScalabilityTester()

    # Run test suite
    tester.run_test_suite()

    # Print analysis
    tester.print_comprehensive_report()
    tester.print_delta_impact_analysis()

    # Save results
    tester.save_results()

    print("\n" + "=" * 90)
    print("🏆 DELTA SCALABILITY TEST COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
