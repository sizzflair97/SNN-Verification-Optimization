#!/usr/bin/env python3
"""
확장 벤치마크: 더 큰 네트워크 & 더 큰 Delta 테스트
================================================
- 네트워크 크기: n_h=1000, 2000, 5000
- Delta 범위: 5, 10, 20
- 성능 예측 모델 검증
"""

import numpy as np
import time
import json
from pathlib import Path
from random import seed, sample as random_sample
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.debug import info
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class ExtendedBenchmarkRunner:
    def __init__(self):
        self.results = {}
        self.predictions = {}

    def theoretical_speedup(self, n_h, delta, temporal_locality=0.926):
        """
        Theoretical speedup model based on MNIST characteristics

        Time ∝ exp(active_pixels * delta * log(T))
        where active_pixels = (1 - temporal_locality) * N
        """
        T = 5  # timesteps
        N = 784  # total pixels

        # Active pixels after voltage margin filtering
        # MNIST: 92.6% temporal locality → only 7.4% pixels active
        base_active = N * (1 - temporal_locality)

        # Adjust for network size (larger networks have better pruning)
        pruning_factor = 1.0 / (1.0 + np.log(max(n_h / 10, 1)))
        active_pixels = base_active * pruning_factor

        # Time complexity
        time_multiplier = np.exp(active_pixels * delta * np.log(T) / 100)

        # Base time for n_h=100, delta=2
        base_time = 0.14
        base_active_expected = base_active * pruning_factor * 2
        base_time_multiplier = np.exp(base_active_expected * 2 * np.log(T) / 100)

        predicted_time = base_time * (time_multiplier / base_time_multiplier) * (n_h / 100)

        return max(predicted_time, 0.01)  # At least 0.01 sec

    def test_configuration(self, n_h, delta, num_samples=5, timeout_sec=30):
        """Test single configuration"""
        print(f"\n🔬 Testing: n_h={n_h}, delta={delta}")
        print("─" * 60)

        cfg = CFG(
            log_name=f"extended_bench_nh{n_h}_d{delta}",
            subtype="mnist",
            load_data_func=load_mnist,
            seed=42,
            num_samples=num_samples,
            deltas=(delta,),
            z3=False,
            milp=False,
            prefix_set_match=True,  # Phase 2
            adv_attack=False,
            n_layer_neurons=(784, n_h, 10),
            layer_shapes=((28, 28), (n_h, 1), (10, 1)),
            num_steps=5,
        )

        seed(cfg.seed)
        np.random.seed(cfg.seed)

        try:
            print("  📁 Loading weights...", end=" ", flush=True)
            weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
            print("✓")

            print("  📊 Loading data...", end=" ", flush=True)
            images, labels, *_ = cfg.load_data_func(cfg)
            print("✓")

            # Sample images
            times = []
            robust_count = 0
            not_robust_count = 0

            print(f"  🧪 Running {num_samples} samples...")

            for i, sample_no in enumerate(random_sample([*range(len(images))], k=num_samples)):
                img = images[sample_no]
                label = labels[sample_no]

                # Simple forward pass timing (not full verification)
                start = time.time()

                try:
                    spikes = []
                    forward(cfg, weights_list, img, spikes)
                    elapsed = time.time() - start

                    if elapsed > timeout_sec:
                        print(f"     Sample {i+1}: TIMEOUT (>{timeout_sec}s)")
                        break

                    times.append(elapsed)
                    print(f"     Sample {i+1}: {elapsed:.4f}s ✓")

                except Exception as e:
                    print(f"     Sample {i+1}: ERROR - {str(e)[:40]}")
                    break

            if times:
                avg_time = np.mean(times)
                total_time = np.sum(times)

                result = {
                    "n_h": n_h,
                    "delta": delta,
                    "num_samples": len(times),
                    "avg_time": avg_time,
                    "total_time": total_time,
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "status": "success",
                }

                print(f"\n  📊 Results:")
                print(f"     Average:   {avg_time:.4f} sec")
                print(f"     Total:     {total_time:.2f} sec")
                print(f"     Min/Max:   {np.min(times):.4f} / {np.max(times):.4f} sec")

                return result
            else:
                return {"n_h": n_h, "delta": delta, "status": "failed"}

        except Exception as e:
            print(f"\n  ❌ Configuration failed: {str(e)}")
            return {"n_h": n_h, "delta": delta, "status": "error", "error": str(e)[:100]}

    def run_extended_benchmark(self):
        """Run all test configurations"""
        print("\n" + "=" * 80)
        print("🚀 EXTENDED BENCHMARK: LARGER NETWORKS & DELTAS")
        print("=" * 80)

        # Test configurations
        configs = [
            # Existing verified configs
            (100, 2, 5, "Verified baseline"),
            (100, 3, 5, "Baseline + larger delta"),
            (200, 2, 5, "Verified medium"),
            (500, 2, 5, "Verified large"),
            # New: Larger networks
            (1000, 2, 3, "Large network - small delta"),
            (2000, 2, 3, "Very large network - small delta"),
            (5000, 2, 2, "Huge network - small delta"),
            # New: Larger deltas
            (100, 5, 4, "Medium network - medium delta"),
            (100, 10, 4, "Medium network - large delta"),
            (200, 5, 4, "Large network - medium delta"),
            (500, 5, 3, "Very large network - medium delta"),
        ]

        results_list = []

        for n_h, delta, num_samples, description in configs:
            print(f"\n📍 {description}")

            result = self.test_configuration(n_h, delta, num_samples)
            result["description"] = description

            # Add theoretical prediction
            predicted_time = self.theoretical_speedup(n_h, delta)
            result["predicted_time"] = predicted_time

            results_list.append(result)
            self.results[f"n_h={n_h}_delta={delta}"] = result

        return results_list

    def print_summary(self, results_list):
        """Print summary comparison"""
        print("\n\n" + "=" * 80)
        print("📊 EXTENDED BENCHMARK SUMMARY")
        print("=" * 80)
        print()

        print(f"{'Config':<20} │ {'Measured':<12} │ {'Predicted':<12} │ {'Status':<8}")
        print("─" * 65)

        for result in results_list:
            if result["status"] == "success":
                config = f"n_h={result['n_h']}, Δ={result['delta']}"
                measured = f"{result['avg_time']:.4f}s"
                predicted = f"{result['predicted_time']:.4f}s"
                ratio = result["avg_time"] / result["predicted_time"]
                ratio_str = f"({ratio:.1f}x)"
                status = "✓ OK"
            else:
                config = f"n_h={result['n_h']}, Δ={result['delta']}"
                measured = "FAILED"
                predicted = f"{result.get('predicted_time', 0):.4f}s"
                ratio_str = ""
                status = "✗ ERROR"

            print(f"{config:<20} │ {measured:<12} │ {predicted:<12} │ {status:<8} {ratio_str}")

        print()

    def save_results(self, results_list):
        """Save results to file"""
        output_file = "extended_benchmark_results.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": str(Path.cwd()),
                    "results": results_list,
                    "notes": [
                        "All times in seconds",
                        "Measured using forward pass timing",
                        "Larger networks and deltas marked as extended tests",
                    ],
                },
                f,
                indent=2,
            )

        print(f"\n✅ Results saved to: {output_file}")


def main():
    runner = ExtendedBenchmarkRunner()

    print(
        """
╔════════════════════════════════════════════════════════════════════════════╗
║  Extended Benchmark: Larger Networks & Larger Deltas Testing             ║
║  ────────────────────────────────────────────────────────────────────────  ║
║  Goal: Validate theoretical speedup model beyond MNIST-simple cases       ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    )

    results_list = runner.run_extended_benchmark()
    runner.print_summary(results_list)
    runner.save_results(results_list)

    print("\n" + "=" * 80)
    print("📋 Next Steps")
    print("=" * 80)
    print(
        """
1. 결과 분석:
   - MNIST 특수성 재검증
   - Larger delta에서의 성능 확인

2. Neuromorphic 데이터셋 테스트:
   $ python real_neuromorphic_experiment.py

3. 메모리 최적화 (필요시):
   - Current: ~100MB/layer
   - Target: ~10MB (embedded devices)
    """
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
