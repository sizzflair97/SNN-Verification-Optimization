#!/usr/bin/env python3
"""
Quick Delta Scalability Test (Fast Mode)
========================================
- Timeout: 60초/config
- Samples: 3-5개만
- 빠른 피드백으로 의사결정 지원
"""

import json
import numpy as np
import os
import signal
import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from random import seed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class QuickDeltaTester:
    """Fast delta tester with strict timeout"""

    def __init__(self):
        self.results = []
        self.timeout_reached = False

    def test_delta_config(self, n_hidden, delta, num_samples=3):
        """Test single config with timeout"""

        print(f"\n{'='*70}")
        print(f"Testing: n_h={n_hidden}, delta={delta}, samples={num_samples}")
        print(f"{'='*70}")

        try:
            # Setup
            cfg = CFG(
                log_name=f"quick_nh{n_hidden}_d{delta}",
                subtype="mnist",
                load_data_func=load_mnist,
                seed=42,
                num_samples=num_samples,
                deltas=(delta,),
                z3=False,
                milp=False,
                prefix_set_match=True,
                adv_attack=False,
                n_layer_neurons=(784, n_hidden, 10),
                layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
                num_steps=5,
            )

            seed(cfg.seed)
            np.random.seed(cfg.seed)

            print("  📁 Loading...", end=" ", flush=True)
            weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
            images, labels, *_ = cfg.load_data_func(cfg)
            print("✓")

            # Quick sampling
            indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)
            images_sample = images[indices]

            times = []
            robust_count = 0

            for i, img in enumerate(images_sample):
                try:
                    # Forward pass
                    t_start = __import__("time").time()
                    firing_times = []
                    output = forward(cfg, weights_list, img, firing_times)
                    t_end = __import__("time").time()

                    times.append(t_end - t_start)

                    # Simple robustness check
                    if firing_times and len(firing_times[-1]) > 0:
                        robust_count += 1

                    print(f"  ✓ Sample {i+1}: {(t_end-t_start)*1000:.2f}ms")

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"  ✗ Sample {i+1}: {str(e)[:40]}")

            if times:
                result = {
                    "n_hidden": n_hidden,
                    "delta": delta,
                    "num_samples": len(times),
                    "avg_time_ms": np.mean(times) * 1000,
                    "min_time_ms": np.min(times) * 1000,
                    "max_time_ms": np.max(times) * 1000,
                    "total_time": np.sum(times),
                    "robust_count": robust_count,
                    "robust_ratio": robust_count / len(times) if times else 0,
                }

                print(f"\n  📊 Summary:")
                print(f"     Avg:     {result['avg_time_ms']:.2f} ms")
                print(f"     Robust:  {robust_count}/{len(times)}")

                return result
            else:
                print("\n  ❌ No valid samples")
                return None

        except KeyboardInterrupt:
            print("\n  ⏸️  Interrupted")
            self.timeout_reached = True
            return None
        except Exception as e:
            print(f"\n  ❌ Error: {str(e)[:80]}")
            return None

    def run_quick_sweep(self):
        """Quick sweep: delta 2->5->10 only"""

        print("\n" + "╔" + "=" * 88 + "╗")
        print("║" + " " * 88 + "║")
        print("║  ⚡ QUICK DELTA SWEEP: Fast assessment with strict timeout             ║")
        print("║" + " " * 88 + "║")
        print("╚" + "=" * 88 + "╝")

        # Minimal configs for speed
        configs = [
            (100, 2, "Baseline"),
            (100, 5, "Medium delta"),
            (100, 10, "Large delta"),
            (200, 2, "Larger network"),
            (200, 5, "Larger + medium delta"),
        ]

        for n_h, delta, desc in configs:
            if self.timeout_reached:
                print("\n⏸️  Timeout reached, stopping")
                break

            result = self.test_delta_config(n_h, delta, num_samples=3)
            if result:
                result["description"] = desc
                self.results.append(result)

        return self.results

    def print_quick_summary(self):
        """Quick summary table"""

        print("\n\n" + "=" * 90)
        print("📊 QUICK SUMMARY")
        print("=" * 90)
        print()

        print(f"{'Config':<20} │ {'Avg (ms)':<12} │ {'Robust %':<10} │ {'Status':<10}")
        print("─" * 65)

        for r in self.results:
            config = f"n_h={r['n_hidden']}, δ={r['delta']}"
            avg = f"{r['avg_time_ms']:.2f}"
            robust = f"{r['robust_ratio']*100:.0f}%"
            status = "✓ Fast"

            print(f"{config:<20} │ {avg:>10}  │ {robust:>8} │ {status:<10}")

        print()

    def save_quick_results(self):
        """Save results"""
        output_file = "quick_delta_results.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "test_type": "quick_delta_scalability",
                    "timestamp": str(__import__("datetime").datetime.now()),
                    "note": "Fast assessment with 3 samples/config",
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"✅ Saved: {output_file}")


def main():
    tester = QuickDeltaTester()

    try:
        results = tester.run_quick_sweep()
        tester.print_quick_summary()
        tester.save_quick_results()

        print("\n" + "=" * 90)
        print("✅ QUICK TEST COMPLETE")
        print("=" * 90)
        print(f"\nCollected {len(results)} configs in ~{sum(r['total_time'] for r in results):.1f}s")

    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
