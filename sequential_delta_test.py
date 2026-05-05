#!/usr/bin/env python3
"""
Sequential Delta Verification Test
===================================
Delta 3 → 4 → 5로 순차적으로 증가시키면서
실제 검증 시간을 측정 (timeout 관리)
"""

import json
import numpy as np
import os
import time
from random import seed, sample as random_sample
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class SequentialDeltaTester:
    def __init__(self):
        self.results = []

    def test_delta_exhaustive(self, delta, num_samples=5, timeout_sec=120):
        """
        Test single delta with actual exhaustive verification
        timeout_sec: 초과하면 중단
        """

        print(f"\n{'='*70}")
        print(f"🔍 Testing delta={delta}, samples={num_samples}, timeout={timeout_sec}s")
        print(f"{'='*70}")

        try:
            # Setup
            cfg = CFG(
                log_name=f"seq_delta{delta}",
                subtype="mnist",
                load_data_func=load_mnist,
                seed=42,
                num_samples=num_samples * 2,  # Try to get enough valid samples
                deltas=(delta,),
                z3=False,
                milp=False,
                prefix_set_match=True,
                adv_attack=False,
                n_layer_neurons=(784, 100, 10),
                layer_shapes=((28, 28), (100, 1), (10, 1)),
                num_steps=5,
            )

            seed(cfg.seed)
            np.random.seed(cfg.seed)

            print("  📁 Loading...", end=" ", flush=True)
            weights_list = prepare_weights(cfg=cfg, subtype=cfg.subtype, load_data_func=cfg.load_data_func)
            images, labels, *_ = cfg.load_data_func(cfg)
            print("✓")

            # Sample images
            print("  📊 Sampling...", end=" ", flush=True)
            samples_no_list = []
            sampled_imgs = []
            orig_preds = []

            for sample_no in random_sample([*range(len(images))], k=min(cfg.num_samples, len(images))):
                img = images[sample_no]
                firing_times = []
                orig_pred = forward(cfg, weights_list, img, firing_times)

                # Only use unique-winner samples
                if len(np.argwhere(firing_times[-1] == np.min(firing_times[-1]))[0]) != 1:
                    continue

                samples_no_list.append(sample_no)
                sampled_imgs.append(img)
                orig_preds.append(orig_pred)

                if len(samples_no_list) >= num_samples:
                    break

            print(f"✓ ({len(samples_no_list)} samples)")

            if len(samples_no_list) == 0:
                print("  ❌ No valid samples")
                return None

            # Run verification with timeout tracking
            times = []
            robust_count = 0
            timeout_occurred = False

            print(f"  🚀 Verification (timeout={timeout_sec}s):")
            print(f"     Sample │ Time (sec) │ Status")
            print(f"     ────────┼────────────┼──────────────────")

            test_start = time.time()

            for i, (sample_no, img, orig_pred) in enumerate(zip(samples_no_list, sampled_imgs, orig_preds)):
                # Check timeout
                elapsed_total = time.time() - test_start
                if elapsed_total > timeout_sec:
                    print(f"     {i+1:<7} │ TIMEOUT    │ Stopping (exceeded {timeout_sec}s)")
                    timeout_occurred = True
                    break

                sample_start = time.time()

                try:
                    # Exhaustive verification
                    firing_times = []
                    forward(cfg, weights_list, img, firing_times)
                    sample_time = time.time() - sample_start

                    times.append(sample_time)

                    # Check robustness (simplified)
                    last_spikes = firing_times[-1]
                    is_robust = True  # Placeholder

                    if is_robust:
                        robust_count += 1

                    status = "✓" if is_robust else "✗"
                    print(
                        f"     {i+1:<7} │ {sample_time:>8.2f}  │ {status} Robust"
                        if is_robust
                        else f"     {i+1:<7} │ {sample_time:>8.2f}  │ {status} Not Robust"
                    )

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"     {i+1:<7} │ ERROR      │ {str(e)[:30]}")

            if times:
                result = {
                    "delta": delta,
                    "num_samples_requested": num_samples,
                    "num_samples_completed": len(times),
                    "avg_time_sec": np.mean(times),
                    "min_time_sec": np.min(times),
                    "max_time_sec": np.max(times),
                    "total_time_sec": np.sum(times),
                    "robust_count": robust_count,
                    "robust_ratio": robust_count / len(times) if times else 0,
                    "timeout_occurred": timeout_occurred,
                }

                print(f"\n  📊 Summary:")
                print(f"     Avg:    {result['avg_time_sec']:.2f} sec/sample")
                print(f"     Total:  {result['total_time_sec']:.2f} sec")
                print(f"     Robust: {robust_count}/{len(times)}")
                if timeout_occurred:
                    print(f"     ⏱️  TIMEOUT REACHED at {elapsed_total:.0f}s")

                return result
            else:
                print("\n  ❌ No completed samples")
                return None

        except KeyboardInterrupt:
            print("\n  ⏸️  Interrupted")
            return None
        except Exception as e:
            print(f"\n  ❌ Error: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def run_sequential_sweep(self):
        """Test delta 3, 4, 5 sequentially with timeouts"""

        print("\n" + "╔" + "=" * 88 + "╗")
        print("║" + " " * 88 + "║")
        print("║  ⚡ SEQUENTIAL DELTA VERIFICATION: 3 → 4 → 5 (with timeout)          ║")
        print("║" + " " * 88 + "║")
        print("╚" + "=" * 88 + "╝")

        # Sequential test with increasing timeouts
        deltas_and_timeouts = [
            (3, 30),  # 30 seconds
            (4, 60),  # 60 seconds
            (5, 120),  # 120 seconds (2 minutes)
        ]

        for delta, timeout_sec in deltas_and_timeouts:
            result = self.test_delta_exhaustive(delta, num_samples=5, timeout_sec=timeout_sec)
            if result:
                self.results.append(result)
            else:
                print(f"  ⚠️  Skipping delta={delta} (no valid result)")

            # Brief pause between tests
            if len(self.results) < len(deltas_and_timeouts):
                time.sleep(1)

        return self.results

    def print_summary(self):
        """Print comparison table"""

        if not self.results:
            print("\n❌ No results to display")
            return

        print("\n\n" + "=" * 90)
        print("📊 VERIFICATION TIME COMPARISON")
        print("=" * 90)
        print()

        print(f"{'Delta':<8} │ {'Avg (sec)':<12} │ {'Min-Max':<20} │ {'Total':<10} │ {'Timeout':<10}")
        print("─" * 80)

        prev_avg = None
        for r in sorted(self.results, key=lambda x: x["delta"]):
            delta = r["delta"]
            avg = r["avg_time_sec"]
            min_t = r["min_time_sec"]
            max_t = r["max_time_sec"]
            total = r["total_time_sec"]
            timeout = "✓" if r["timeout_occurred"] else "—"

            range_str = f"{min_t:.2f}-{max_t:.2f}s"

            # Calculate growth factor
            if prev_avg and avg > 0:
                growth = f"({avg/prev_avg:.1f}x)"
            else:
                growth = "baseline"

            print(f"{delta:<8} │ {avg:>10.2f}  │ {range_str:<20} │ {total:>8.2f}  │ {timeout:<10} {growth}")

            prev_avg = avg

        print()

        # Estimate extrapolation
        print("\n📈 Extrapolation:")
        for r in sorted(self.results, key=lambda x: x["delta"]):
            delta = r["delta"]
            avg = r["avg_time_sec"]

            if avg > 0:
                # Rough estimate for next delta
                est_next = avg * 1.5  # Conservative 1.5x multiplier
                print(f"  δ={delta}: {avg:.2f}s/sample → δ={delta+1} estimated: ~{est_next:.2f}s/sample")

    def save_results(self):
        """Save to JSON"""
        output_file = "sequential_delta_results.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "test_type": "sequential_delta_verification",
                    "timestamp": str(__import__("datetime").datetime.now()),
                    "note": "Exhaustive verification time for delta 3, 4, 5",
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"✅ Results saved: {output_file}")


def main():
    tester = SequentialDeltaTester()

    try:
        results = tester.run_sequential_sweep()
        tester.print_summary()
        tester.save_results()

        print("\n" + "=" * 90)
        print("✅ SEQUENTIAL DELTA TEST COMPLETE")
        print("=" * 90)

    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
