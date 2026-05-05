"""Soundness/effectiveness check for multi-hidden BC-IBP.

For each sample:
  1. Run exhaustive DFS oracle to get ground-truth robust/not-robust.
  2. Run ibp_prove_robust_multilayer to see if it proves robust at root.
  3. Verify: if ibp says "robust", oracle says robust (no false positive).

Usage:
  python test_multilayer_bcibp.py --hidden 20 20 --tmax 4 --delta 2 --n-samples 5
"""
import argparse
import time
import numpy as np
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.dictionary_mnist import threshold
from bnb_ibp import ibp_prove_robust_multilayer, ibp_prove_robust_budgeted


parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, nargs="+", default=[20, 20])
parser.add_argument("--tmax", type=int, default=4)
parser.add_argument("--delta", type=int, default=2)
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exhaustive-timeout", type=float, default=120.0)
args = parser.parse_args()

n_layer_neurons = tuple([784] + list(args.hidden) + [10])
layer_shapes = (
    (28, 28),
    *[(h, 1) for h in args.hidden],
    (10, 1),
)

cfg = CFG(
    log_name="multilayer_test",
    subtype="mnist",
    load_data_func=load_mnist,
    seed=args.seed,
    num_samples=args.n_samples,
    deltas=(args.delta,),
    n_layer_neurons=n_layer_neurons,
    layer_shapes=layer_shapes,
    num_steps=args.tmax + 1,
)
print(f"Architecture: {n_layer_neurons}, T={args.tmax+1}, delta={args.delta}")

weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
print(f"Loaded {len(weights)} weight layers")
for i, w in enumerate(weights):
    print(f"  weights[{i}]: shape {w.shape}")

# Sample selection (deterministic)
images, labels, *_ = load_mnist(cfg)
np.random.seed(args.seed)
indices = np.random.permutation(len(images))[:args.n_samples]

T = args.tmax + 1
delta = args.delta


def exhaustive_oracle(img, orig_pred, delta, timeout=120.0):
    """DFS over all admissible TTFS perturbations within L1 budget delta."""
    found = [False]
    n_pixels = 28 * 28
    pixels = [(i, j) for i in range(28) for j in range(28)]
    tic = time.time()
    timed_out = [False]

    def dfs(current_img, pixel_pos, rem_budget):
        if found[0] or timed_out[0]:
            return
        if time.time() - tic > timeout:
            timed_out[0] = True
            return
        # Adversarial check
        pred = forward(cfg, weights, current_img)
        if pred != orig_pred:
            l1 = int(np.sum(np.abs(current_img.astype(int) - img.astype(int))))
            if l1 <= delta:
                found[0] = True
                return
        if pixel_pos == n_pixels or rem_budget == 0:
            return
        px, py = pixels[pixel_pos]
        orig_val = int(current_img[px, py])
        # Try all values within budget for this pixel, then recurse
        for new_val in range(T):
            cost = abs(new_val - orig_val)
            if cost > rem_budget:
                continue
            current_img[px, py] = new_val
            dfs(current_img, pixel_pos + 1, rem_budget - cost)
            if found[0] or timed_out[0]:
                current_img[px, py] = orig_val
                return
            current_img[px, py] = orig_val

    dfs(img.copy(), 0, delta)
    return ("not_robust" if found[0] else
            ("timeout" if timed_out[0] else "robust"))


print(f"\n{'sample':>6} {'orig':>4} {'oracle':>10} {'time(o)':>8} {'mlIBP':>6} {'time(i)':>8}  match?")
print("-" * 70)
mismatches = 0
for idx in indices:
    img = images[idx].astype(int)
    orig_pred = int(forward(cfg, weights, img))

    # Oracle
    tic = time.time()
    oracle_v = exhaustive_oracle(img, orig_pred, delta, timeout=args.exhaustive_timeout)
    oracle_t = time.time() - tic

    # Multi-layer BC-IBP at root: check across all (rem_neg, rem_pos) splits
    tic = time.time()
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)], dtype=int)
    proved_all = True
    for rn in range(delta + 1):
        rp = delta - rn
        ok = ibp_prove_robust_multilayer(
            pixel_img=img.copy(),
            remaining_pixels=all_pixels,
            rem_neg=rn, rem_pos=rp,
            weights_list=list(weights),
            num_steps=T,
            threshold=threshold,
            orig_pred=orig_pred,
            num_classes=10,
        )
        if not ok:
            proved_all = False
            break
    ibp_t = time.time() - tic
    ibp_v = "robust" if proved_all else "inconclusive"

    # Soundness check: if BC-IBP says "robust", oracle must agree
    if ibp_v == "robust" and oracle_v == "not_robust":
        match = "FAIL!"
        mismatches += 1
    else:
        match = "ok"

    print(f"{idx:>6} {orig_pred:>4} {oracle_v:>10} {oracle_t:>7.2f}s {ibp_v:>6} {ibp_t:>7.2f}s  {match}")

print(f"\nSoundness: {mismatches} mismatch(es) out of {args.n_samples} samples")
