"""Unit tests for bnb_ibp.ibp_prove_robust.

Tests:
  (a) Budget=0 → bounds should match forward exactly (spike times deterministic).
  (b) Budget > 0 on KNOWN robust samples → IBP should return True (or False if bound is loose).
  (c) Budget > 0 on KNOWN adversarial samples → IBP must NOT return True (soundness).
      If it ever returns True on an adversarial sample, IBP is UNSOUND.
"""
import numpy as np
from random import seed as rseed, sample as rsample
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights
from utils.dictionary_mnist import threshold
from bnb_ibp import ibp_prove_robust


def setup(n_hidden=200, num_steps=5, delta=2, num_samples=10):
    cfg = CFG(
        log_name="ibp_test",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=42,
        num_samples=num_samples,
        deltas=(delta,),
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=num_steps,
    )
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist(cfg)
    rseed(42); np.random.seed(42)
    picks = []
    for sno in rsample([*range(len(images))], k=num_samples * 3):
        img = images[sno]
        ft = []
        orig_pred = forward(cfg, weights, img, ft)
        if len(np.argwhere(ft[-1] == np.min(ft[-1]))[0]) != 1:
            continue
        picks.append((sno, img, int(orig_pred)))
        if len(picks) >= num_samples:
            break
    return cfg, weights, picks


def test_budget_zero_matches_forward():
    """With rem_neg=rem_pos=0 and no remaining pixels (all fixed at originals),
    bounds collapse to the actual forward-pass trajectory — must prove robustness
    iff the baseline prediction already dominates with strict margin (it should).
    """
    cfg, weights, picks = setup(n_hidden=100, delta=1, num_samples=5)
    w1, w2 = weights[0], weights[1][:, :, 0]
    passed = 0
    for sno, img, orig_pred in picks:
        result = ibp_prove_robust(
            pixel_img=img.astype(int),
            remaining_pixels=np.zeros((0, 2), dtype=int),
            rem_neg=0, rem_pos=0,
            w1=w1, w2_flat=w2,
            num_steps=cfg.num_steps, threshold=threshold,
            orig_pred=orig_pred, num_classes=10,
        )
        # With no perturbation, result depends solely on baseline spike times
        ft = []
        forward(cfg, weights, img, ft)
        baseline = ft[-1]
        # Baseline is robust iff pred = argmin(baseline), which is true for our picks (we filtered ties).
        # IBP should prove this robustness: s_o_min and s_o_max collapse to the baseline spike times.
        assert result, f"budget=0: IBP failed to prove on sample {sno}"
        passed += 1
    print(f"[test_budget_zero_matches_forward] passed {passed}/{len(picks)}")


def test_ibp_sound_vs_exhaustive(delta=2, n_hidden=200):
    """For each sample, compare IBP verdict (prove-robust / unknown) against
    exhaustive DFS ground truth. IBP must never say `True (robust)` on a
    not-robust sample — that would be UNSOUND.

    Missed prunes (IBP says False on robust sample) are fine — just loose.
    """
    cfg, weights, picks = setup(n_hidden=n_hidden, delta=delta, num_samples=10)
    w1, w2 = weights[0], weights[1][:, :, 0]

    # Exhaustive DFS (same as oracle in realistic_large_benchmark.py)
    def exhaustive(img, orig_pred):
        pixels = [(i, j) for i in range(28) for j in range(28)]
        found = [False]
        def dfs(cur, pos, rem):
            if found[0]: return
            spks = []
            forward(cfg, weights, cur, spks)
            last = spks[-1]
            tgt = last[orig_pred]
            min_nt = min(last[i] for i in range(10) if i != orig_pred)
            is_adv = (min_nt < tgt) or (min_nt == tgt and any(last[i] == tgt for i in range(orig_pred)))
            if is_adv:
                if int(np.sum(np.abs(cur.astype(int) - img.astype(int)))) <= delta:
                    found[0] = True; return
            if pos == len(pixels) or rem == 0: return
            x, y = pixels[pos]
            ov = int(cur[x, y])
            for v in range(cfg.num_steps):
                c = abs(v - ov)
                if c > rem: continue
                cur[x, y] = v
                dfs(cur, pos + 1, rem - c)
                cur[x, y] = ov
                if found[0]: return
        dfs(img.copy(), 0, delta)
        return found[0]

    # At root, all 784 pixels are remaining with full budget split (rem_neg+rem_pos=delta)
    # For the tightest IBP bound we check each budget split; if ANY proves robust, prune.
    all_pixels = np.array([(i, j) for i in range(28) for j in range(28)], dtype=int)

    unsound = 0
    ibp_proved = 0
    rob_count = 0
    nrob_count = 0
    for sno, img, orig_pred in picks:
        nrob = exhaustive(img, orig_pred)
        if nrob:
            nrob_count += 1
        else:
            rob_count += 1

        # Try each budget split at root
        prove = False
        for rn in range(delta + 1):
            rp = delta - rn
            if ibp_prove_robust(
                pixel_img=img.astype(int),
                remaining_pixels=all_pixels,
                rem_neg=rn, rem_pos=rp,
                w1=w1, w2_flat=w2,
                num_steps=cfg.num_steps, threshold=threshold,
                orig_pred=orig_pred, num_classes=10,
            ):
                continue  # this split proves robust, try next
            else:
                prove = False
                break
        else:
            prove = True  # all splits proved robust

        tag = f"sample={sno} orig_pred={orig_pred} exhaustive={'NR' if nrob else 'R'} IBP_proves={prove}"
        if prove and nrob:
            print(f"[UNSOUND] {tag}")
            unsound += 1
        elif prove:
            ibp_proved += 1
            print(f"[OK]     {tag}")
        else:
            print(f"[unknown] {tag}")

    print(f"\nResults @ delta={delta}, n_h={n_hidden}: {rob_count} robust, {nrob_count} not-robust")
    print(f"IBP proved: {ibp_proved}/{rob_count} robust samples ({100*ibp_proved/max(1,rob_count):.0f}%)")
    print(f"IBP unsound: {unsound} (MUST be 0)")
    assert unsound == 0, "SOUNDNESS VIOLATION — IBP produced false robust on not-robust sample"


if __name__ == "__main__":
    test_budget_zero_matches_forward()
    print()
    test_ibp_sound_vs_exhaustive(delta=1, n_hidden=100)
    print()
    test_ibp_sound_vs_exhaustive(delta=2, n_hidden=200)
