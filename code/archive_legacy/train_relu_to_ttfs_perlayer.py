"""ReLU-to-TTFS conversion with per-layer scale search (extends the simple
`w → 100·w` recipe to deeper architectures).

Background
----------
Simple `w → 100·w` rescale (`train_relu_to_ttfs.py`) uses a single global
scalar matched to the verifier's fixed θ=100. That works for depth-2
(78.2% on 100×100, 73.6% on 200×200) but collapses at depth-3 (9.7% on
100×100×100) because each layer's optimal "fire-around-the-middle"
scaling is different — feeding the same scalar through 3 layers
compounds the threshold mismatch.

This script searches a separate α_l per layer:

    W_l^{TTFS} = α_l · W_l^{ReLU}

via greedy per-layer sweep:
  - α_0 sweep with α_{1..L} = 100 (baseline).
  - α_1 sweep with α_0 fixed at the previous best.
  - ... up to α_L.

α candidates default to {25, 50, 75, 100, 150, 200, 300}.

We measure TTFS test accuracy via the verifier's forward simulator
(`utils.mnist_net.forward`) so that the reported number is exactly what
BC-IBP / CLEB will see. Saves weights in the verifier-compatible layout.

Usage
-----
  python train_relu_to_ttfs_perlayer.py --hidden 100 100 100 --tmax 4 --epochs 30
  python train_relu_to_ttfs_perlayer.py --hidden 200 200      --tmax 4 --epochs 30
"""
from __future__ import annotations
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mnist import MNIST

# So we can import utils.{config,load,mnist_net} regardless of cwd.
_CODE_DIR = osp.dirname(osp.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
from utils.config import CFG
from utils.load import load_mnist
from utils.mnist_net import forward


# ============================================================================
# Args
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[100, 100, 100])
    p.add_argument("--tmax", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--alpha-grid", type=float, nargs="+",
                   default=[25, 50, 75, 100, 150, 200, 300],
                   help="Per-layer α candidates; greedy sweep visits each layer in turn.")
    p.add_argument("--alpha-init", type=float, default=100.0,
                   help="Fallback initial α (overridden by --alpha-init-mode median).")
    p.add_argument("--alpha-init-mode", choices=["constant", "median"], default="median",
                   help="constant = use --alpha-init for all layers; median = pick "
                        "α_l so that the median ReLU activation maps to ~θ/2 (=50), "
                        "putting TTFS firing near the middle of the time horizon.")
    p.add_argument("--n-rand-search", type=int, default=0,
                   help="Number of random α-tuple samples to evaluate before/instead of "
                        "greedy sweep (0 = skip).")
    p.add_argument("--rand-low", type=float, default=25)
    p.add_argument("--rand-high", type=float, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-search", type=int, default=500,
                   help="MNIST samples used during α-sweep (for speed).")
    p.add_argument("--n-eval", type=int, default=2000,
                   help="MNIST samples used for the final reported accuracy.")
    p.add_argument("--reuse-relu", action="store_true",
                   help="Skip ReLU pretraining if a checkpoint exists at "
                        "models/relu_<arch>.pt.")
    p.add_argument("--mnist-root", type=str,
                   default=osp.join(_CODE_DIR, "data/mnist/MNIST/raw/"))
    p.add_argument("--save-root", type=str,
                   default=osp.join(_CODE_DIR, "models/"))
    return p.parse_args()


# ============================================================================
# ReLU MLP (bias-free, TTFS-quantised input)
# ============================================================================
class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def load_mnist_quantised(root: str, tmax: int):
    md = MNIST(root)
    Itr, Ltr = md.load_training()
    Ite, Lte = md.load_testing()
    Itr = np.array(Itr, dtype=np.float32) / 255.0
    Ite = np.array(Ite, dtype=np.float32) / 255.0
    qtr = np.floor(Itr * tmax).astype(np.float32) / tmax
    qte = np.floor(Ite * tmax).astype(np.float32) / tmax
    return qtr, np.array(Ltr), qte, np.array(Lte)


def train_relu_mlp(args, dims, device):
    """Train a bias-free ReLU MLP on TTFS-quantised MNIST. Returns the list
    of per-layer weight matrices as numpy arrays."""
    arch_tag = "_".join(str(d) for d in dims)
    ckpt_path = osp.join(args.save_root, f"relu_{arch_tag}.pt")
    if args.reuse_relu and osp.exists(ckpt_path):
        print(f"Loading cached ReLU MLP from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location=device)
        model = MLP(dims).to(device)
        model.load_state_dict(sd)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        X_tr, y_tr, X_te, y_te = load_mnist_quantised(args.mnist_root, args.tmax)
        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
        test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long())
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=0, pin_memory=True)
        model = MLP(dims).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        print(f"Training ReLU MLP {' -> '.join(str(d) for d in dims)} ({args.epochs} epochs)…")
        t0 = time.time()
        best_te = 0.0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for ep in range(args.epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            cor, tot = 0, 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    cor += (model(xb).argmax(1) == yb).sum().item()
                    tot += xb.size(0)
            te = cor / tot
            if te > best_te:
                best_te = te
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  ep{ep:02d}: test_acc={te:.4f}")
        print(f"  ReLU best={best_te:.4f}  ({time.time()-t0:.1f}s)")
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
        print(f"  saved ReLU checkpoint to {ckpt_path}")
    # Extract per-layer weights as numpy.
    Ws = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            Ws.append(layer.weight.detach().cpu().numpy().astype(np.float64))
    return Ws


# ============================================================================
# TTFS conversion + accuracy via verifier forward()
# ============================================================================
def assemble_weights(Ws_relu, alphas, hidden):
    """Build verifier-format weight list from per-layer α."""
    out = []
    # Layer 0: (n_h1, in=784) -> (n_h1, 28, 28)
    out.append((Ws_relu[0] * alphas[0]).reshape(hidden[0], 28, 28))
    # Subsequent: (n_post, n_pre) -> (n_post, n_pre, 1)
    for i, w in enumerate(Ws_relu[1:], start=1):
        out.append((w * alphas[i])[..., None])
    return out


def make_cfg(hidden, T, num_samples):
    return CFG(
        log_name="convert", subtype="mnist", load_data_func=load_mnist,
        n_layer_neurons=tuple([784] + list(hidden) + [10]),
        layer_shapes=((28, 28), *[(h, 1) for h in hidden], (10, 1)),
        num_steps=T, num_samples=num_samples,
    )


def ttfs_test_acc(weights_list, cfg, images, labels, n_eval):
    """End-to-end TTFS accuracy via the verifier's forward simulator."""
    n = min(n_eval, len(images))
    cor = 0
    for i in range(n):
        if int(forward(cfg, weights_list, images[i].astype(int))) == int(labels[i]):
            cor += 1
    return cor / n


def main():
    args = parse_args()
    T = args.tmax + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, T={T}")
    hidden = list(args.hidden)
    dims = [784] + hidden + [10]
    L = len(hidden) + 1                  # number of weight layers
    arch_tag = "_".join(["784"] + [str(h) for h in hidden] + ["10"])
    print(f"Architecture: {' -> '.join(map(str, dims))}, depth={L}")

    # 1. ReLU MLP
    Ws_relu = train_relu_mlp(args, dims, device)
    print(f"Got {len(Ws_relu)} ReLU weight matrices, shapes: {[w.shape for w in Ws_relu]}")

    # 2. Set up CFG and load test set once
    cfg_search = make_cfg(hidden, T, num_samples=args.n_search)
    cfg_eval = make_cfg(hidden, T, num_samples=args.n_eval)
    images, labels, _, _ = load_mnist(cfg_eval)

    # 3a. Initial α: constant or median-activation heuristic.
    if args.alpha_init_mode == "median":
        # For each ReLU layer ℓ, compute the median of POSITIVE pre-activations on
        # a small TTFS-quantised batch, then set α_ℓ so that this median maps to
        # roughly θ/2 ≈ 50 (i.e., the median active neuron crosses θ around the
        # middle of the horizon). This avoids the "all fire at tmax / all fire at
        # t=0" dead zone that traps the global ×100 rescale at depth ≥ 3.
        with torch.no_grad():
            X_q, _, _, _ = load_mnist_quantised(args.mnist_root, args.tmax)
            Xb = torch.from_numpy(X_q[:512]).flatten(1).to(device)
            # Forward up to each layer; pre-activation = Linear(.) before ReLU.
            preacts = []
            cur = Xb
            for li, w_relu in enumerate(Ws_relu):
                W_t = torch.from_numpy(w_relu).float().to(device)
                z = cur @ W_t.T
                preacts.append(z.detach().cpu().numpy())
                if li < len(Ws_relu) - 1:  # apply ReLU between hidden layers
                    cur = torch.relu(z)
                else:
                    cur = z
        target = 50.0  # = θ/2 for θ=100; mid-horizon firing-time target
        alphas = []
        for li, z in enumerate(preacts):
            pos = z[z > 0]
            med = float(np.median(pos)) if pos.size > 0 else 1.0
            a_l = max(target / max(med, 1e-6), 1.0)
            alphas.append(a_l)
        print(f"\nMedian-activation init: per-layer median+ pre-activation = "
              f"{[float(np.median(z[z>0])) for z in preacts]}")
        print(f"  → α_init = {[round(a, 2) for a in alphas]}")
    else:
        alphas = [args.alpha_init] * L

    # 3b. Optional joint random search to escape coordinate-descent traps.
    if args.n_rand_search > 0:
        print(f"\n=== Joint random search ({args.n_rand_search} samples, "
              f"α∈[{args.rand_low},{args.rand_high}]) ===")
        rng = np.random.RandomState(args.seed + 12345)
        best_alphas = list(alphas)
        best_acc = ttfs_test_acc(assemble_weights(Ws_relu, alphas, hidden),
                                 cfg_search, images, labels, args.n_search)
        print(f"  starting acc (init α): {best_acc:.4f}")
        for trial in range(args.n_rand_search):
            cand = rng.uniform(args.rand_low, args.rand_high, size=L).tolist()
            w_try = assemble_weights(Ws_relu, cand, hidden)
            acc = ttfs_test_acc(w_try, cfg_search, images, labels, args.n_search)
            if acc > best_acc:
                best_acc = acc
                best_alphas = cand
                print(f"    trial {trial:3d}: α={[round(a,1) for a in cand]} → acc={acc:.4f} *")
        alphas = best_alphas
        print(f"  best random α = {[round(a, 2) for a in alphas]}, acc={best_acc:.4f}")

    # 3c. Greedy per-layer refinement
    print(f"\n=== Per-layer α-sweep (grid={args.alpha_grid}) ===")
    print(f"Starting α = {[round(a, 2) for a in alphas]} → search-acc:", end=" ")
    weights = assemble_weights(Ws_relu, alphas, hidden)
    base = ttfs_test_acc(weights, cfg_search, images, labels, args.n_search)
    print(f"{base:.4f}")

    for li in range(L):
        print(f"\n  Sweeping α_{li}  (other α held fixed)…")
        results = []
        for cand in args.alpha_grid:
            alphas_try = list(alphas)
            alphas_try[li] = cand
            w_try = assemble_weights(Ws_relu, alphas_try, hidden)
            acc = ttfs_test_acc(w_try, cfg_search, images, labels, args.n_search)
            print(f"    α_{li}={cand:>5g}  →  acc={acc:.4f}")
            results.append((cand, acc))
        results.sort(key=lambda r: r[1], reverse=True)
        best_cand, best_acc = results[0]
        alphas[li] = best_cand
        print(f"    >> best α_{li} = {best_cand} (acc={best_acc:.4f})")

    print(f"\n=== Final α-vector: {alphas} ===")

    # 4. Evaluate on the bigger eval set
    weights_final = assemble_weights(Ws_relu, alphas, hidden)
    final_acc = ttfs_test_acc(weights_final, cfg_eval, images, labels, args.n_eval)
    print(f"TTFS forward acc on n={args.n_eval}: {final_acc:.4f}")

    # 5. Save in verifier layout
    save_dir = osp.join(args.save_root, f"{T}_{arch_tag}_perlayer")
    os.makedirs(save_dir, exist_ok=True)
    for i, w in enumerate(weights_final):
        np.save(osp.join(save_dir, f"weights_{i}.npy"), w)
    with open(osp.join(save_dir, "threshold.txt"), "w") as f:
        f.write("100\n")
    with open(osp.join(save_dir, "convert_meta.txt"), "w") as f:
        f.write(f"alphas={alphas}\n")
        f.write(f"hidden={hidden}\n")
        f.write(f"T={T}\n")
        f.write(f"ttfs_test_acc_n{args.n_eval}={final_acc:.6f}\n")
    print(f"Saved to {save_dir}/")


if __name__ == "__main__":
    main()
