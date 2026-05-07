"""ReLU-to-TTFS conversion (simple ×100 rescale) — paper Appendix D recipe.

Procedure (matches the 78% MNIST converted network used in Appendix D):
  1. Train a bias-free ReLU MLP on TTFS-quantized inputs (v_p ∈ {0,…,1}).
  2. Run a small ReLU-side threshold sweep so ReLU's effective thr is 1.0.
  3. Multiply all weights by 100 (= verifier's fixed θ) so the saved net runs
     with θ = 100 in the verifier's TTFS forward.

This matches the conversion behind `models/5_784_100_100_10` (78% acc).
We use it to extend the paper's multi-hidden suite to (n_h=200 depth=2)
and (depth=3) for v3-gpu cross-layer experiments.

Usage:
  python train_relu_to_ttfs_simple.py --hidden 200 200 --tmax 4 --epochs 30
  python train_relu_to_ttfs_simple.py --hidden 100 100 100 --tmax 4 --epochs 30
"""
from __future__ import annotations
import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mnist import MNIST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[200, 200])
    p.add_argument("--tmax", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--scale", type=float, default=100.0,
                   help="Multiplicative weight scale matching θ=100.")
    p.add_argument("--seed", type=int, default=0)
    code_dir = osp.dirname(osp.abspath(__file__))
    p.add_argument("--mnist-root", type=str,
                   default=osp.join(code_dir, "data/mnist/MNIST/raw/"))
    p.add_argument("--save-root", type=str,
                   default=osp.join(code_dir, "models/"))
    p.add_argument("--n-eval", type=int, default=1000)
    return p.parse_args()


def load_mnist_quantized(root: str, tmax: int):
    md = MNIST(root)
    Itr, Ltr = md.load_training()
    Ite, Lte = md.load_testing()
    Itr = np.array(Itr, dtype=np.float32) / 255.0
    Ite = np.array(Ite, dtype=np.float32) / 255.0
    qtr = np.floor(Itr * tmax).astype(np.float32) / tmax
    qte = np.floor(Ite * tmax).astype(np.float32) / tmax
    return qtr, np.array(Ltr), qte, np.array(Lte)


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


def main():
    args = parse_args()
    T = args.tmax + 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading MNIST… T={T}, device={device}")
    X_tr, y_tr, X_te, y_te = load_mnist_quantized(args.mnist_root, args.tmax)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False,
                         num_workers=0, pin_memory=True)

    dims = [784] + list(args.hidden) + [10]
    model = MLP(dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f"Training ReLU MLP {' -> '.join(map(str, dims))} ({args.epochs} epochs)…")
    t0 = time.time()
    best_te = 0.0
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
        best_te = max(best_te, te)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep:02d}: test_acc={te:.4f}")
    print(f"  ReLU best={best_te:.4f}  ({time.time()-t0:.1f}s)")

    # Extract & rescale: ReLU bias-free, weights (out, in)
    Ws = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            Ws.append(layer.weight.detach().cpu().numpy().astype(np.float64) * args.scale)

    # Reshape to verifier format
    arch = "_".join(["784"] + [str(h) for h in args.hidden] + ["10"])
    save_dir = osp.join(args.save_root, f"{T}_{arch}")
    os.makedirs(save_dir, exist_ok=True)

    # Layer 0: (n_h1, in=784) -> (n_h1, 28, 28)
    np.save(osp.join(save_dir, "weights_0.npy"),
            Ws[0].reshape(args.hidden[0], 28, 28))
    # Subsequent: (n_post, n_pre) -> (n_post, n_pre, 1)
    for i, w in enumerate(Ws[1:], start=1):
        np.save(osp.join(save_dir, f"weights_{i}.npy"), w[..., None])
    with open(osp.join(save_dir, "threshold.txt"), "w") as f:
        f.write("100\n")

    # Evaluate TTFS forward via verifier's forward()
    import sys
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from utils.config import CFG
    from utils.dictionary_mnist import threshold
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights

    cfg = CFG(log_name="convert", subtype="mnist", load_data_func=load_mnist,
              n_layer_neurons=tuple([784] + list(args.hidden) + [10]),
              layer_shapes=((28, 28),
                            *[(h, 1) for h in args.hidden], (10, 1)),
              num_steps=T, num_samples=10)
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=None)
    images, labels, _, _ = load_mnist(cfg)
    n_eval = min(args.n_eval, len(images))
    cor = sum(int(forward(cfg, weights, images[i].astype(int)) == labels[i])
              for i in range(n_eval))
    ttfs_acc = cor / n_eval
    print(f"\nTTFS forward acc (θ=100, n={n_eval}): {ttfs_acc:.4f}")
    with open(osp.join(save_dir, "convert_meta.txt"), "w") as f:
        f.write(f"relu_test_acc={best_te:.6f}\n")
        f.write(f"ttfs_test_acc_n{n_eval}={ttfs_acc:.6f}\n")
        f.write(f"scale={args.scale}\n")
        f.write(f"hidden={args.hidden}\n")
        f.write(f"T={T}\n")
    print(f"Saved -> {save_dir}/")


if __name__ == "__main__":
    main()
