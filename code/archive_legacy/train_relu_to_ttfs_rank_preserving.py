"""ReLU-to-TTFS conversion v2: rank-preserving rescaling for multi-hidden TTFS.

Strategy
--------
1. Train a bias-free ReLU MLP on TTFS-quantized MNIST inputs (v_p ∈ {0, 1/(T-1), …, 1}).
2. Convert layer-by-layer with **rank-preserving rescaling**:
     - Run ReLU MLP on training data; for each hidden layer l, take the median
       positive pre-activation \\bar z_l.
     - TTFS hidden neurons must reach the (fixed) threshold θ=100 at roughly
       the middle of the time horizon, so the firing-time argmin ranking
       matches the ReLU pre-activation ranking. We pick a per-layer weight
       scaling \\alpha_l so that V_h(T-2) ≈ θ on the median active neuron.
3. Re-evaluate end-to-end TTFS accuracy on the verifier's exact forward
   simulator. Save weights in the project's standard layout.

Why ranks suffice
-----------------
At each layer the verifier computes argmax_h V_h(t) > θ over t ∈ [0, T-1].
If the ranking of hidden activations is preserved layer-wise, the output
argmin in the final layer is dominated by the same logits. Concretely we
match the ReLU-active hidden neurons to "fire by t ≤ T-2" (no clamp), and
the inactive ones to "clamp to T-1". Since θ is fixed at 100, all
calibration is done by per-layer weight scaling.

Usage
-----
  python train_relu_to_ttfs_v2.py --hidden 100 100 --tmax 4 --epochs 30
  python train_relu_to_ttfs_v2.py --hidden 200 200 200 --tmax 4 --epochs 30
"""
from __future__ import annotations
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mnist import MNIST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[100, 100])
    p.add_argument("--tmax", type=int, default=4, help="time horizon T = tmax + 1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--threshold", type=float, default=100.0,
                   help="Verifier threshold (project-wide constant)")
    p.add_argument("--target-cross-time", type=int, default=None,
                   help="Time step at which the median ACTIVE hidden neuron "
                        "should cross threshold. Default: T-2.")
    p.add_argument("--seed", type=int, default=0)
    code_dir = osp.dirname(osp.abspath(__file__))
    p.add_argument("--mnist-root", type=str,
                   default=osp.join(code_dir, "data/mnist/MNIST/raw/"))
    p.add_argument("--save-root", type=str,
                   default=osp.join(code_dir, "models/"))
    p.add_argument("--n-eval", type=int, default=2000,
                   help="MNIST test samples to evaluate TTFS accuracy on.")
    return p.parse_args()


# ============================================================================
# Data
# ============================================================================
def load_mnist_quantized(root: str, T: int):
    """Load MNIST and quantize pixel values to {0, 1/(T-1), …, 1}."""
    md = MNIST(root)
    Itr, Ltr = md.load_training()
    Ite, Lte = md.load_testing()
    Itr = np.array(Itr, dtype=np.float32) / 255.0
    Ite = np.array(Ite, dtype=np.float32) / 255.0
    # Quantize so the ReLU MLP sees the same input alphabet as the SNN.
    qtr = np.floor(Itr * (T - 1)).astype(np.float32) / (T - 1)
    qte = np.floor(Ite * (T - 1)).astype(np.float32) / (T - 1)
    return qtr, np.array(Ltr), qte, np.array(Lte)


# ============================================================================
# ReLU MLP (bias-free)
# ============================================================================
class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def train_relu_mlp(args, X_tr, y_tr, X_te, y_te, dims, device):
    model = MLP(dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False,
                         num_workers=0, pin_memory=True)
    print(f"\nTraining ReLU MLP {' -> '.join(map(str, dims))}, "
          f"{args.epochs} epochs, lr={args.lr}, wd={args.weight_decay}")
    best_te_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        cor, tot, ls = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls += loss.item() * xb.size(0)
            cor += (logits.argmax(1) == yb).sum().item()
            tot += xb.size(0)
        model.eval()
        cte, tte = 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                cte += (model(xb).argmax(1) == yb).sum().item()
                tte += xb.size(0)
        te_acc = cte / tte
        best_te_acc = max(best_te_acc, te_acc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep:02d}: train_loss={ls/tot:.4f}, "
                  f"train_acc={cor/tot:.4f}, test_acc={te_acc:.4f}")
    print(f"  Final test_acc={te_acc:.4f}, best={best_te_acc:.4f}")
    return model, te_acc


def extract_weights(model: nn.Module) -> list[np.ndarray]:
    Ws: list[np.ndarray] = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            Ws.append(layer.weight.detach().cpu().numpy())  # (out, in)
    return Ws


# ============================================================================
# TTFS forward simulator (matches utils/mnist_net.py exactly)
# ============================================================================
def ttfs_forward_batch(images_int: np.ndarray, W_list: list[np.ndarray],
                       threshold: float, T: int) -> np.ndarray:
    """Vectorized TTFS forward over a batch.

    images_int:  (B, 28, 28) integer spike times in [0, T-1]
    W_list:
      - W_list[0]: (n_h1, 28, 28)
      - W_list[k] (k≥1): (n_post, n_pre, 1) — last axis is the residual "1"
                          dim left over from the project's spatial layout
    Returns predicted classes, shape (B,).
    """
    B = images_int.shape[0]
    H, W = images_int.shape[1], images_int.shape[2]
    # Build (B, H, W, T+1) input spike tensor
    SpikeImage = np.zeros((B, H, W, T + 1), dtype=np.float32)
    bx, hx, wx = np.mgrid[0:B, 0:H, 0:W].reshape(3, -1)
    SpikeImage[bx, hx, wx, images_int.flatten()] = 1.0

    # Layer 0: input -> hidden_1. weight shape (n_h1, H, W)
    W0 = W_list[0].astype(np.float32)
    n_post = W0.shape[0]
    # Voltage[b, h, t] = Σ_{x,y} W0[h,x,y] * SpikeImage[b,x,y, ≤ t]
    pre = np.einsum('hxy,bxyt->bht', W0, SpikeImage)  # contribution at exact t
    voltage = np.cumsum(pre, axis=2)
    voltage[:, :, T - 1] = threshold + 1
    ft = (np.argmax(voltage > threshold, axis=2) + 1).astype(int)
    ft = np.minimum(ft, T - 1)

    # Layers 1+
    for k in range(1, len(W_list)):
        Wk = W_list[k].astype(np.float32)
        if Wk.ndim == 3:
            Wk2d = Wk[..., 0]  # (n_post, n_pre)
        else:
            Wk2d = Wk
        n_pre = Wk2d.shape[1]
        n_post_k = Wk2d.shape[0]
        # pre-spike times: ft, shape (B, n_pre). Build (B, n_pre, T+1) one-hot.
        spike_pre = np.zeros((B, n_pre, T + 1), dtype=np.float32)
        bp, ip = np.mgrid[0:B, 0:n_pre].reshape(2, -1)
        spike_pre[bp, ip, ft.flatten()] = 1.0
        pre_v = np.einsum('op,bpt->bot', Wk2d, spike_pre)
        voltage = np.cumsum(pre_v, axis=2)
        voltage[:, :, T - 1] = threshold + 1
        ft = (np.argmax(voltage > threshold, axis=2) + 1).astype(int)
        ft = np.minimum(ft, T - 1)
    return np.argmin(ft, axis=1)


# ============================================================================
# Per-layer rank-preserving rescaling (Stanojevic-style)
# ============================================================================
def calibrate_per_layer_scale(
    W_relu: list[np.ndarray],
    X_tr_q: np.ndarray,
    threshold: float,
    T: int,
    target_cross_time: int,
    n_calib: int = 4096,
) -> tuple[list[np.ndarray], list[float]]:
    """Pick per-layer scale α_l so that the median active hidden neuron's
    cumulative voltage crosses threshold around t=target_cross_time.

    Procedure:
      For each layer l:
        - simulate TTFS forward with current rescaled W (layers <= l-1) to get
          spike times entering layer l.
        - compute V_h(target_cross_time) for each (sample, h).
        - among (sample, h) with V_h > 0, take the median value m_l.
        - α_l = threshold / m_l   (so median active neuron's V_h(target_cross_time) ≈ θ).

    Returns rescaled weights and the chosen scales.
    """
    np.random.seed(0)
    idx = np.random.choice(X_tr_q.shape[0], n_calib, replace=False)
    images_q = X_tr_q[idx].reshape(-1, 28, 28)  # (B, 28, 28) ∈ [0,1]
    images_int = np.floor((1.0 - images_q) * (T - 1)).astype(int)  # spike times

    H, W = images_int.shape[1], images_int.shape[2]
    B = images_int.shape[0]
    SpikeImage = np.zeros((B, H, W, T + 1), dtype=np.float32)
    bx, hx, wx = np.mgrid[0:B, 0:H, 0:W].reshape(3, -1)
    SpikeImage[bx, hx, wx, images_int.flatten()] = 1.0

    W_scaled: list[np.ndarray] = []
    scales: list[float] = []
    # Reshape ReLU layer 0: (n_h1, 784) -> (n_h1, 28, 28)
    n_h1 = W_relu[0].shape[0]
    W0_rs = W_relu[0].reshape(n_h1, 28, 28).astype(np.float32)

    pre = np.einsum('hxy,bxyt->bht', W0_rs, SpikeImage)
    V0 = np.cumsum(pre, axis=2)  # (B, n_h1, T+1)
    V_at_target = V0[:, :, target_cross_time]  # (B, n_h1)
    pos = V_at_target[V_at_target > 0]
    if pos.size < 10:
        m = max(1.0, V_at_target.max())
    else:
        m = float(np.median(pos))
    alpha0 = threshold / m
    W0_scaled = W0_rs * alpha0
    W_scaled.append(W0_scaled)
    scales.append(alpha0)

    # Now simulate forward through layer 0 (with new scale) to get ft_layer0.
    pre0 = np.einsum('hxy,bxyt->bht', W0_scaled, SpikeImage)
    V0s = np.cumsum(pre0, axis=2)
    V0s[:, :, T - 1] = threshold + 1
    ft = (np.argmax(V0s > threshold, axis=2) + 1).astype(int)
    ft = np.minimum(ft, T - 1)

    # For layers 1..L
    for l in range(1, len(W_relu)):
        Wl = W_relu[l].astype(np.float32)  # (n_post, n_pre) for ReLU
        n_post = Wl.shape[0]
        n_pre = Wl.shape[1]
        # Build (B, n_pre, T+1) one-hot from ft
        spike_pre = np.zeros((B, n_pre, T + 1), dtype=np.float32)
        bp, ip = np.mgrid[0:B, 0:n_pre].reshape(2, -1)
        spike_pre[bp, ip, ft.flatten()] = 1.0
        pre_l = np.einsum('op,bpt->bot', Wl, spike_pre)
        Vl = np.cumsum(pre_l, axis=2)
        V_at_target_l = Vl[:, :, target_cross_time]
        pos = V_at_target_l[V_at_target_l > 0]
        if pos.size < 10:
            m = max(1.0, V_at_target_l.max())
        else:
            m = float(np.median(pos))
        alpha_l = threshold / m
        Wl_scaled = Wl * alpha_l
        # Project shape: (n_post, n_pre, 1)
        W_scaled.append(Wl_scaled[..., None])
        scales.append(alpha_l)

        # Forward this layer with scaled weights to feed the next layer
        pre_ls = np.einsum('op,bpt->bot', Wl_scaled, spike_pre)
        Vls = np.cumsum(pre_ls, axis=2)
        Vls[:, :, T - 1] = threshold + 1
        ft = (np.argmax(Vls > threshold, axis=2) + 1).astype(int)
        ft = np.minimum(ft, T - 1)
    return W_scaled, scales


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    T = args.tmax + 1
    target_t = args.target_cross_time if args.target_cross_time is not None else T - 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading MNIST (T={T}, T-1 quantization)…")
    X_tr, y_tr, X_te, y_te = load_mnist_quantized(args.mnist_root, T)
    print(f"  Train {X_tr.shape}, Test {X_te.shape}, "
          f"alphabet ≈ {sorted(set(np.unique(X_tr)))[:5]}…")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    dims = [784] + list(args.hidden) + [10]
    model, relu_te_acc = train_relu_mlp(args, X_tr, y_tr, X_te, y_te, dims, device)
    W_relu = extract_weights(model)
    print("\nReLU weight shapes:", [w.shape for w in W_relu])

    print(f"\nCalibrating per-layer scales (target_cross_time={target_t}, θ={args.threshold})…")
    W_scaled, scales = calibrate_per_layer_scale(
        W_relu, X_tr, args.threshold, T, target_t,
    )
    for i, (a, w) in enumerate(zip(scales, W_scaled)):
        print(f"  layer {i}: α={a:.4g}, weight shape={w.shape}, "
              f"|w|.max={np.abs(w).max():.3f}, |w|.mean={np.abs(w).mean():.3f}")

    # Evaluate TTFS accuracy on test set (vectorized, batched)
    print(f"\nEvaluating TTFS accuracy on first {args.n_eval} test samples…")
    images_int_te = np.floor((1.0 - X_te[:args.n_eval]) * (T - 1)).astype(int)
    images_int_te = images_int_te.reshape(-1, 28, 28)
    bs = 512
    preds = np.empty(args.n_eval, dtype=int)
    for s in range(0, args.n_eval, bs):
        e = min(s + bs, args.n_eval)
        preds[s:e] = ttfs_forward_batch(
            images_int_te[s:e], W_scaled, args.threshold, T,
        )
    ttfs_acc = float((preds == y_te[:args.n_eval]).mean())
    print(f"  TTFS test acc: {ttfs_acc:.4f}  (vs. ReLU best={relu_te_acc:.4f})")

    # Save in project's standard format
    arch = "_".join(["784"] + [str(h) for h in args.hidden] + ["10"])
    save_dir = osp.join(args.save_root, f"{T}_{arch}")
    os.makedirs(save_dir, exist_ok=True)
    for i, w in enumerate(W_scaled):
        np.save(osp.join(save_dir, f"weights_{i}.npy"), w.astype(np.float64))
    with open(osp.join(save_dir, "threshold.txt"), "w") as f:
        f.write(f"{args.threshold}\n")
    # Sidecar: also save ReLU + TTFS metrics
    with open(osp.join(save_dir, "convert_meta.txt"), "w") as f:
        f.write(f"relu_test_acc={relu_te_acc:.6f}\n")
        f.write(f"ttfs_test_acc_n{args.n_eval}={ttfs_acc:.6f}\n")
        f.write(f"target_cross_time={target_t}\n")
        f.write(f"T={T}\n")
        f.write(f"scales={scales}\n")
    print(f"\nSaved to {save_dir}/  (threshold={args.threshold})")
    print(f"  relu_acc={relu_te_acc:.4f}, ttfs_acc({args.n_eval})={ttfs_acc:.4f}")


if __name__ == "__main__":
    main()
