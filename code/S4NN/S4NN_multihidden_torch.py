"""S4NN multi-hidden TTFS training, PyTorch GPU port.

Direct port of S4NN_multihidden_np.py with batch processing on GPU. The
forward, target-time, and backward rules are identical to the numpy
implementation; only the loop is vectorised across a batch and lifted to
torch tensors so RTX 4090 wall-time per epoch drops from ~22 min (numpy)
to ~minutes.

Saved weight format matches utils/mnist_net.py expectations:
  models/{T}_784_{n_h1}_..._{n_L}/weights_{layer}.npy
  - weights_0:  (n_h1, 28, 28)
  - weights_k for k≥1: (n_{k+1}, n_k, 1)

Usage
-----
  python S4NN_multihidden_torch.py --hidden 100 100 --tmax 4 --epochs 30
  python S4NN_multihidden_torch.py --hidden 200 200 --tmax 4 --epochs 30
  python S4NN_multihidden_torch.py --hidden 100 100 100 --tmax 4 --epochs 40
"""
from __future__ import annotations
import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
from mnist import MNIST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[100, 100])
    p.add_argument("--tmax", type=int, default=4, help="num_steps - 1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=100.0)
    p.add_argument("--lr-input", type=float, default=0.2)
    p.add_argument("--lr-hidden", type=float, default=0.2)
    p.add_argument("--lamda", type=float, default=1e-6)
    p.add_argument("--gamma", type=int, default=3,
                   help="Margin between target winner and others (timesteps).")
    p.add_argument("--init-low", type=float, default=0.0)
    p.add_argument("--init-high-input", type=float, default=5.0)
    p.add_argument("--init-high-hidden", type=float, default=50.0,
                   help="Hidden->hidden / hidden->output init upper bound.")
    p.add_argument("--test-n", type=int, default=2000)
    code_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    p.add_argument("--mnist-root", type=str,
                   default=osp.join(code_dir, "data/mnist/MNIST/raw/"))
    p.add_argument("--save-root", type=str,
                   default=osp.join(code_dir, "models/"))
    return p.parse_args()


# ============================================================================
# Data
# ============================================================================
def load_mnist_ttfs(root: str, tmax: int):
    md = MNIST(root)
    Itr, Ltr = md.load_training()
    Ite, Lte = md.load_testing()
    Itr = np.array(Itr, dtype=np.int32)
    Ite = np.array(Ite, dtype=np.int32)
    # TTFS encoding: bright pixel -> early spike. integer in [0, tmax].
    s_tr = np.floor((255 - Itr.reshape(-1, 28, 28)) * tmax / 255).astype(np.int64)
    s_te = np.floor((255 - Ite.reshape(-1, 28, 28)) * tmax / 255).astype(np.int64)
    return s_tr, np.array(Ltr, dtype=np.int64), s_te, np.array(Lte, dtype=np.int64)


# ============================================================================
# Forward (batched)
# ============================================================================
def build_one_hot_input(spike_times: torch.Tensor, T: int) -> torch.Tensor:
    """spike_times: (B, 28, 28) ∈ [0, tmax]. Returns (B, 28, 28, T+1)."""
    B = spike_times.shape[0]
    out = torch.zeros(B, 28, 28, T + 1, device=spike_times.device, dtype=torch.float32)
    out.scatter_(3, spike_times.unsqueeze(-1), 1.0)
    return out


def forward_layer(SpikeIn: torch.Tensor, W: torch.Tensor, threshold: float,
                  tmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One TTFS layer.

    SpikeIn shape:
      - input layer: (B, H, W, T+1)
      - hidden  layer: (B, n_pre, 1, T+1)
    W shape:
      - input layer: (n_post, H, W)
      - hidden layer: (n_post, n_pre, 1)
    Returns (firing_time (B, n_post) int64, spike_one_hot (B, n_post, 1, T+1)).
    """
    if W.dim() == 3 and SpikeIn.dim() == 4 and SpikeIn.shape[1] == W.shape[1] and SpikeIn.shape[2] == W.shape[2]:
        # Input layer: tensordot over (H, W)
        # voltage[b, h, t] = Σ_{x,y} W[h,x,y] * SpikeIn[b,x,y,t]
        pre = torch.einsum('hxy,bxyt->bht', W, SpikeIn)
    else:
        # Hidden layer: SpikeIn (B, n_pre, 1, T+1), W (n_post, n_pre, 1)
        # Collapse the singleton axis
        pre = torch.einsum('opd,bpdt->bot', W, SpikeIn)
    voltage = torch.cumsum(pre, dim=2)  # (B, n_post, T+1)
    voltage[:, :, tmax] = threshold + 1.0  # force last-step crossing
    fired = voltage > threshold
    ft = fired.float().argmax(dim=2) + 1  # (B, n_post)
    ft = torch.clamp(ft, max=tmax).long()
    # one-hot output spikes (B, n_post, 1, T+1)
    one_hot = torch.zeros(ft.shape[0], ft.shape[1], 1, voltage.shape[2],
                          device=voltage.device, dtype=torch.float32)
    one_hot.scatter_(3, ft.view(*ft.shape, 1, 1), 1.0)
    return ft, one_hot


def full_forward(SpikeImage: torch.Tensor, Ws: list[torch.Tensor],
                 thresholds: list[float], tmax: int):
    """Run all layers. Returns list of firingTimes per layer and the input
    spike representations passed to each layer (needed for weight update)."""
    fts: list[torch.Tensor] = []
    SpikeList = [SpikeImage]
    cur = SpikeImage
    for L, W in enumerate(Ws):
        ft, sp = forward_layer(cur, W, thresholds[L], tmax)
        fts.append(ft)
        cur = sp
        SpikeList.append(sp)
    return fts, SpikeList


# ============================================================================
# Target spike-time computation (S4NN rule, batched)
# ============================================================================
def compute_targets(ft_out: torch.Tensor, labels: torch.Tensor,
                    tmax: int, gamma: int) -> torch.Tensor:
    """ft_out: (B, n_out). labels: (B,). Returns target (B, n_out) int64."""
    B, n_out = ft_out.shape
    minFiring_long, _ = ft_out.min(dim=1, keepdim=True)  # (B, 1) long
    minFiring = minFiring_long.float()
    target = ft_out.float().clone()
    is_dead = (minFiring_long.squeeze(1) == tmax)  # (B,)
    # Push non-target neurons farther from min if within gamma
    too_close = (ft_out.float() - minFiring) < gamma
    pushed = torch.clamp(minFiring + gamma, max=float(tmax)).expand_as(target)
    target = torch.where(too_close, pushed, target)
    # Target neuron -> minFiring
    label_idx = labels.unsqueeze(1)  # (B, 1)
    target.scatter_(1, label_idx, minFiring)
    # Dead-network branch
    dead_target = torch.full_like(target, float(tmax))
    dead_winner = torch.full_like(label_idx, float(tmax - gamma), dtype=target.dtype)
    dead_target.scatter_(1, label_idx, dead_winner)
    target = torch.where(is_dead.unsqueeze(1), dead_target, target)
    return target.long()


# ============================================================================
# Backward + weight update (batched, S4NN rule)
# ============================================================================
def backward_step(
    fts: list[torch.Tensor],          # per-layer firing times, each (B, n_L)
    SpikeList: list[torch.Tensor],    # input spikes to each layer
    Ws: list[torch.Tensor],           # weights
    target_out: torch.Tensor,         # (B, n_out)
    tmax: int,
    lrs: list[float],
    lamda: float,
):
    """Performs one S4NN-style update step over a batch.

    Returns new Ws (in-place style mutation, but we accumulate dW for
    determinism and do W -= dW.mean(over batch) at the end).
    """
    Nlayers = len(Ws)
    B = fts[0].shape[0]

    # delta at output layer
    delta_out = (fts[-1].float() - target_out.float()) / tmax  # (B, n_out)
    deltas: list[torch.Tensor] = [None] * Nlayers  # type: ignore
    deltas[-1] = delta_out

    # Backprop deltas to earlier layers
    for L in range(Nlayers - 2, -1, -1):
        # deltas[L+1]: (B, n_{L+1})
        # Ws[L+1]: (n_{L+1}, n_L, 1)
        W_eff = Ws[L + 1].squeeze(-1)  # (n_{L+1}, n_L)
        ft_post = fts[L + 1]            # (B, n_{L+1})
        ft_pre = fts[L]                # (B, n_L)
        # Contribution mask: (B, n_{L+1}, n_L), True if ft_pre fired strictly before ft_post
        contrib = (ft_pre.unsqueeze(1) < ft_post.unsqueeze(2)).float()
        # delta_L[b, h_pre] = Σ_{h_post} delta_{L+1}[b, h_post] * W_eff[h_post, h_pre] * contrib[b, h_post, h_pre]
        deltas[L] = torch.einsum('bP,Pp,bPp->bp', deltas[L + 1], W_eff, contrib)

    # Weight updates per layer
    for L in range(Nlayers):
        # ft_post: (B, n_post). pre fires at SpikeList[L] (B, ..., T+1).
        ft_post = fts[L]                # (B, n_post)
        # We need "did pre fire before ft_post" — equivalent to spike indicator
        # being 1 at any t < ft_post. In one-hot rep, this is t < ft_post
        # check directly on stored spike times of pre layer.
        pre_spike_times: torch.Tensor
        if L == 0:
            # Input layer: pre is SpikeList[0] (B, H, W, T+1) one-hot at image[b,x,y]
            spk = SpikeList[L]
            # Recover spike time per (b, x, y)
            pre_spike_times = spk.argmax(dim=-1)  # (B, H, W)
            # contribution[b, h_post, x, y] = 1 if pre_spike_times[b,x,y] < ft_post[b,h_post]
            contrib_4d = (pre_spike_times.unsqueeze(1) < ft_post.unsqueeze(-1).unsqueeze(-1)).float()
            # delta accumulation: dW[h_post, x, y] -= lr * delta[b, h_post] * contrib[b, h_post, x, y]
            dW = -lrs[L] * torch.einsum('bH,bHxy->Hxy', deltas[L], contrib_4d) / B
            # L1 regularization
            dW = dW - lrs[L] * lamda * torch.sign(Ws[L])
            Ws[L] = Ws[L] + dW
        else:
            spk = SpikeList[L]            # (B, n_pre, 1, T+1)
            pre_spike_times = spk.squeeze(2).argmax(dim=-1)  # (B, n_pre)
            contrib_3d = (pre_spike_times.unsqueeze(1) < ft_post.unsqueeze(2)).float()
            # dW[h_post, h_pre, 1] -= lr * delta[b, h_post] * contrib[b, h_post, h_pre]
            dW2 = -lrs[L] * torch.einsum('bP,bPp->Pp', deltas[L], contrib_3d) / B
            dW2 = dW2.unsqueeze(-1)  # (n_post, n_pre, 1)
            dW2 = dW2 - lrs[L] * lamda * torch.sign(Ws[L])
            Ws[L] = Ws[L] + dW2
    return Ws


# ============================================================================
# Driver
# ============================================================================
def main():
    args = parse_args()
    T = args.tmax + 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    s_tr, y_tr, s_te, y_te = load_mnist_ttfs(args.mnist_root, args.tmax)
    print(f"Train spikes {s_tr.shape}, Test spikes {s_te.shape}")
    s_tr_t = torch.from_numpy(s_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    s_te_t = torch.from_numpy(s_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    # Architecture
    Nnrn = list(args.hidden) + [10]
    Nlayers = len(Nnrn)
    thresholds = [args.threshold] * Nlayers
    lrs = [args.lr_input] + [args.lr_hidden] * (Nlayers - 1)

    # Initialize weights — match S4NN paper: input layer high=5, others high=50
    Ws: list[torch.Tensor] = []
    rng = np.random.RandomState(args.seed)
    # Layer 0: (n_h1, 28, 28)
    W0 = (args.init_high_input - args.init_low) * rng.random_sample((Nnrn[0], 28, 28)) + args.init_low
    Ws.append(torch.from_numpy(W0).float().to(device))
    for L in range(1, Nlayers):
        Wk = (args.init_high_hidden - args.init_low) * rng.random_sample((Nnrn[L], Nnrn[L - 1], 1)) + args.init_low
        Ws.append(torch.from_numpy(Wk).float().to(device))

    print(f"Architecture: 784 -> {' -> '.join(map(str, Nnrn))}, T={T}, batch={args.batch}")
    save_dir = osp.join(args.save_root, f"{T}_784_{'_'.join(map(str, Nnrn))}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    best_te = 0.0
    n_train = s_tr.shape[0]

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(n_train, device=device)
        cor, tot = 0, 0
        for s in range(0, n_train, args.batch):
            e = min(s + args.batch, n_train)
            idx = perm[s:e]
            spk_in = s_tr_t[idx]                     # (B, 28, 28)
            labels = y_tr_t[idx]
            SpikeImage = build_one_hot_input(spk_in, T)
            fts, SpikeList = full_forward(SpikeImage, Ws, thresholds, args.tmax)
            preds = fts[-1].argmin(dim=1)
            cor += (preds == labels).sum().item()
            tot += labels.size(0)
            target_out = compute_targets(fts[-1], labels, args.tmax, args.gamma)
            Ws = backward_step(fts, SpikeList, Ws, target_out, args.tmax, lrs, args.lamda)
        train_acc = cor / tot

        # Test
        n_te = min(args.test_n, s_te.shape[0])
        cor_te, tot_te = 0, 0
        with torch.no_grad():
            for s in range(0, n_te, args.batch):
                e = min(s + args.batch, n_te)
                spk_in = s_te_t[s:e]
                SpikeImage = build_one_hot_input(spk_in, T)
                fts, _ = full_forward(SpikeImage, Ws, thresholds, args.tmax)
                preds = fts[-1].argmin(dim=1)
                cor_te += (preds == y_te_t[s:e]).sum().item()
                tot_te += (e - s)
        te_acc = cor_te / tot_te
        dt = time.time() - t0
        print(f"epoch {epoch:02d}: train={train_acc:.4f}  test={te_acc:.4f}  ({dt:.1f}s)")

        if te_acc > best_te:
            best_te = te_acc
            for L in range(Nlayers):
                np.save(osp.join(save_dir, f"weights_{L}.npy"),
                        Ws[L].detach().cpu().numpy().astype(np.float64))
            with open(osp.join(save_dir, "threshold.txt"), "w") as f:
                f.write(f"{args.threshold}\n")
            print(f"  ↳ best test={best_te:.4f}, saved")

    print(f"\nDone. Best test acc: {best_te:.4f}")


if __name__ == "__main__":
    main()
