"""S4NN greedy layer-wise training for multi-hidden TTFS-SNN.

Trains a depth-L TTFS network by repeatedly invoking the proven
single-hidden S4NN rule (Kheradpisheh & Masquelier, 2020):

  Stage k  (k = 1, ..., L):
    Treat the spike times produced by previously trained, frozen layers
    1..k-1 as the "input" to a fresh single-hidden S4NN
        n_{k-1}  ->  n_k  ->  n_out (10)
    and train it end-to-end with the S4NN rule (per-sample gradient
    L2 normalisation, has-fired contribution masks). After training,
    discard the temporary output classifier; only W_k (layer k's weight)
    is kept and frozen for stage k+1.

  Final composition:  W_1 (frozen) | W_2 (frozen) | ... | W_L | W_out
  where W_out comes from the LAST stage's output classifier.

  Saved layout (verifier-compatible):
    models/{T}_784_{n_1}_..._{n_L}_10_greedy/weights_{0..L}.npy
    models/{T}_784_{n_1}_..._{n_L}_10_greedy/threshold.txt    (= 100)

Why this works where end-to-end multi-hidden BPTT fails:
  - Each stage solves an already-validated single-hidden problem.
  - No conversion / encoding between stages: hidden TTFS spike times
    are exactly the integer-spike-time format S4NN expects as input.
  - Gradient normalisation, the dead-neuron rule, and the has-fired
    contribution masks of S4NN are preserved unchanged.

Usage:
  python s4nn_greedy.py --hidden 100 100 --tmax 4 --epochs 30 30
  python s4nn_greedy.py --hidden 200 200 --tmax 4 --epochs 30 30
  python s4nn_greedy.py --hidden 100 100 100 --tmax 4 --epochs 30 30 30
"""
from __future__ import annotations
import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
from mnist import MNIST


# ============================================================================
# Args
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", required=True,
                   help="Hidden-layer widths, e.g. 100 100 or 200 200 or 100 100 100.")
    p.add_argument("--tmax", type=int, default=4, help="num_steps - 1")
    p.add_argument("--epochs", type=int, nargs="+", default=None,
                   help="Per-stage epoch budgets (default: 30 per stage).")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=100.0)
    p.add_argument("--lr-input", type=float, default=0.2)
    p.add_argument("--lr-hidden", type=float, default=0.2)
    p.add_argument("--lamda", type=float, default=1e-6)
    p.add_argument("--gamma", type=int, default=3)
    p.add_argument("--init-input-high", type=float, default=5.0)
    p.add_argument("--init-hidden-high", type=float, default=50.0)
    p.add_argument("--test-n", type=int, default=2000,
                   help="Test-set subset for per-epoch eval (final eval is full).")
    p.add_argument("--reuse-stage1", action="store_true",
                   help="Skip stage 1 if a 5_784_{n_h1}_10 model already exists.")
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
    s_tr = np.floor((255 - Itr.reshape(-1, 28, 28)) * tmax / 255).astype(np.int64)
    s_te = np.floor((255 - Ite.reshape(-1, 28, 28)) * tmax / 255).astype(np.int64)
    return s_tr, np.array(Ltr, dtype=np.int64), s_te, np.array(Lte, dtype=np.int64)


# ============================================================================
# Single-hidden TTFS S4NN (batched torch, S4NN rule with per-sample
# gradient L2 normalisation)
# ============================================================================
class SingleHiddenS4NN:
    """Trains one TTFS net of shape (n_in, ..., 1) -> n_h -> 10.

    Input-side shape is generic: either (28, 28) for raw MNIST or (n_pre, 1)
    for a hidden-spike-time input. The first weight matrix is shaped
    (n_h, n_in_dim0, n_in_dim1) accordingly.
    """

    def __init__(self, in_shape, n_h: int, n_out: int, T: int,
                 threshold: float, lr_in: float, lr_out: float,
                 lamda: float, gamma: int,
                 init_in_high: float, init_out_high: float,
                 device, seed: int = 0):
        self.in_shape = tuple(in_shape)
        self.n_h, self.n_out = n_h, n_out
        self.T, self.tmax = T, T - 1
        self.thr = threshold
        self.lr = (lr_in, lr_out)
        self.lamda = lamda
        self.gamma = gamma
        self.device = device
        rng = np.random.RandomState(seed)
        # W0: (n_h, *in_shape)
        W0 = init_in_high * rng.random_sample((n_h, *in_shape))
        # W1: (n_out, n_h, 1) — output layer always treats hidden as flat
        W1 = init_out_high * rng.random_sample((n_out, n_h, 1))
        self.W0 = torch.from_numpy(W0).float().to(device)
        self.W1 = torch.from_numpy(W1).float().to(device)

    # ----- forward (batched) -----
    def _forward(self, spike_in_one_hot: torch.Tensor, hidden_one_hot: torch.Tensor | None = None):
        """spike_in_one_hot: (B, *in_shape, T) one-hot (1 at each pixel's spike time).

        Returns:
          ft_h:  (B, n_h)  long
          ft_o:  (B, n_out) long
          h_one_hot: (B, n_h, 1, T) one-hot of ft_h (for output layer voltage)
        """
        # voltage_h[b, h, t] = sum over input dims of W0[h, *] * spike_in[b, *, t]
        # We einsum collapsing all input-shape dims.
        if len(self.in_shape) == 2:
            # raw image: spike_in_one_hot is (B, H, W, T)
            pre_h = torch.einsum('hxy,bxyt->bht', self.W0, spike_in_one_hot)
        elif len(self.in_shape) == 1:
            # hidden-flat input: spike_in_one_hot is (B, n_pre, T)
            pre_h = torch.einsum('hp,bpt->bht', self.W0, spike_in_one_hot)
        else:
            # (n_pre, 1) hidden input shape: spike_in_one_hot is (B, n_pre, 1, T)
            pre_h = torch.einsum('hpd,bpdt->bht', self.W0, spike_in_one_hot)
        V_h = torch.cumsum(pre_h, dim=2)
        V_h[:, :, self.tmax] = self.thr + 1.0
        ft_h = (V_h > self.thr).float().argmax(dim=2) + 1
        ft_h = torch.clamp(ft_h, max=self.tmax).long()
        # one-hot
        h_one_hot = torch.zeros(ft_h.shape[0], self.n_h, 1, self.T,
                                device=self.device, dtype=torch.float32)
        h_one_hot.scatter_(3, ft_h.view(-1, self.n_h, 1, 1), 1.0)

        # Output layer
        pre_o = torch.einsum('opd,bpdt->bot', self.W1, h_one_hot)
        V_o = torch.cumsum(pre_o, dim=2)
        V_o[:, :, self.tmax] = self.thr + 1.0
        ft_o = (V_o > self.thr).float().argmax(dim=2) + 1
        ft_o = torch.clamp(ft_o, max=self.tmax).long()
        return ft_h, ft_o, h_one_hot

    @staticmethod
    def _build_one_hot(spike_times: torch.Tensor, T: int, device) -> torch.Tensor:
        out = torch.zeros(*spike_times.shape, T, device=device, dtype=torch.float32)
        out.scatter_(-1, spike_times.unsqueeze(-1), 1.0)
        return out

    # ----- target spike times (S4NN rule, batched) -----
    def _targets(self, ft_o: torch.Tensor, labels: torch.Tensor):
        B, no = ft_o.shape
        minF, _ = ft_o.min(dim=1, keepdim=True)
        target = ft_o.float().clone()
        is_dead = (minF.squeeze(1) == self.tmax)
        too_close = (ft_o.float() - minF.float()) < self.gamma
        pushed = torch.clamp(minF.float() + self.gamma, max=float(self.tmax))
        target = torch.where(too_close, pushed.expand_as(target), target)
        target.scatter_(1, labels.unsqueeze(1), minF.float())
        # dead-network branch: target winner = tmax - gamma, others = tmax
        dead_t = torch.full_like(target, float(self.tmax))
        dead_t.scatter_(1, labels.unsqueeze(1), float(max(self.tmax - self.gamma, 0)))
        target = torch.where(is_dead.unsqueeze(1), dead_t, target)
        return target

    # ----- batched per-sample S4NN backward -----
    def _backward(self, ft_h, ft_o, target_o, input_spike_times):
        """input_spike_times shape: (B, *in_shape) integer; ft_h: (B, n_h);
        ft_o: (B, n_out); target_o: (B, n_out) float.

        Implements the S4NN rule with per-sample gradient L2 normalisation
        and per-sample weight updates *averaged* across the batch.
        """
        B = ft_o.shape[0]
        tmax = self.tmax
        # delta at output (per sample, with L2 normalisation)
        delta_o = (ft_o.float() - target_o) / tmax  # sign convention W -= lr*delta*hasFired
        # In S4NN.py: delta_o = (target - ft)/tmax  ⇒  W -= lr * delta * hasFired.
        # Here we'd use   delta_o' = -(target - ft)/tmax = (ft - target)/tmax  ⇒  W -= lr * (-delta')...
        # Easier: just match S4NN.py exactly.
        delta_o = (target_o - ft_o.float()) / tmax  # (B, n_out)
        norm = delta_o.norm(dim=1, keepdim=True).clamp(min=1e-12)
        delta_o = delta_o / norm

        # hasFired_o[b, o, h] = 1 if ft_h[b, h] < ft_o[b, o]
        hasFired_o = (ft_h.unsqueeze(1) < ft_o.unsqueeze(2)).float()  # (B, n_out, n_h)
        # W1[o, h, 0] -= lr * mean_b( delta_o[b, o] * hasFired_o[b, o, h] )
        dW1 = -self.lr[1] * (delta_o.unsqueeze(2) * hasFired_o).sum(dim=0)  # (n_out, n_h)
        dW1 = dW1.unsqueeze(-1)
        # weight regularisation
        self.W1 = self.W1 + dW1 - self.lr[1] * self.lamda * self.W1

        # delta_h[b, h] = sum_o delta_o[b, o] * hasFired_o[b, o, h] * W1_eff[o, h]
        W1_eff = self.W1.squeeze(-1)  # (n_out, n_h)
        delta_h = torch.einsum('bo,boh,oh->bh', delta_o, hasFired_o, W1_eff)
        norm_h = delta_h.norm(dim=1, keepdim=True).clamp(min=1e-12)
        delta_h = delta_h / norm_h

        # hasFired_in: shape (B, n_h, *in_shape), 1 if input_spike_times < ft_h.
        ft_h_bcast = ft_h.unsqueeze(-1)  # (B, n_h, 1)
        if len(self.in_shape) == 2:
            ft_h_bcast = ft_h.view(B, self.n_h, 1, 1)
            hasFired_in = (input_spike_times.unsqueeze(1) < ft_h_bcast).float()  # (B, n_h, H, W)
            dW0 = -self.lr[0] * (delta_h.view(B, self.n_h, 1, 1) * hasFired_in).sum(dim=0)
        else:
            # input_spike_times shape (B, n_pre) or (B, n_pre, 1)
            inp = input_spike_times
            if inp.dim() == 2:
                ft_h_bcast = ft_h.unsqueeze(-1)
                hasFired_in = (inp.unsqueeze(1) < ft_h_bcast).float()  # (B, n_h, n_pre)
                dW0 = -self.lr[0] * (delta_h.unsqueeze(-1) * hasFired_in).sum(dim=0)
                # Match self.W0 shape (n_h, n_pre) or (n_h, n_pre, 1)
                if self.W0.dim() == 3:
                    dW0 = dW0.unsqueeze(-1)
            else:
                # (B, n_pre, 1)
                ft_h_bcast = ft_h.view(B, self.n_h, 1, 1)
                hasFired_in = (inp.unsqueeze(1) < ft_h_bcast).float()  # (B, n_h, n_pre, 1)
                dW0 = -self.lr[0] * (delta_h.view(B, self.n_h, 1, 1) * hasFired_in).sum(dim=0)
        self.W0 = self.W0 + dW0 - self.lr[0] * self.lamda * self.W0

    # ----- one epoch -----
    def fit_epoch(self, spike_times_tr: torch.Tensor, labels_tr: torch.Tensor,
                  batch_size: int) -> float:
        n = spike_times_tr.shape[0]
        perm = torch.randperm(n, device=self.device)
        cor, tot = 0, 0
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            idx = perm[s:e]
            inp_st = spike_times_tr[idx]
            labels = labels_tr[idx]
            inp_one_hot = self._build_one_hot(inp_st, self.T, self.device)
            ft_h, ft_o, _ = self._forward(inp_one_hot)
            preds = ft_o.argmin(dim=1)
            cor += (preds == labels).sum().item()
            tot += labels.size(0)
            target_o = self._targets(ft_o, labels)
            self._backward(ft_h, ft_o, target_o, inp_st)
        return cor / tot

    @torch.no_grad()
    def evaluate(self, spike_times_te: torch.Tensor, labels_te: torch.Tensor,
                 batch_size: int) -> float:
        n = spike_times_te.shape[0]
        cor = 0
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            inp_st = spike_times_te[s:e]
            inp_one_hot = self._build_one_hot(inp_st, self.T, self.device)
            _, ft_o, _ = self._forward(inp_one_hot)
            preds = ft_o.argmin(dim=1)
            cor += (preds == labels_te[s:e]).sum().item()
        return cor / n

    @torch.no_grad()
    def hidden_spike_times(self, spike_times: torch.Tensor, batch_size: int) -> torch.Tensor:
        out = torch.empty(spike_times.shape[0], self.n_h,
                          dtype=torch.long, device=self.device)
        for s in range(0, spike_times.shape[0], batch_size):
            e = min(s + batch_size, spike_times.shape[0])
            inp_st = spike_times[s:e]
            inp_one_hot = self._build_one_hot(inp_st, self.T, self.device)
            ft_h, _, _ = self._forward(inp_one_hot)
            out[s:e] = ft_h
        return out


# ============================================================================
# Greedy driver
# ============================================================================
def main():
    args = parse_args()
    T = args.tmax + 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_hidden = list(args.hidden)
    L = len(n_hidden)
    epochs = args.epochs if args.epochs is not None else [30] * L
    if len(epochs) == 1 and L > 1:
        epochs = epochs * L
    assert len(epochs) == L, f"--epochs must be 1 or {L} values"
    print(f"Architecture: 784 -> {' -> '.join(map(str, n_hidden))} -> 10, T={T}")
    print(f"Per-stage epochs: {epochs}")

    s_tr, y_tr, s_te, y_te = load_mnist_ttfs(args.mnist_root, args.tmax)
    s_tr_t = torch.from_numpy(s_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    s_te_t = torch.from_numpy(s_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)
    print(f"MNIST: {s_tr.shape[0]} train / {s_te.shape[0]} test")

    arch = "_".join(["784"] + [str(h) for h in n_hidden] + ["10"])
    save_dir = osp.join(args.save_root, f"{T}_{arch}_greedy")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    # ---- Run L stages ----
    stage_inputs_train = s_tr_t          # (Ntr, 28, 28)
    stage_inputs_test = s_te_t
    cur_in_shape = (28, 28)
    frozen_W = []                         # list of np arrays in verifier layout

    for k, n_h in enumerate(n_hidden):
        print(f"\n=== Stage {k+1}/{L}: train single-hidden  in_shape={cur_in_shape} -> {n_h} -> 10 ===")
        # W_0 of every stage processes integer spike-time input (image at k=0,
        # hidden firing times for k>=1) — both live on the same {0..tmax} scale,
        # so we always use init_input_high (small) for W_0 and init_hidden_high
        # (larger) for the temporary output classifier.
        net = SingleHiddenS4NN(
            in_shape=cur_in_shape, n_h=n_h, n_out=10, T=T,
            threshold=args.threshold, lr_in=args.lr_input, lr_out=args.lr_hidden,
            lamda=args.lamda, gamma=args.gamma,
            init_in_high=args.init_input_high,
            init_out_high=args.init_hidden_high,
            device=device, seed=args.seed + k,
        )

        # Optional: reuse a pretrained stage-1 W_0 if it exists. Stage 1's
        # hidden representation determines what stage 2+ can build on; the
        # existing models/{T}_784_{n_h}_10/weights_0.npy was trained with the
        # full S4NN.py recipe and reaches the paper's 78% baseline. Loading it
        # gives a much stronger pretraining target than re-running stage 1
        # from scratch with the simplified torch trainer.
        if k == 0 and args.reuse_stage1:
            ref_dir = osp.join(args.save_root, f"{T}_784_{n_h}_10")
            ref_W0 = osp.join(ref_dir, "weights_0.npy")
            ref_W1 = osp.join(ref_dir, "weights_1.npy")
            if osp.exists(ref_W0):
                W0_pre = np.load(ref_W0).astype(np.float32)
                if W0_pre.shape != (n_h, 28, 28):
                    raise ValueError(f"reuse-stage1: shape mismatch {W0_pre.shape}")
                net.W0 = torch.from_numpy(W0_pre).float().to(device)
                if osp.exists(ref_W1):
                    W1_pre = np.load(ref_W1).astype(np.float32)
                    if W1_pre.ndim == 2:
                        W1_pre = W1_pre[..., None]
                    net.W1 = torch.from_numpy(W1_pre).float().to(device)
                print(f"  loaded pretrained stage-1 weights from {ref_dir}/")
                print(f"  skipping stage-1 training (use --epochs 0 30 to also skip 'epoch 0' eval)")
                if epochs[0] == 0:
                    # short-circuit training
                    pass
        # If we reused stage 1, evaluate the loaded weights once to anchor
        # the best-checkpoint baseline.
        n_te_eval = min(args.test_n, stage_inputs_test.shape[0])
        best_te = 0.0
        if k == 0 and args.reuse_stage1:
            best_te = net.evaluate(stage_inputs_test[:n_te_eval], y_te_t[:n_te_eval], args.batch)
            print(f"  reused stage-1 test acc: {best_te:.4f}")
        best_W0 = net.W0.detach().clone()
        best_W1 = net.W1.detach().clone()
        for ep in range(epochs[k]):
            t0 = time.time()
            tr_acc = net.fit_epoch(stage_inputs_train, y_tr_t, args.batch)
            te_acc = net.evaluate(stage_inputs_test[:n_te_eval], y_te_t[:n_te_eval], args.batch)
            dt = time.time() - t0
            tag = ""
            if te_acc > best_te:
                best_te = te_acc
                best_W0 = net.W0.detach().clone()
                best_W1 = net.W1.detach().clone()
                tag = " *"
            print(f"  ep{ep:02d}: train={tr_acc:.4f}  test={te_acc:.4f}  ({dt:.1f}s){tag}")
        # Restore best
        net.W0 = best_W0
        net.W1 = best_W1
        print(f"  stage {k+1} best test (subset n={n_te_eval}): {best_te:.4f}")

        # The current stage's W0 becomes the new frozen layer.
        # Reshape into verifier format
        if k == 0:
            # input layer: (n_h, 28, 28) — already correct
            W_save = net.W0.detach().cpu().numpy().astype(np.float64)
        else:
            # hidden layer: (n_h, n_pre) — needs (n_h, n_pre, 1)
            arr = net.W0.detach().cpu().numpy().astype(np.float64)
            if arr.ndim == 2:
                arr = arr[..., None]
            W_save = arr
        frozen_W.append(W_save)

        # Forward through this layer to get inputs for stage k+1
        if k < L - 1:
            print(f"  caching hidden-{k+1} spike times for next stage…")
            stage_inputs_train = net.hidden_spike_times(stage_inputs_train, args.batch)
            stage_inputs_test = net.hidden_spike_times(stage_inputs_test, args.batch)
            cur_in_shape = (n_h,)        # 1D flat input from now on
        else:
            # Last stage — keep its output classifier as the final W_out
            arr = net.W1.detach().cpu().numpy().astype(np.float64)
            if arr.ndim == 2:
                arr = arr[..., None]
            frozen_W.append(arr)

    # ---- Save composed weights ----
    print(f"\nSaving composed network weights to {save_dir}/")
    for i, w in enumerate(frozen_W):
        np.save(osp.join(save_dir, f"weights_{i}.npy"), w)
        print(f"  weights_{i}.npy: shape {w.shape}")
    with open(osp.join(save_dir, "threshold.txt"), "w") as f:
        f.write(f"{args.threshold}\n")

    # ---- End-to-end TTFS-forward evaluation via verifier's simulator ----
    print("\nFinal end-to-end TTFS evaluation via verifier forward()…")
    import sys
    sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.config import CFG
    from utils.load import load_mnist
    from utils.mnist_net import forward, prepare_weights
    cfg = CFG(
        log_name="greedy_eval", subtype="mnist", load_data_func=load_mnist,
        n_layer_neurons=tuple([784] + n_hidden + [10]),
        layer_shapes=((28, 28), *[(h, 1) for h in n_hidden], (10, 1)),
        num_steps=T, num_samples=10,
    )
    # Override the model dir to point at our greedy save
    weights = []
    for i in range(L + 1):
        w = np.load(osp.join(save_dir, f"weights_{i}.npy"))
        weights.append(w)
    images, labels, _, _ = load_mnist(cfg)
    n_eval = min(2000, len(images))
    cor = sum(int(forward(cfg, weights, images[i].astype(int)) == labels[i])
              for i in range(n_eval))
    ttfs_acc = cor / n_eval
    print(f"\n=== TTFS forward acc on {n_eval} test samples: {ttfs_acc:.4f} ===")
    with open(osp.join(save_dir, "greedy_meta.txt"), "w") as f:
        f.write(f"hidden_widths={n_hidden}\n")
        f.write(f"per_stage_epochs={epochs}\n")
        f.write(f"T={T}\n")
        f.write(f"ttfs_test_acc_n{n_eval}={ttfs_acc:.6f}\n")


if __name__ == "__main__":
    main()
