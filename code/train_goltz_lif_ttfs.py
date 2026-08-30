"""Train a continuous-time LIF-TTFS MNIST network with exact spike gradients.

The core forward equation and gradients follow Goeltz et al. (Nature Machine
Intelligence, 2021), using the analytically tractable ``tau_m == tau_s`` case.

Example smoke run
-----------------
python train_goltz_lif_ttfs.py --hidden 32 --epochs 2 --train-n 512 --test-n 256

Paper-scale starting point
--------------------------
python train_goltz_lif_ttfs.py --hidden 350 --epochs 150 --batch 80

Checkpoints are saved under ``models/lif_goltz_ct_*`` and intentionally do
not use the cumulative-IF verifier's model-directory convention.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset

from goltz_lif_ttfs import GoeltzLIFNetwork, SpikeTimeLayerResult


@dataclass
class TrainConfig:
    hidden: list[int]
    epochs: int
    batch: int
    lr: float
    seed: int
    tau_syn: float
    threshold: float
    early: float
    late: float
    input_noise: float
    xi: float
    early_reg_alpha: float
    early_reg_beta: float
    max_update: float
    weight_bump: float
    max_missing: list[float]
    train_n: int
    test_n: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[350])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau-syn", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--early", type=float, default=0.15)
    p.add_argument("--late", type=float, default=2.0)
    p.add_argument("--input-noise", type=float, default=0.3)
    p.add_argument("--xi", type=float, default=0.2,
                   help="Temperature in cross-entropy over negative spike times.")
    p.add_argument("--early-reg-alpha", type=float, default=0.005)
    p.add_argument("--early-reg-beta", type=float, default=1.0)
    p.add_argument("--max-update", type=float, default=0.2)
    p.add_argument("--weight-bump", type=float, default=0.005)
    p.add_argument("--max-missing", type=float, nargs="+", default=[0.15, 0.05],
                   help="Maximum silent ratio per layer; extend last value for deeper nets.")
    p.add_argument("--train-n", type=int, default=0, help="0 uses all training samples.")
    p.add_argument("--test-n", type=int, default=0, help="0 uses all test samples.")
    code_dir = osp.dirname(osp.abspath(__file__))
    p.add_argument("--mnist-root", default=osp.join(code_dir, "data/mnist/MNIST/raw/"))
    p.add_argument("--save-root", default=osp.join(code_dir, "models"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_mnist_spike_times(
    root: str,
    early: float,
    late: float,
    train_n: int = 0,
    test_n: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    md = MNIST(root)
    x_train, y_train = md.load_training()
    x_test, y_test = md.load_testing()
    x_train = np.asarray(x_train, dtype=np.float32) / 255.0
    x_test = np.asarray(x_test, dtype=np.float32) / 255.0
    # Bright pixels carry stronger evidence and therefore spike earlier.
    t_train = early + (1.0 - x_train) * (late - early)
    t_test = early + (1.0 - x_test) * (late - early)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    if train_n > 0:
        t_train, y_train = t_train[:train_n], y_train[:train_n]
    if test_n > 0:
        t_test, y_test = t_test[:test_n], y_test[:test_n]
    return t_train, y_train, t_test, y_test


def spike_time_loss(
    output: SpikeTimeLayerResult,
    labels: torch.Tensor,
    *,
    tau_syn: float,
    xi: float,
    early_reg_alpha: float,
    early_reg_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = -output.times / (xi * tau_syn)
    ce = F.cross_entropy(logits, labels)
    correct_t = output.times.gather(1, labels[:, None]).squeeze(1)
    correct_spiked = output.spiked.gather(1, labels[:, None]).squeeze(1)
    # Eq. (6) plus the early-spike regularizer described in Methods.  Missing
    # spikes have no analytical gradient and are handled by weight bumps.
    early_term = torch.expm1(
        (correct_t / (early_reg_beta * tau_syn)).clamp(max=12.0)
    )
    early_term = torch.where(correct_spiked, early_term, torch.zeros_like(early_term))
    reg = early_reg_alpha * early_term.mean()
    return ce + reg, ce.detach(), reg.detach()


@torch.no_grad()
def apply_silent_neuron_bump(
    model: GoeltzLIFNetwork,
    results: list[SpikeTimeLayerResult],
    max_missing: list[float],
    base_bump: float,
    consecutive_failures: list[int],
) -> int | None:
    """Apply the paper's local heuristic to the first overly silent layer."""
    for layer_index, (layer, result) in enumerate(zip(model.layers, results)):
        allowed = max_missing[min(layer_index, len(max_missing) - 1)]
        missing_by_neuron = (~result.spiked).float().mean(dim=0)
        if float(missing_by_neuron.mean()) <= allowed:
            consecutive_failures[layer_index] = 0
            continue
        consecutive_failures[layer_index] += 1
        silent = missing_by_neuron > allowed
        if silent.any():
            multiplier = 2.0 ** min(consecutive_failures[layer_index] - 1, 6)
            layer.weight[silent] += base_bump * multiplier
        return layer_index
    return None


@torch.no_grad()
def bounded_optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    max_update: float,
) -> int:
    """Run Adam and reject coordinate updates larger than the paper's limit."""
    before = [p.detach().clone() for p in parameters]
    optimizer.step()
    rejected = 0
    for p, old in zip(parameters, before):
        too_large = (p - old).abs() > max_update
        rejected += int(too_large.sum())
        p[too_large] = old[too_large]
    return rejected


@torch.no_grad()
def evaluate(model: GoeltzLIFNetwork, loader: DataLoader, device: torch.device) -> tuple[float, list[float]]:
    model.eval()
    correct = total = 0
    missing_sum = [0.0 for _ in model.layers]
    for times, labels in loader:
        times, labels = times.to(device), labels.to(device)
        results = model(times)
        output = results[-1]
        # Match the Fast&Deep decoder: classify solely by the earliest actual
        # output spike. A sample with no output spike is rejected (-1), not
        # assigned to class zero through argmin's tie-breaking.
        masked_times = output.times.masked_fill(~output.spiked, float("inf"))
        prediction = masked_times.argmin(dim=1)
        prediction = prediction.masked_fill(~output.spiked.any(dim=1), -1)
        correct += int((prediction == labels).sum())
        total += labels.numel()
        for index, result in enumerate(results):
            missing_sum[index] += float((~result.spiked).sum())
    denominators = [total * layer.out_features for layer in model.layers]
    return correct / max(total, 1), [value / denom for value, denom in zip(missing_sum, denominators)]


def main() -> None:
    args = parse_args()
    if args.early >= args.late:
        raise ValueError("--early must be smaller than --late")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    config = TrainConfig(
        hidden=list(args.hidden), epochs=args.epochs, batch=args.batch, lr=args.lr,
        seed=args.seed, tau_syn=args.tau_syn, threshold=args.threshold,
        early=args.early, late=args.late, input_noise=args.input_noise, xi=args.xi,
        early_reg_alpha=args.early_reg_alpha, early_reg_beta=args.early_reg_beta,
        max_update=args.max_update, weight_bump=args.weight_bump,
        max_missing=list(args.max_missing), train_n=args.train_n, test_n=args.test_n,
    )

    train_t, train_y, test_t, test_y = load_mnist_spike_times(
        args.mnist_root, args.early, args.late, args.train_n, args.test_n
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_t), torch.from_numpy(train_y)),
        batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_t), torch.from_numpy(test_y)),
        batch_size=max(args.batch, 128), shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    dims = [784] + list(args.hidden) + [10]
    model = GoeltzLIFNetwork(
        dims, tau_syn=args.tau_syn, threshold=args.threshold,
        no_spike_time=max(10.0 * args.tau_syn, args.late + 6.0 * args.tau_syn),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
    params = list(model.parameters())
    consecutive_failures = [0] * len(model.layers)

    arch = "_".join(str(value) for value in dims)
    run_name = f"lif_goltz_ct_{arch}_seed{args.seed}"
    save_dir = osp.join(args.save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print(f"device={device} architecture={' -> '.join(map(str, dims))}")
    print(f"train={len(train_t)} test={len(test_t)} save={save_dir}")
    best_accuracy = -1.0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_total = ce_total = reg_total = 0.0
        sample_total = rejected_total = 0
        bumps = [0] * len(model.layers)
        for clean_times, labels in train_loader:
            clean_times = clean_times.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.input_noise > 0:
                input_times = clean_times + torch.randn_like(clean_times) * args.input_noise
            else:
                input_times = clean_times

            optimizer.zero_grad(set_to_none=True)
            results = model(input_times)
            loss, ce, reg = spike_time_loss(
                results[-1], labels, tau_syn=args.tau_syn, xi=args.xi,
                early_reg_alpha=args.early_reg_alpha,
                early_reg_beta=args.early_reg_beta,
            )
            if loss.requires_grad:
                loss.backward()
                rejected_total += bounded_optimizer_step(optimizer, params, args.max_update)
            bumped = apply_silent_neuron_bump(
                model, results, list(args.max_missing), args.weight_bump, consecutive_failures
            )
            if bumped is not None:
                bumps[bumped] += 1

            n = labels.numel()
            loss_total += float(loss.detach()) * n
            ce_total += float(ce) * n
            reg_total += float(reg) * n
            sample_total += n

        scheduler.step()
        accuracy, missing = evaluate(model, test_loader, device)
        elapsed = time.time() - started
        print(
            f"epoch={epoch + 1:03d} loss={loss_total/sample_total:.5f} "
            f"ce={ce_total/sample_total:.5f} reg={reg_total/sample_total:.5f} "
            f"test_acc={accuracy:.4f} missing={','.join(f'{x:.3f}' for x in missing)} "
            f"bumps={bumps} rejected={rejected_total} elapsed={elapsed:.1f}s"
        )
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1, "test_accuracy": accuracy, "config": asdict(config),
        }
        torch.save(state, osp.join(save_dir, "last.pt"))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(state, osp.join(save_dir, "best.pt"))

    for index, layer in enumerate(model.layers):
        np.save(osp.join(save_dir, f"weights_{index}.npy"), layer.weight.detach().cpu().numpy())
    best_state = torch.load(osp.join(save_dir, "best.pt"), map_location="cpu", weights_only=False)
    for index in range(len(model.layers)):
        best_weight = best_state["model"][f"layers.{index}.weight"].numpy()
        np.save(osp.join(save_dir, f"weights_best_{index}.npy"), best_weight)
    with open(osp.join(save_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump({"best_test_accuracy": best_accuracy}, handle, indent=2)
    print(f"best_test_accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
