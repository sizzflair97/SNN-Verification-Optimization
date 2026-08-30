"""Validate LIF BC-IBP on a trained Goeltz MNIST checkpoint.

The script runs one sound root certificate and, for budget one, exhaustively
evaluates every single-input +/- shift.  It also checks that every exact
hidden/output spike lies inside the reported BC-IBP intervals.
"""
from __future__ import annotations

import argparse
import json
import time

from mnist import MNIST
import numpy as np
import torch

from goltz_lif_ttfs import GoeltzLIFNetwork
from lif_bnb_ibp import LIFBCIBPResult, lif_bcibp_prove_robust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mnist-root", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--shift-step", type=float, default=None)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--oracle-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    hidden = list(config["hidden"])
    dims = [784] + hidden + [10]
    model = GoeltzLIFNetwork(
        dims,
        tau_syn=float(config["tau_syn"]),
        threshold=float(config["threshold"]),
        no_spike_time=max(
            10.0 * float(config["tau_syn"]),
            float(config["late"]) + 6.0 * float(config["tau_syn"]),
        ),
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    weights = [
        state["model"][f"layers.{index}.weight"].detach().cpu().numpy().astype(np.float64)
        for index in range(len(dims) - 1)
    ]
    return state, config, model, weights


def load_test_times(root: str, early: float, late: float) -> tuple[np.ndarray, np.ndarray]:
    images, labels = MNIST(root).load_testing()
    images_np = np.asarray(images, dtype=np.float32) / 255.0
    times = early + (1.0 - images_np) * (late - early)
    return times.astype(np.float64), np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def forward_model(model: GoeltzLIFNetwork, batch: np.ndarray, device: torch.device):
    results = model(torch.as_tensor(batch, dtype=torch.float32, device=device))
    output = results[-1]
    prediction = output.times.argmin(dim=1)
    return results, prediction


def exact_budget_one_inputs(
    baseline: np.ndarray,
    shift_step: float,
    input_min: float,
    input_max: float,
):
    yield np.array(baseline, copy=True), (None, 0)
    for index in range(baseline.size):
        for direction in (-1, 1):
            changed = np.array(baseline, copy=True)
            changed[index] = np.clip(
                changed[index] + direction * shift_step, input_min, input_max
            )
            if changed[index] != baseline[index]:
                yield changed, (index, direction)


def check_layer_containment(result, bound, tolerance: float = 2e-5) -> int:
    times = result.times.detach().cpu().numpy()
    spiked = result.spiked.detach().cpu().numpy()
    violations = 0
    for row in range(times.shape[0]):
        finite = spiked[row]
        violations += int(np.any(times[row, finite] + tolerance < bound.earliest[finite]))
        violations += int(np.any(times[row, finite] > bound.possible_latest[finite] + tolerance))
        guaranteed = np.isfinite(bound.latest)
        violations += int(np.any(times[row, finite & guaranteed] > bound.latest[finite & guaranteed] + tolerance))
        violations += int(np.any((~finite) & guaranteed))
    return violations


def main() -> None:
    args = parse_args()
    if args.budget != 1:
        raise ValueError("the complete parent-model oracle currently supports --budget 1")
    device = torch.device(args.device)
    state, config, model, weights = load_checkpoint(args.checkpoint, device)
    early = float(config["early"])
    late = float(config["late"])
    tau = float(config["tau_syn"])
    threshold = float(config["threshold"])
    shift_step = args.shift_step
    if shift_step is None:
        shift_step = (late - early) / 4.0  # one normalized T=5 input step

    test_times, labels = load_test_times(args.mnist_root, early, late)
    sample = args.sample
    while sample < len(labels):
        _, pred = forward_model(model, test_times[sample:sample + 1], device)
        if int(pred.item()) == int(labels[sample]):
            break
        sample += 1
    if sample >= len(labels):
        raise RuntimeError("no correctly classified sample found")
    baseline = test_times[sample]
    baseline_prediction = int(pred.item())

    started = time.perf_counter()
    certificate = lif_bcibp_prove_robust(
        baseline,
        weights,
        budget=args.budget,
        shift_step=shift_step,
        input_min=early,
        input_max=late,
        tau=tau,
        threshold=threshold,
        no_spike_time=max(10.0 * tau, late + 6.0 * tau),
        time_step=args.time_step,
        prediction=baseline_prediction,
    )
    bc_seconds = time.perf_counter() - started

    candidates = exact_budget_one_inputs(baseline, shift_step, early, late)
    prediction_set: set[int] = set()
    adversarial = None
    bound_violations = 0
    candidate_count = 0
    batch_inputs: list[np.ndarray] = []
    batch_meta: list[tuple[int | None, int]] = []
    oracle_started = time.perf_counter()

    def flush() -> None:
        nonlocal adversarial, bound_violations, candidate_count
        if not batch_inputs:
            return
        results, predictions = forward_model(model, np.asarray(batch_inputs), device)
        preds = predictions.detach().cpu().numpy()
        prediction_set.update(int(value) for value in preds)
        bound_violations += check_layer_containment(results[0], certificate.hidden)
        bound_violations += check_layer_containment(results[1], certificate.output)
        if adversarial is None:
            for meta, value in zip(batch_meta, preds):
                if int(value) != baseline_prediction:
                    adversarial = {"input": meta[0], "direction": meta[1], "prediction": int(value)}
                    break
        candidate_count += len(batch_inputs)
        batch_inputs.clear()
        batch_meta.clear()

    for perturbed, meta in candidates:
        batch_inputs.append(perturbed)
        batch_meta.append(meta)
        if len(batch_inputs) >= args.oracle_batch:
            flush()
    flush()
    oracle_seconds = time.perf_counter() - oracle_started

    if certificate.robust and prediction_set != {baseline_prediction}:
        raise AssertionError("unsound certificate: exhaustive oracle found another prediction")
    if bound_violations:
        raise AssertionError(f"exact trajectories escaped BC-IBP intervals: {bound_violations}")

    report = {
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_accuracy": float(state["test_accuracy"]),
        "sample": int(sample),
        "label": int(labels[sample]),
        "baseline_prediction": baseline_prediction,
        "budget": args.budget,
        "shift_step": shift_step,
        "time_step": args.time_step,
        "bcibp_robust": certificate.robust,
        "bcibp_reason": certificate.reason,
        "bcibp_seconds": bc_seconds,
        "oracle_candidates": candidate_count,
        "oracle_predictions": sorted(prediction_set),
        "oracle_adversarial": adversarial,
        "oracle_seconds": oracle_seconds,
        "bound_violations": bound_violations,
        "hidden_guaranteed_spike_ratio": float(np.isfinite(certificate.hidden.latest).mean()),
        "output_guaranteed_spike_ratio": float(np.isfinite(certificate.output.latest).mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
