#!/usr/bin/env python3
"""Filter-free exhaustive Delta<=2 adversarial search on the MNIST TTFS model.

For each selected input, this script evaluates:

* every valid one-pixel shift of L1 cost 1;
* every valid one-pixel shift of L1 cost 2; and
* every pair of distinct pixels shifted by one timestep each.

Thus the search covers the complete discrete L1 ball for Delta <= 2.  The
first-hidden voltage deltas are cached per pixel option, while candidate
pairs are forwarded in GPU batches.  No active-set filter, IBP prune, or BnB
prune is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from utils.config import CFG
from utils.dictionary_mnist import threshold
from utils.load import load_mnist
from utils.mnist_net import forward, prepare_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--near-threshold-tol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        help="Merge completed shard JSON files instead of running the search.",
    )
    return parser.parse_args()


def make_cfg(n_samples: int, n_hidden: int, seed: int) -> CFG:
    return CFG(
        log_name="reviewer_incidence_exact_gpu",
        subtype="mnist",
        load_data_func=load_mnist,
        seed=seed,
        num_samples=n_samples,
        deltas=(1, 2),
        n_layer_neurons=(784, n_hidden, 10),
        layer_shapes=((28, 28), (n_hidden, 1), (10, 1)),
        num_steps=5,
    )


def select_examples(
    cfg: CFG,
    weights: list[np.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict], int]:
    rng = np.random.default_rng(cfg.seed)
    selected: list[dict] = []
    eligible = 0
    for idx in rng.permutation(len(images)):
        firing_times: list[np.ndarray] = []
        pred = int(forward(cfg, weights, images[int(idx)], firing_times))
        unique = int(np.sum(firing_times[-1] == np.min(firing_times[-1]))) == 1
        if pred == int(labels[int(idx)]) and unique:
            eligible += 1
            if len(selected) < cfg.num_samples:
                selected.append(
                    {
                        "selection_order": len(selected),
                        "index": int(idx),
                        "label": int(labels[int(idx)]),
                        "orig_pred": pred,
                    }
                )
    if len(selected) != cfg.num_samples:
        raise RuntimeError(
            f"Only selected {len(selected)} of {cfg.num_samples} requested examples"
        )
    return selected, eligible


class ExactDelta2Enumerator:
    def __init__(
        self,
        cfg: CFG,
        weights: list[np.ndarray],
        device: str,
        dtype: str,
        chunk_size: int,
        near_threshold_tol: float,
    ) -> None:
        if len(weights) != 2:
            raise ValueError("This audit expects one hidden layer and one output layer")
        self.cfg = cfg
        self.weights = weights
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.chunk_size = chunk_size
        self.near_threshold_tol = near_threshold_tol
        self.num_steps = cfg.num_steps
        self.t_grid = torch.arange(self.num_steps, device=self.device)

        w1 = weights[0].reshape(weights[0].shape[0], -1)
        w2 = weights[1]
        if w2.ndim == 3:
            w2 = w2[:, :, 0]
        self.w1 = torch.as_tensor(w1, dtype=self.dtype, device=self.device)
        self.w2 = torch.as_tensor(w2, dtype=self.dtype, device=self.device)

    def _forward_hidden_voltage(
        self, hidden_voltage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        voltage_h = hidden_voltage.clone()
        voltage_h[:, :, self.num_steps - 1] = float(threshold + 1)
        spike_h = torch.argmax((voltage_h > threshold).to(torch.int8), dim=2) + 1
        spike_h.clamp_(max=self.num_steps - 1)

        fired_h = spike_h.unsqueeze(-1) <= self.t_grid.view(1, 1, -1)
        voltage_o = torch.einsum("oh,bht->bot", self.w2, fired_h.to(self.dtype))
        voltage_o[:, :, self.num_steps - 1] = float(threshold + 1)
        spike_o = torch.argmax((voltage_o > threshold).to(torch.int8), dim=2) + 1
        spike_o.clamp_(max=self.num_steps - 1)
        prediction = torch.argmin(spike_o, dim=1)
        return prediction, voltage_h, voltage_o

    def _base_and_options(
        self, image: np.ndarray
    ) -> tuple[
        torch.Tensor,
        dict[int, torch.Tensor],
        dict[int, list[tuple[int, int]]],
    ]:
        flat = image.reshape(-1).astype(np.int64)
        spike_time = torch.as_tensor(flat, device=self.device)
        fired = spike_time.unsqueeze(-1) <= self.t_grid.view(1, -1)
        base_voltage = torch.einsum("hp,pt->ht", self.w1, fired.to(self.dtype))

        records: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
        delta_voltage: dict[int, list[torch.Tensor]] = {1: [], 2: []}
        for pixel, old_time in enumerate(flat):
            for shift in (-2, -1, 1, 2):
                cost = abs(shift)
                new_time = int(old_time) + shift
                if new_time < 0 or new_time >= self.num_steps:
                    continue
                indicator_delta = (new_time <= self.t_grid).to(self.dtype) - (
                    int(old_time) <= self.t_grid
                ).to(self.dtype)
                records[cost].append((pixel, shift))
                delta_voltage[cost].append(
                    self.w1[:, pixel].unsqueeze(1) * indicator_delta.unsqueeze(0)
                )

        stacked: dict[int, torch.Tensor] = {}
        for cost in (1, 2):
            stacked[cost] = torch.stack(delta_voltage[cost], dim=0)
        return base_voltage, stacked, records

    def _update_margin_audit(
        self,
        voltage_h: torch.Tensor,
        voltage_o: torch.Tensor,
        audit: dict,
    ) -> None:
        hidden_margin = torch.abs(
            voltage_h[:, :, : self.num_steps - 1] - threshold
        ).amin()
        output_margin = torch.abs(
            voltage_o[:, :, : self.num_steps - 1] - threshold
        ).amin()
        audit["min_hidden_threshold_margin"] = min(
            audit["min_hidden_threshold_margin"], float(hidden_margin.item())
        )
        audit["min_output_threshold_margin"] = min(
            audit["min_output_threshold_margin"], float(output_margin.item())
        )
        near_h = torch.any(
            torch.abs(voltage_h[:, :, : self.num_steps - 1] - threshold)
            <= self.near_threshold_tol,
            dim=(1, 2),
        )
        near_o = torch.any(
            torch.abs(voltage_o[:, :, : self.num_steps - 1] - threshold)
            <= self.near_threshold_tol,
            dim=(1, 2),
        )
        audit["near_threshold_candidates"] += int(torch.sum(near_h | near_o).item())

    def _validate_witness(
        self,
        image: np.ndarray,
        orig_pred: int,
        changes: list[dict],
        claimed_pred: int,
        claimed_cost: int,
    ) -> dict:
        perturbed = image.copy()
        for change in changes:
            x, y = change["coordinate"]
            if int(perturbed[x, y]) != change["from"]:
                raise AssertionError("Witness source value mismatch")
            perturbed[x, y] = change["to"]
        witness_pred = int(forward(self.cfg, self.weights, perturbed))
        l1_cost = int(
            np.sum(np.abs(perturbed.astype(np.int64) - image.astype(np.int64)))
        )
        valid = (
            witness_pred == claimed_pred
            and witness_pred != orig_pred
            and l1_cost == claimed_cost
        )
        if not valid:
            raise AssertionError(
                f"GPU witness did not replay: expected {claimed_pred}/{claimed_cost}, "
                f"got {witness_pred}/{l1_cost}"
            )
        return {
            "witness_pred": witness_pred,
            "l1_cost": l1_cost,
            "num_changed_pixels": len(changes),
            "changes": changes,
        }

    def _changes(
        self,
        image: np.ndarray,
        option_records: list[tuple[int, int]],
    ) -> list[dict]:
        changes = []
        for pixel, shift in option_records:
            x, y = np.unravel_index(pixel, image.shape)
            old_time = int(image[x, y])
            changes.append(
                {
                    "coordinate": [int(x), int(y)],
                    "from": old_time,
                    "to": old_time + int(shift),
                }
            )
        return changes

    def evaluate(self, image: np.ndarray, orig_pred: int) -> dict:
        started = time.perf_counter()
        base_voltage, delta_voltage, records = self._base_and_options(image)
        audit = {
            "min_hidden_threshold_margin": float("inf"),
            "min_output_threshold_margin": float("inf"),
            "near_threshold_candidates": 0,
        }
        candidate_counts = {
            "baseline": 1,
            "cost1_single": len(records[1]),
            "cost2_single": len(records[2]),
            "cost2_pair": 0,
        }

        base_pred_gpu, voltage_h, voltage_o = self._forward_hidden_voltage(
            base_voltage.unsqueeze(0)
        )
        self._update_margin_audit(voltage_h, voltage_o, audit)
        if int(base_pred_gpu.item()) != orig_pred:
            raise AssertionError("GPU and ordinary forward disagree on baseline")

        # Delta=1: every valid unit shift of one pixel.
        for start in range(0, len(records[1]), self.chunk_size):
            end = min(start + self.chunk_size, len(records[1]))
            voltage = base_voltage.unsqueeze(0) + delta_voltage[1][start:end]
            predictions, voltage_h, voltage_o = self._forward_hidden_voltage(voltage)
            self._update_margin_audit(voltage_h, voltage_o, audit)
            adversarial = torch.nonzero(predictions != orig_pred).flatten()
            if len(adversarial):
                local = int(adversarial[0].item())
                option_index = start + local
                predicted = int(predictions[local].item())
                witness = self._validate_witness(
                    image,
                    orig_pred,
                    self._changes(image, [records[1][option_index]]),
                    predicted,
                    1,
                )
                return {
                    "verdict_delta1": "not_robust",
                    "verdict_delta2": "not_robust",
                    "minimum_adversarial_cost": 1,
                    "witness": witness,
                    "candidate_counts": candidate_counts,
                    "evaluated_candidates": 1 + end,
                    "audit": audit,
                    "elapsed_s": time.perf_counter() - started,
                }

        # Cost-2 shift of one pixel.
        for start in range(0, len(records[2]), self.chunk_size):
            end = min(start + self.chunk_size, len(records[2]))
            voltage = base_voltage.unsqueeze(0) + delta_voltage[2][start:end]
            predictions, voltage_h, voltage_o = self._forward_hidden_voltage(voltage)
            self._update_margin_audit(voltage_h, voltage_o, audit)
            adversarial = torch.nonzero(predictions != orig_pred).flatten()
            if len(adversarial):
                local = int(adversarial[0].item())
                option_index = start + local
                predicted = int(predictions[local].item())
                witness = self._validate_witness(
                    image,
                    orig_pred,
                    self._changes(image, [records[2][option_index]]),
                    predicted,
                    2,
                )
                return {
                    "verdict_delta1": "robust",
                    "verdict_delta2": "not_robust",
                    "minimum_adversarial_cost": 2,
                    "witness": witness,
                    "candidate_counts": candidate_counts,
                    "evaluated_candidates": (
                        1 + len(records[1]) + end
                    ),
                    "audit": audit,
                    "elapsed_s": time.perf_counter() - started,
                }

        # Two distinct pixels, each shifted by one timestep.
        unit_pixels = np.asarray([pixel for pixel, _ in records[1]], dtype=np.int64)
        left, right = np.triu_indices(len(records[1]), k=1)
        distinct = unit_pixels[left] != unit_pixels[right]
        left = left[distinct]
        right = right[distinct]
        candidate_counts["cost2_pair"] = int(len(left))
        evaluated = 1 + len(records[1]) + len(records[2])

        for start in range(0, len(left), self.chunk_size):
            end = min(start + self.chunk_size, len(left))
            left_idx = torch.as_tensor(left[start:end], device=self.device)
            right_idx = torch.as_tensor(right[start:end], device=self.device)
            voltage = (
                base_voltage.unsqueeze(0)
                + delta_voltage[1][left_idx]
                + delta_voltage[1][right_idx]
            )
            predictions, voltage_h, voltage_o = self._forward_hidden_voltage(voltage)
            self._update_margin_audit(voltage_h, voltage_o, audit)
            adversarial = torch.nonzero(predictions != orig_pred).flatten()
            if len(adversarial):
                local = int(adversarial[0].item())
                left_option = int(left[start + local])
                right_option = int(right[start + local])
                predicted = int(predictions[local].item())
                witness = self._validate_witness(
                    image,
                    orig_pred,
                    self._changes(
                        image,
                        [records[1][left_option], records[1][right_option]],
                    ),
                    predicted,
                    2,
                )
                return {
                    "verdict_delta1": "robust",
                    "verdict_delta2": "not_robust",
                    "minimum_adversarial_cost": 2,
                    "witness": witness,
                    "candidate_counts": candidate_counts,
                    "evaluated_candidates": evaluated + (end - start),
                    "audit": audit,
                    "elapsed_s": time.perf_counter() - started,
                }
            evaluated += end - start

        return {
            "verdict_delta1": "robust",
            "verdict_delta2": "robust",
            "minimum_adversarial_cost": None,
            "witness": None,
            "candidate_counts": candidate_counts,
            "evaluated_candidates": evaluated,
            "audit": audit,
            "elapsed_s": time.perf_counter() - started,
        }


def merge_results(paths: list[Path], output: Path) -> None:
    documents = [json.loads(path.read_text()) for path in paths]
    results = sorted(
        [row for document in documents for row in document["results"]],
        key=lambda row: row["selection_order"],
    )
    if len({row["selection_order"] for row in results}) != len(results):
        raise RuntimeError("Duplicate selection_order while merging shards")
    delta1_not_robust = sum(
        row["verdict_delta1"] == "not_robust" for row in results
    )
    delta2_not_robust = sum(
        row["verdict_delta2"] == "not_robust" for row in results
    )
    merged = {
        "config": documents[0]["config"]
        | {
            "num_shards": len(documents),
            "shard_index": None,
            "merged_from": [str(path) for path in paths],
        },
        "eligible_population": documents[0]["eligible_population"],
        "summary": {
            "n_samples": len(results),
            "delta1_not_robust": delta1_not_robust,
            "delta1_robust": len(results) - delta1_not_robust,
            "delta2_not_robust": delta2_not_robust,
            "delta2_robust": len(results) - delta2_not_robust,
            "minimum_cost_1": sum(
                row["minimum_adversarial_cost"] == 1 for row in results
            ),
            "minimum_cost_2": sum(
                row["minimum_adversarial_cost"] == 2 for row in results
            ),
            "all_near_threshold_candidates": sum(
                row["audit"]["near_threshold_candidates"] for row in results
            ),
            "minimum_hidden_threshold_margin": min(
                row["audit"]["min_hidden_threshold_margin"] for row in results
            ),
            "minimum_output_threshold_margin": min(
                row["audit"]["min_output_threshold_margin"] for row in results
            ),
            "total_elapsed_s_sum": sum(row["elapsed_s"] for row in results),
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(merged["summary"], indent=2), flush=True)
    print(f"Saved merged result: {output}", flush=True)


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_results(args.merge, args.output)
        return
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    cfg = make_cfg(args.n_samples, args.n_hidden, args.seed)
    weights = prepare_weights(cfg=cfg, subtype="mnist", load_data_func=load_mnist)
    _, _, images_test, labels_test = load_mnist(cfg)
    selected, eligible = select_examples(cfg, weights, images_test, labels_test)
    shard = selected[args.shard_index :: args.num_shards]
    enumerator = ExactDelta2Enumerator(
        cfg,
        weights,
        args.device,
        args.dtype,
        args.chunk_size,
        args.near_threshold_tol,
    )

    results = []
    for position, example in enumerate(shard, start=1):
        row = example | enumerator.evaluate(
            images_test[example["index"]], example["orig_pred"]
        )
        results.append(row)
        print(
            f"[{position}/{len(shard)}] idx={row['index']} "
            f"d1={row['verdict_delta1']} d2={row['verdict_delta2']} "
            f"min_cost={row['minimum_adversarial_cost']} "
            f"time={row['elapsed_s']:.3f}s",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "config": {
                        "n_samples": args.n_samples,
                        "n_hidden": args.n_hidden,
                        "seed": args.seed,
                        "split": "test",
                        "device": args.device,
                        "dtype": args.dtype,
                        "chunk_size": args.chunk_size,
                        "near_threshold_tol": args.near_threshold_tol,
                        "num_shards": args.num_shards,
                        "shard_index": args.shard_index,
                    },
                    "eligible_population": eligible,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Saved shard result: {args.output}", flush=True)


if __name__ == "__main__":
    main()
