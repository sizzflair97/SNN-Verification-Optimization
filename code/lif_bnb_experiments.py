#!/usr/bin/env python3
"""Reproducible experiment matrix for continuous-time LIF BnB verification."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

from goltz_lif_ttfs import GoeltzLIFNetwork
from lif_bnb import (
    LIFThreatSpec,
    LIFVerifyConfig,
    load_lif_model_spec,
    verify_lif_ttfs,
)


DEFAULT_CHECKPOINT = Path(
    "/data/SNN-Verification-Optimization/models/"
    "lif_goltz_ct_784_350_10_seed0/best.pt"
)
REFERENCE_CHECKPOINT_SHA256 = (
    "a7c7c660857bec105abad959fa546de840f63634687ed19fc838546c7c257eac"
)
DEFAULT_MNIST_ROOT = Path(
    "/data/SNN-Verification-Optimization/code/data/mnist/MNIST/raw"
)
FIXED_TRAIN_SAMPLES = [
    41905, 7296, 1639, 48598, 18024, 16049, 14628,
    9144, 48265, 6717, 44348, 48540, 58469, 35741,
]
METHOD_MATRIX = [
    "exhaustive",
    "bnb_no_bound",
    "bnb_ordered",
    "bnb_uncoupled",
    "bnb_coupled",
    "bnb_coupled_cache",
]
KAPPA_VALUES = [10, 20, 50, 100, 200, 500, 1000]
_WORKER_MODEL_CACHE: dict[str, Any] = {}
_WORKER_DEVICE: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("pilot", "table1", "scaling", "kappa", "incidence", "all", "manifest"),
        default="pilot",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--mnist-root", type=Path, default=DEFAULT_MNIST_ROOT)
    parser.add_argument("--results-dir", type=Path, default=Path("code/bench_results/lif"))
    parser.add_argument("--backend", choices=("numpy", "torch"),
                        default="torch" if torch.cuda.is_available() else "numpy")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--min-time-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-samples", type=int, default=10)
    parser.add_argument("--incidence-samples", type=int, default=50)
    parser.add_argument("--exact-batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_idx(path: Path, offset: int, shape: tuple[int, ...]) -> np.ndarray:
    if path.exists():
        raw = path.read_bytes()
    elif path.with_suffix(path.suffix + ".gz").exists():
        with gzip.open(path.with_suffix(path.suffix + ".gz"), "rb") as handle:
            raw = handle.read()
    else:
        raise FileNotFoundError(path)
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)


def load_mnist(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        count, image_name, label_name = (
            60000, "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
        )
    elif split == "test":
        count, image_name, label_name = (
            10000, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
        )
    else:
        raise ValueError("split must be train or test")
    images = _read_idx(root / image_name, 16, (count, 784))
    labels = _read_idx(root / label_name, 8, (count,))
    return images, labels


def encode_times(images: np.ndarray, early: float, late: float) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32) / 255.0
    return (early + (1.0 - values) * (late - early)).astype(np.float64)


def load_torch_model(checkpoint: Path, device: str):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    weight_keys = sorted(
        (key for key in state["model"] if key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    dims = [state["model"][weight_keys[0]].shape[1]]
    dims.extend(state["model"][key].shape[0] for key in weight_keys)
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
    return model.to(device).eval(), config


@torch.no_grad()
def select_correct_samples(
    checkpoint: Path,
    images: np.ndarray,
    labels: np.ndarray,
    count: int,
    seed: int,
    device: str,
) -> list[dict[str, int]]:
    model, config = load_torch_model(checkpoint, device)
    permutation = np.random.default_rng(seed).permutation(len(images))
    selected: list[dict[str, int]] = []
    for start in range(0, len(permutation), 128):
        indices = permutation[start:start + 128]
        times = encode_times(images[indices], float(config["early"]), float(config["late"]))
        output = model(torch.as_tensor(times, dtype=torch.float32, device=device))[-1]
        masked = output.times.masked_fill(~output.spiked, float("inf"))
        prediction = masked.argmin(dim=1).masked_fill(~output.spiked.any(dim=1), -1)
        sorted_times = masked.sort(dim=1).values
        unique = (sorted_times[:, 1] - sorted_times[:, 0]) > 1e-7
        for index, pred, is_unique in zip(indices, prediction.cpu(), unique.cpu()):
            if bool(is_unique) and int(pred) == int(labels[index]):
                selected.append(
                    {"index": int(index), "label": int(labels[index]), "prediction": int(pred)}
                )
                if len(selected) == count:
                    return selected
    raise RuntimeError(f"selected only {len(selected)} of {count} requested samples")


def coverage_manifest(model_dims: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "artifact": "Table 1/5 SMT",
            "status": "N/A",
            "reason": "exact continuous-LIF SMT encoding unavailable",
        },
        {
            "artifact": "Table 1/5 MILP",
            "status": "N/A",
            "reason": "exact continuous-LIF MILP encoding unavailable",
        },
        {
            "artifact": "Table 2 widths 100/200/300/500",
            "status": "pending_checkpoint",
            "available_model": list(model_dims),
        },
        {
            "artifact": "Table 3 cross-dataset",
            "status": "pending_checkpoint",
        },
        {
            "artifact": "Table 4 multi-hidden CLEB",
            "status": "pending_checkpoint",
            "reason": "current model has one hidden layer",
        },
        {
            "artifact": "Figure 2 width scalability",
            "status": "pending_checkpoint",
        },
    ]


def make_job(
    *,
    suite: str,
    split: str,
    sample: int,
    label: int,
    input_times: np.ndarray,
    method: str,
    delta: int,
    kappa: int,
    timeout: float,
    args: argparse.Namespace,
    ordinal: int,
) -> dict[str, Any]:
    device = args.device
    if args.backend == "torch" and device.startswith("cuda") and torch.cuda.device_count() > 1:
        device = f"cuda:{ordinal % torch.cuda.device_count()}"
    job_id = f"{suite}:{split}:{sample}:d{delta}:{method}:k{kappa}"
    return {
        "job_id": job_id,
        "suite": suite,
        "split": split,
        "sample": int(sample),
        "label": int(label),
        "input_times": np.asarray(input_times, dtype=np.float64),
        "method": method,
        "delta": int(delta),
        "kappa": int(kappa),
        "timeout": float(timeout),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_source": str(args.checkpoint_source),
        "backend": args.backend,
        "device": device,
        "time_step": float(args.time_step),
        "min_time_step": float(args.min_time_step),
        "exact_batch_size": int(args.exact_batch_size),
    }


def initialize_worker(base_device: str, device_count: int) -> None:
    """Permanently bind one spawned worker to one CUDA device."""
    global _WORKER_DEVICE
    if base_device.startswith("cuda") and device_count > 0:
        identity = multiprocessing.current_process()._identity
        ordinal = (identity[0] - 1) if identity else 0
        index = ordinal % device_count
        _WORKER_DEVICE = f"cuda:{index}"
        torch.cuda.set_device(index)
    else:
        _WORKER_DEVICE = base_device
def _worker_model(checkpoint: str):
    model = _WORKER_MODEL_CACHE.get(checkpoint)
    if model is None:
        model = load_lif_model_spec(checkpoint)
        _WORKER_MODEL_CACHE[checkpoint] = model
    return model


@torch.no_grad()
def replay_torch_witness(
    checkpoint: str, perturbed_times: list[float], expected: int, device: str
) -> tuple[bool, int]:
    model, _ = load_torch_model(Path(checkpoint), device)
    output = model(
        torch.as_tensor([perturbed_times], dtype=torch.float32, device=device)
    )[-1]
    masked = output.times.masked_fill(~output.spiked, float("inf"))
    prediction = masked.argmin(dim=1).masked_fill(~output.spiked.any(dim=1), -1)
    actual = int(prediction.item())
    return actual == expected, actual


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    actual_device = _WORKER_DEVICE or job["device"]
    model = _worker_model(job["checkpoint"])
    threat = LIFThreatSpec(
        job["delta"],
        (model.input_max - model.input_min) / 4.0,
        model.input_min,
        model.input_max,
    )
    config = LIFVerifyConfig(
        method=job["method"],
        bound_every=job["kappa"],
        timeout_s=job["timeout"],
        backend=job["backend"],
        device=actual_device,
        time_step=job["time_step"],
        min_time_step=job["min_time_step"],
        exact_batch_size=job["exact_batch_size"],
    )
    result = verify_lif_ttfs(model, job["input_times"], threat, config)
    row = {
        key: value for key, value in job.items() if key != "input_times"
    }
    row.update(
        {
            "device": actual_device,
            "eta": threat.shift_step,
            "model_dims": list(model.dims),
            "checkpoint_hash": model.checkpoint_hash,
            "checkpoint_epoch": model.checkpoint_epoch,
            "checkpoint_accuracy": model.checkpoint_accuracy,
            "result": result.to_dict(),
        }
    )
    if result.verdict == "not_robust" and result.witness is not None:
        matched, torch_prediction = replay_torch_witness(
            job["checkpoint"],
            result.witness["perturbed_times"],
            result.witness["prediction"],
            actual_device,
        )
        row["torch_witness_prediction"] = torch_prediction
        row["torch_witness_matches"] = matched
        if not matched:
            row["result"]["verdict"] = "numerical_unknown"
            row["result"]["reason"] = "NumPy and checkpoint Torch witness predictions disagree"
    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    payload = (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_completed(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return {row["job_id"] for row in rows}, rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def freeze_checkpoint(args: argparse.Namespace):
    """Snapshot one immutable checkpoint and enforce the requested hash."""
    source = args.checkpoint.resolve()
    source_model = load_lif_model_spec(source)
    expected = (
        args.expected_checkpoint_sha256
        or (
            REFERENCE_CHECKPOINT_SHA256
            if source == DEFAULT_CHECKPOINT.resolve()
            else None
        )
    )
    if expected is not None and source_model.checkpoint_hash != expected:
        raise RuntimeError(
            "checkpoint hash mismatch: "
            f"expected {expected}, got {source_model.checkpoint_hash} at {source}"
        )
    frozen_dir = args.results_dir / "checkpoints"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_dir / f"{source_model.checkpoint_hash}.pt"
    if not frozen.exists():
        temporary = frozen.with_suffix(f".tmp.{os.getpid()}")
        shutil.copy2(source, temporary)
        copied = load_lif_model_spec(temporary)
        if copied.checkpoint_hash != source_model.checkpoint_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("checkpoint changed while it was being snapshotted")
        os.replace(temporary, frozen)
    frozen_model = load_lif_model_spec(frozen)
    if frozen_model.checkpoint_hash != source_model.checkpoint_hash:
        raise RuntimeError("frozen checkpoint hash does not match its filename")
    args.checkpoint_source = source
    args.checkpoint = frozen
    return frozen_model


def build_jobs(args: argparse.Namespace, model, train, train_labels, test, test_labels):
    train_times = encode_times(train, model.input_min, model.input_max)
    test_times = encode_times(test, model.input_min, model.input_max)
    correct_test = select_correct_samples(
        args.checkpoint,
        test,
        test_labels,
        max(args.test_samples, args.incidence_samples),
        args.seed,
        args.device,
    )
    jobs: list[dict[str, Any]] = []

    def add(**kwargs):
        jobs.append(make_job(args=args, ordinal=len(jobs), **kwargs))

    suites = (
        {"table1", "scaling", "kappa", "incidence"}
        if args.suite == "all"
        else {args.suite}
    )
    if "pilot" in suites:
        pilot = correct_test[0]
        samples = [
            ("train", 9144, int(train_labels[9144]), train_times[9144]),
            ("test", pilot["index"], pilot["label"], test_times[pilot["index"]]),
        ]
        for split, index, label, times in samples:
            for method in METHOD_MATRIX:
                add(
                    suite="pilot", split=split, sample=index, label=label,
                    input_times=times, method=method, delta=1, kappa=1,
                    timeout=args.timeout or 300.0,
                )
    if "table1" in suites:
        for index in FIXED_TRAIN_SAMPLES:
            for method in METHOD_MATRIX:
                add(
                    suite="table1_5", split="train", sample=index,
                    label=int(train_labels[index]), input_times=train_times[index],
                    method=method, delta=1, kappa=100,
                    timeout=args.timeout or 300.0,
                )
    if "scaling" in suites:
        for delta in range(1, 6):
            for selected in correct_test[:args.test_samples]:
                index = selected["index"]
                for method in METHOD_MATRIX:
                    add(
                        suite="table2_7_scaling", split="test", sample=index,
                        label=selected["label"], input_times=test_times[index],
                        method=method, delta=delta, kappa=100,
                        timeout=args.timeout or 300.0,
                    )
    if "kappa" in suites:
        for kappa in KAPPA_VALUES:
            add(
                suite="table6_kappa", split="train", sample=9144,
                label=int(train_labels[9144]), input_times=train_times[9144],
                method="bnb_coupled", delta=2, kappa=kappa,
                timeout=args.timeout or 300.0,
            )
    if "incidence" in suites:
        for selected in correct_test[:args.incidence_samples]:
            index = selected["index"]
            add(
                suite="incidence_d1", split="test", sample=index,
                label=selected["label"], input_times=test_times[index],
                method="bnb_coupled", delta=1, kappa=100,
                timeout=args.timeout or 120.0,
            )
    return jobs, correct_test


def summarize(rows: list[dict[str, Any]], manifest, model, selected):
    counts = defaultdict(Counter)
    for row in rows:
        key = f"{row['suite']}:{row['method']}:d{row['delta']}"
        counts[key][row["result"]["verdict"]] += 1
    delta2 = [
        row for row in rows
        if row["suite"] == "table2_7_scaling" and row["delta"] == 2
    ]
    row_hashes = sorted({row.get("checkpoint_hash") for row in rows})
    checkpoint_consistent = row_hashes in ([], [model.checkpoint_hash])
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "checkpoint": model.checkpoint,
            "sha256": model.checkpoint_hash,
            "epoch": model.checkpoint_epoch,
            "accuracy": model.checkpoint_accuracy,
            "dims": list(model.dims),
        },
        "integrity": {
            "checkpoint_hashes": row_hashes,
            "checkpoint_consistent": checkpoint_consistent,
        },
        "threat": {
            "T": 5,
            "eta": (model.input_max - model.input_min) / 4.0,
            "input_min": model.input_min,
            "input_max": model.input_max,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
        "coverage": manifest,


        "selected_correct_test": selected,
        "counts": {key: dict(value) for key, value in counts.items()},
        "table7_delta2_per_sample": delta2,
        "rows": len(rows),
    }
def execute_jobs(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    jsonl: Path,
    rows: list[dict[str, Any]],
    *,
    phase: str,
) -> None:
    if not jobs:
        return
    if args.workers == 1:
        for done, job in enumerate(jobs, 1):
            row = run_job(job)
            append_jsonl(jsonl, row)
            rows.append(row)
            print(
                f"[{phase} {done}/{len(jobs)}] {job['job_id']} "
                f"{row['result']['verdict']} "
                f"{row['result'].get('elapsed_s', 0.0):.3f}s",
                flush=True,
            )
        return

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(args.device if args.backend == "torch" else "cpu", torch.cuda.device_count()),
    ) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    key: value for key, value in job.items() if key != "input_times"
                }
                row["result"] = {
                    "verdict": "error",
                    "reason": "worker process failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            append_jsonl(jsonl, row)
            rows.append(row)
            print(
                f"[{phase} {done}/{len(jobs)}] {job['job_id']} "
                f"{row['result']['verdict']}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model = freeze_checkpoint(args)
    manifest = coverage_manifest(model.dims)
    manifest_path = args.results_dir / "coverage_manifest.json"
    atomic_json(manifest_path, {"model_dims": list(model.dims), "coverage": manifest})
    if args.suite == "manifest":
        print(manifest_path)
        return

    train, train_labels = load_mnist(args.mnist_root, "train")
    test, test_labels = load_mnist(args.mnist_root, "test")
    jobs, selected = build_jobs(
        args, model, train, train_labels, test, test_labels
    )
    jsonl = args.results_dir / f"h350_{args.suite}.jsonl"
    if args.force and jsonl.exists():
        jsonl.unlink()
    completed, rows = load_completed(jsonl)
    row_hashes = {row.get("checkpoint_hash") for row in rows}
    if row_hashes and row_hashes != {model.checkpoint_hash}:
        raise RuntimeError(
            "refusing to resume results from a different checkpoint: "
            f"rows contain {sorted(row_hashes)}, current snapshot is "
            f"{model.checkpoint_hash}"
        )
    pending = [job for job in jobs if job["job_id"] not in completed]
    print(
        f"suite={args.suite} jobs={len(jobs)} completed={len(completed)} "
        f"pending={len(pending)} output={jsonl}",
        flush=True,
    )

    execute_jobs(pending, args, jsonl, rows, phase="primary")

    if args.suite in {"incidence", "all"}:
        robust_at_one = {
            int(row["sample"])
            for row in rows
            if row.get("suite") == "incidence_d1"
            and row.get("result", {}).get("verdict") == "robust"
        }
        completed_ids = {row["job_id"] for row in rows}
        test_times = encode_times(test, model.input_min, model.input_max)
        delta_two_jobs = []
        for selected_row in selected[:args.incidence_samples]:
            index = int(selected_row["index"])
            if index not in robust_at_one:
                continue
            job = make_job(
                suite="incidence_d2",
                split="test",
                sample=index,
                label=int(selected_row["label"]),
                input_times=test_times[index],
                method="bnb_coupled",
                delta=2,
                kappa=100,
                timeout=args.timeout or 120.0,
                args=args,
                ordinal=len(jobs) + len(delta_two_jobs),
            )
            if job["job_id"] not in completed_ids:
                delta_two_jobs.append(job)
        print(
            f"incidence Delta=2 eligible={len(robust_at_one)} "
            f"pending={len(delta_two_jobs)}",
            flush=True,
        )
        execute_jobs(
            delta_two_jobs,
            args,
            jsonl,
            rows,
            phase="incidence-d2-survivors",
        )
    summary = summarize(rows, manifest, model, selected)
    summary_path = args.results_dir / f"h350_{args.suite}_summary.json"
    atomic_json(summary_path, summary)
    if args.suite in {"scaling", "all"}:
        delta2_path = args.results_dir / f"h350_{args.suite}_delta2_per_sample.json"
        atomic_json(
            delta2_path,
            {
                "model": summary["model"], "threat": summary["threat"],
                "rows": summary["table7_delta2_per_sample"],
            },
        )
    print(summary_path)


if __name__ == "__main__":
    main()

