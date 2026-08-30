"""Complete finite-domain BnB verification for continuous-time LIF TTFS SNNs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

from goltz_lif_ttfs import GoeltzLIFNetwork
from lif_bnb_ibp import canonical_shift_options, lif_bcibp_bound_node
from utils.lif_ttfs_net import LIFLayerForwardResult, lif_ttfs_forward_batch


VERDICTS = {"robust", "not_robust", "timeout", "numerical_unknown", "error"}
METHODS = {
    "exhaustive",
    "bnb_no_bound",
    "bnb_ordered",
    "bnb_uncoupled",
    "bnb_coupled",
    "bnb_coupled_cache",
}
BOUND_METHODS = {"bnb_uncoupled", "bnb_coupled", "bnb_coupled_cache"}


@dataclass(frozen=True)
class LIFModelSpec:
    weights: tuple[np.ndarray, ...]
    tau: float = 1.0
    threshold: float = 1.0
    input_min: float = 0.15
    input_max: float = 2.0
    checkpoint: str | None = None
    checkpoint_hash: str | None = None
    checkpoint_epoch: int | None = None
    checkpoint_accuracy: float | None = None

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must not be empty")
        previous = self.weights[0].shape[1]
        for index, weight in enumerate(self.weights):
            if weight.ndim != 2:
                raise ValueError("all weights must be matrices")
            if index and weight.shape[1] != self.weights[index - 1].shape[0]:
                raise ValueError("incompatible layer dimensions")
        if self.tau <= 0.0 or self.threshold <= 0.0:
            raise ValueError("tau and threshold must be positive")
        if self.input_min >= self.input_max:
            raise ValueError("input_min must be smaller than input_max")

    @property
    def dims(self) -> tuple[int, ...]:
        return (self.weights[0].shape[1], *(value.shape[0] for value in self.weights))

    @property
    def no_spike_time(self) -> float:
        """Finite decoder deadline shared with the trained Torch model."""
        return max(10.0 * self.tau, self.input_max + 6.0 * self.tau)


@dataclass(frozen=True)
class LIFThreatSpec:
    budget: int
    shift_step: float
    input_min: float
    input_max: float

    def __post_init__(self) -> None:
        if self.budget < 0 or self.shift_step <= 0.0:
            raise ValueError("invalid finite time-shift threat")
        if self.input_min >= self.input_max:
            raise ValueError("invalid clipping range")


@dataclass(frozen=True)
class LIFVerifyConfig:
    method: str = "bnb_coupled"
    bound_every: int = 100
    timeout_s: float = 300.0
    backend: str = "numpy"
    device: str | None = None
    time_step: float = 0.05
    min_time_step: float = 0.05
    numerical_tolerance: float = 1e-9
    exact_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.bound_every <= 0 or self.timeout_s <= 0.0:
            raise ValueError("bound_every and timeout_s must be positive")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("backend must be numpy or torch")
        if not 0.0 < self.min_time_step <= self.time_step:
            raise ValueError("invalid time partition")
        if self.exact_batch_size <= 0:
            raise ValueError("exact_batch_size must be positive")


@dataclass
class LIFVerificationStats:
    nodes: int = 0
    exact_forwards: int = 0
    bound_calls: int = 0
    bound_prunes: int = 0
    bound_cache_hits: int = 0
    state_cache_hits: int = 0
    maximum_depth: int = 0
    canonical_active_inputs: int = 0


@dataclass
class LIFVerificationResult:
    verdict: str
    baseline_prediction: int
    method: str
    budget: int
    shift_step: float
    elapsed_s: float
    stats: LIFVerificationStats
    witness: dict[str, Any] | None = None
    reason: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"invalid verdict: {self.verdict}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


def load_lif_model_spec(checkpoint: str | Path) -> LIFModelSpec:
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    model_state = state["model"]
    indexed: list[tuple[int, np.ndarray]] = []
    pattern = re.compile(r"^layers[.](\d+)[.]weight$")
    for key, tensor in model_state.items():
        match = pattern.match(key)
        if match:
            indexed.append(
                (int(match.group(1)), tensor.detach().cpu().numpy().astype(np.float64))
            )
    indexed.sort(key=lambda item: item[0])
    if not indexed or [item[0] for item in indexed] != list(range(len(indexed))):
        raise ValueError("checkpoint has no contiguous LIF layer weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return LIFModelSpec(
        weights=tuple(item[1] for item in indexed),
        tau=float(config["tau_syn"]),
        threshold=float(config["threshold"]),
        input_min=float(config["early"]),
        input_max=float(config["late"]),
        checkpoint=str(path),
        checkpoint_hash=digest,
        checkpoint_epoch=int(state["epoch"]),
        checkpoint_accuracy=float(state["test_accuracy"]),
    )


def _prediction_and_margin(
    model: LIFModelSpec, input_times: np.ndarray
) -> tuple[int, float]:
    layers: list[LIFLayerForwardResult] = []
    prediction = int(
        lif_ttfs_forward_batch(
            model.weights,
            np.asarray(input_times, dtype=np.float64),
            tau_syn=model.tau,
            threshold=model.threshold,
            no_spike_time=model.no_spike_time,
            layer_results_return=layers,
        )
    )
    output = layers[-1]
    ordered = np.sort(np.asarray(output.times).reshape(-1))
    margin = float(ordered[1] - ordered[0]) if ordered.size >= 2 else float("inf")
    return prediction, margin


def _sensitivity_order(
    model: LIFModelSpec,
    baseline: np.ndarray,
    threat: LIFThreatSpec,
    all_options: Sequence[tuple[tuple[int, int, float], ...]],
) -> list[int]:
    first = model.weights[0]
    scores = []
    for index, choices in enumerate(all_options):
        movement = max(abs(item[2] - baseline[index]) for item in choices)
        score = float(np.max(np.abs(first[:, index]))) * movement / model.tau
        scores.append((score, index))
    return [index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))]


def _verify_batched_enumeration(
    model: LIFModelSpec,
    baseline: np.ndarray,
    threat: LIFThreatSpec,
    config: LIFVerifyConfig,
    baseline_prediction: int,
    all_options: Sequence[tuple[tuple[int, int, float], ...]],
    order: Sequence[int],
    started: float,
    stats: LIFVerificationStats,
) -> LIFVerificationResult:
    """Complete canonical enumeration with vectorized exact forward batches."""
    deadline = started + config.timeout_s
    current = baseline.copy()
    shifts = np.zeros(baseline.size, dtype=np.int16)
    numerical_seen = False
    batch_times: list[np.ndarray] = []
    batch_shifts: list[np.ndarray] = []
    batch_depths: list[int] = []

    torch_exact = None
    torch_device = None
    if config.backend == "torch":
        torch_device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch_exact = GoeltzLIFNetwork(
            list(model.dims),
            tau_syn=model.tau,
            threshold=model.threshold,
            no_spike_time=model.no_spike_time,
        ).to(device=torch_device, dtype=torch.float64).eval()
        with torch.no_grad():
            for layer, weight in zip(torch_exact.layers, model.weights):
                layer.weight.copy_(
                    torch.as_tensor(weight, dtype=torch.float64, device=torch_device)
                )
    def finish(
        verdict: str,
        reason: str,
        witness: dict[str, Any] | None = None,
    ) -> LIFVerificationResult:
        return LIFVerificationResult(
            verdict,
            baseline_prediction,
            config.method,
            threat.budget,
            threat.shift_step,
            time.perf_counter() - started,
            stats,
            witness=witness,
            reason=reason,
        )

    def flush() -> LIFVerificationResult | None:
        nonlocal numerical_seen
        if not batch_times:
            return None
        if torch_exact is None:
            predictions = np.asarray(
                lif_ttfs_forward_batch(
                    model.weights,
                    np.asarray(batch_times, dtype=np.float64),
                    tau_syn=model.tau,
                    threshold=model.threshold,
                    no_spike_time=model.no_spike_time,
                )
            )
        else:
            with torch.no_grad():
                output = torch_exact(
                    torch.as_tensor(
                        np.asarray(batch_times),
                        dtype=torch.float64,
                        device=torch_device,
                    )
                )[-1]
                predictions = output.times.argmin(dim=1).cpu().numpy()
        stats.exact_forwards += len(batch_times)
        stats.nodes += len(batch_times)
        stats.maximum_depth = max(stats.maximum_depth, max(batch_depths))
        for row, prediction in enumerate(predictions):
            if int(prediction) == baseline_prediction:
                continue
            replay_prediction, margin = _prediction_and_margin(model, batch_times[row])
            stats.exact_forwards += 1
            if replay_prediction == baseline_prediction:
                numerical_seen = True
                continue
            shift_vector = batch_shifts[row].astype(int)
            witness = {
                "shift_vector": shift_vector.tolist(),
                "sparse_shifts": {
                    str(index): int(value)
                    for index, value in enumerate(shift_vector)
                    if value
                },
                "l1_cost": int(np.abs(shift_vector).sum()),
                "perturbed_times": batch_times[row].tolist(),
                "prediction": int(replay_prediction),
                "baseline_prediction": int(baseline_prediction),
                "output_margin": margin,
            }
            batch_times.clear()
            batch_shifts.clear()
            batch_depths.clear()
            return finish("not_robust", "exact adversarial shift replayed", witness)
        batch_times.clear()
        batch_shifts.clear()
        batch_depths.clear()
        if time.perf_counter() >= deadline:
            return finish("timeout", "per-sample timeout")
        return None

    def candidates(start_position: int, remaining: int, depth: int):
        for position in range(start_position, len(order)):
            index = order[position]
            original_time = current[index]
            for shift, cost, changed_time in all_options[index]:
                if shift == 0 or cost > remaining:
                    continue
                current[index] = changed_time
                shifts[index] = shift
                yield current.copy(), shifts.copy(), depth + 1
                if remaining > cost:
                    yield from candidates(position + 1, remaining - cost, depth + 1)
                current[index] = original_time
                shifts[index] = 0

    for candidate_times, candidate_shifts, depth in candidates(0, threat.budget, 0):
        batch_times.append(candidate_times)
        batch_shifts.append(candidate_shifts)
        batch_depths.append(depth)
        if len(batch_times) >= config.exact_batch_size:
            result = flush()
            if result is not None:
                return result
    result = flush()
    if result is not None:
        return result
    if numerical_seen:
        return finish(
            "numerical_unknown",
            "only numerically tied prediction changes were encountered",
        )
    return finish(
        "robust",
        "complete canonical finite shift set exhausted",
    )
def verify_lif_ttfs(
    model: LIFModelSpec,
    input_times: np.ndarray,
    threat: LIFThreatSpec,
    config: LIFVerifyConfig,
    *,
    baseline_prediction: int | None = None,
) -> LIFVerificationResult:
    """Verify invariance of the exact earliest-output-spike prediction."""
    started = time.perf_counter()
    stats = LIFVerificationStats()
    baseline = np.asarray(input_times, dtype=np.float64).reshape(-1)
    if baseline.size != model.dims[0]:
        raise ValueError("input dimension does not match model")
    if (
        threat.input_min != model.input_min
        or threat.input_max != model.input_max
    ):
        raise ValueError("threat and model clipping ranges differ")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), baseline.size + 1000))

    try:
        predicted, _ = _prediction_and_margin(model, baseline)
        stats.exact_forwards += 1
        if baseline_prediction is None:
            baseline_prediction = predicted
        elif int(baseline_prediction) != predicted:
            raise ValueError("provided baseline prediction does not match exact forward")
        all_options = [
            canonical_shift_options(
                value,
                threat.budget,
                threat.shift_step,
                threat.input_min,
                threat.input_max,
            )
            for value in baseline
        ]
        active = [
            index
            for index, choices in enumerate(all_options)
            if any(choice[0] != 0 for choice in choices)
        ]
        if config.method in {
            "bnb_ordered",
            "bnb_uncoupled",
            "bnb_coupled",
            "bnb_coupled_cache",
        }:
            ordered = _sensitivity_order(model, baseline, threat, all_options)
            order = [index for index in ordered if index in set(active)]
        else:
            order = active
        stats.canonical_active_inputs = len(order)
        if config.method in {
            "exhaustive",
            "bnb_no_bound",
            "bnb_ordered",
        }:
            return _verify_batched_enumeration(
                model,
                baseline,
                threat,
                config,
                int(baseline_prediction),
                all_options,
                order,
                started,
                stats,
            )

        if config.method in BOUND_METHODS and threat.budget == 1:
            stats.bound_calls += 1
            certificate = lif_bcibp_bound_node(
                baseline,
                model.weights,
                budget=1,
                shift_step=threat.shift_step,
                input_min=threat.input_min,
                input_max=threat.input_max,
                mutable_indices=order,
                coupled=config.method != "bnb_uncoupled",
                tau=model.tau,
                threshold=model.threshold,
                no_spike_time=model.no_spike_time,
                time_step=config.time_step,
                min_time_step=config.min_time_step,
                prediction=int(baseline_prediction),
                backend=config.backend,
                device=config.device,
            )
            if time.perf_counter() >= started + config.timeout_s:
                return LIFVerificationResult(
                    "timeout",
                    int(baseline_prediction),
                    config.method,
                    threat.budget,
                    threat.shift_step,
                    time.perf_counter() - started,
                    stats,
                    reason="per-sample timeout",
                )
            if certificate.robust:
                stats.bound_prunes += 1
                return LIFVerificationResult(
                    "robust",
                    int(baseline_prediction),
                    config.method,
                    threat.budget,
                    threat.shift_step,
                    time.perf_counter() - started,
                    stats,
                    reason="root subtree soundly pruned",
                )
            return _verify_batched_enumeration(
                model,
                baseline,
                threat,
                config,
                int(baseline_prediction),
                all_options,
                order,
                started,
                stats,
            )
        torch_dfs = None
        torch_dfs_device = None
        if config.backend == "torch":
            torch_dfs_device = torch.device(
                config.device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            torch_dfs = GoeltzLIFNetwork(
                list(model.dims),
                tau_syn=model.tau,
                threshold=model.threshold,
                no_spike_time=model.no_spike_time,
            ).to(device=torch_dfs_device, dtype=torch.float64).eval()
            with torch.no_grad():
                for layer, weight in zip(torch_dfs.layers, model.weights):
                    layer.weight.copy_(
                        torch.as_tensor(
                            weight,
                            dtype=torch.float64,
                            device=torch_dfs_device,
                        )
                    )
        current = baseline.copy()
        shifts = np.zeros(baseline.size, dtype=np.int16)
        deadline = started + config.timeout_s
        witness: dict[str, Any] | None = None
        numerical_seen = False
        bound_cache: dict[tuple[Any, ...], bool] = {}
        state_cache: set[tuple[int, ...]] = set()

        class VerificationTimeout(Exception):
            pass

        def check_deadline() -> None:
            if time.perf_counter() >= deadline:
                raise VerificationTimeout

        def exact_candidate() -> bool:
            nonlocal witness, numerical_seen
            check_deadline()
            if torch_dfs is None:
                prediction, margin = _prediction_and_margin(model, current)
            else:
                with torch.no_grad():
                    output = torch_dfs(
                        torch.as_tensor(
                            current[None, :],
                            dtype=torch.float64,
                            device=torch_dfs_device,
                        )
                    )[-1]
                    predicted = output.times.argmin(dim=1)
                    prediction = int(predicted.item())
                    ordered = torch.sort(output.times[0]).values
                    margin = (
                        float((ordered[1] - ordered[0]).item())
                        if ordered.numel() >= 2
                        else float("inf")
                    )
            stats.exact_forwards += 1
            if prediction == baseline_prediction:
                return False
            if torch_dfs is not None:
                replay_prediction, replay_margin = _prediction_and_margin(
                    model, current
                )
                stats.exact_forwards += 1
                if replay_prediction != prediction:
                    numerical_seen = True
                    return False
                margin = min(margin, replay_margin)
            cost = int(np.abs(shifts.astype(np.int64)).sum())
            witness = {
                "shift_vector": shifts.astype(int).tolist(),
                "sparse_shifts": {
                    str(index): int(value)
                    for index, value in enumerate(shifts)
                    if value
                },
                "l1_cost": cost,
                "perturbed_times": current.tolist(),
                "prediction": int(prediction),
                "baseline_prediction": int(baseline_prediction),
                "output_margin": margin,
            }
            return True

        def state_key(position: int, remaining: int) -> tuple[Any, ...]:
            fixed = tuple(
                (int(index), int(shifts[index]))
                for index in order[:position]
                if shifts[index]
            )
            return position, remaining, fixed

        def dfs(position: int, remaining: int) -> bool:
            nonlocal numerical_seen
            check_deadline()
            stats.nodes += 1
            stats.maximum_depth = max(stats.maximum_depth, position)

            if config.method == "bnb_coupled_cache":
                sparse = tuple(int(value) for value in shifts if value)
                if sparse in state_cache:
                    stats.state_cache_hits += 1
                else:
                    state_cache.add(sparse)

            use_bound = (
                config.method in BOUND_METHODS
                and remaining > 0
                and (stats.nodes == 1 or stats.nodes % config.bound_every == 0)
            )
            if use_bound:
                key = state_key(position, remaining)
                cached = bound_cache.get(key) if config.method == "bnb_coupled_cache" else None
                if cached is not None:
                    stats.bound_cache_hits += 1
                    if cached:
                        stats.bound_prunes += 1
                        return False
                else:
                    stats.bound_calls += 1
                    certificate = lif_bcibp_bound_node(
                        current,
                        model.weights,
                        budget=remaining,
                        shift_step=threat.shift_step,
                        input_min=threat.input_min,
                        input_max=threat.input_max,
                        mutable_indices=order[position:],
                        coupled=config.method != "bnb_uncoupled",
                        tau=model.tau,
                        threshold=model.threshold,
                        no_spike_time=model.no_spike_time,
                        time_step=config.time_step,
                        min_time_step=config.min_time_step,
                        prediction=int(baseline_prediction),
                        backend=config.backend,
                        device=config.device,
                    )
                    check_deadline()
                    if config.method == "bnb_coupled_cache":
                        bound_cache[key] = certificate.robust
                    if certificate.robust:
                        stats.bound_prunes += 1
                        return False

            if position >= len(order) or remaining == 0:
                return False
            index = order[position]
            original_time = current[index]
            for shift, cost, changed_time in all_options[index]:
                if shift == 0 or cost > remaining:
                    continue
                current[index] = changed_time
                shifts[index] = shift
                if exact_candidate() or dfs(position + 1, remaining - cost):
                    current[index] = original_time
                    shifts[index] = 0
                    return True
                current[index] = original_time
                shifts[index] = 0
            return dfs(position + 1, remaining)

        try:
            found = dfs(0, threat.budget)
        except VerificationTimeout:
            return LIFVerificationResult(
                "timeout",
                int(baseline_prediction),
                config.method,
                threat.budget,
                threat.shift_step,
                time.perf_counter() - started,
                stats,
                reason="per-sample timeout",
            )
        if found:
            return LIFVerificationResult(
                "not_robust",
                int(baseline_prediction),
                config.method,
                threat.budget,
                threat.shift_step,
                time.perf_counter() - started,
                stats,
                witness=witness,
                reason="exact adversarial shift replayed",
            )
        if numerical_seen:
            return LIFVerificationResult(
                "numerical_unknown",
                int(baseline_prediction),
                config.method,
                threat.budget,
                threat.shift_step,
                time.perf_counter() - started,
                stats,
                reason="only numerically tied prediction changes were encountered",
            )
        return LIFVerificationResult(
            "robust",
            int(baseline_prediction),
            config.method,
            threat.budget,
            threat.shift_step,
            time.perf_counter() - started,
            stats,
            reason="complete finite shift tree exhausted or soundly pruned",
        )
    except Exception as error:
        return LIFVerificationResult(
            "error",
            int(baseline_prediction if baseline_prediction is not None else -1),
            config.method,
            threat.budget,
            threat.shift_step,
            time.perf_counter() - started,
            stats,
            reason="verification raised an exception",
            error=f"{type(error).__name__}: {error}",
        )
