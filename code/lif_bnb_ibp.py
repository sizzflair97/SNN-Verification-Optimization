"""Sound BC-IBP bounds for reset-free first-spike Goeltz LIF networks.

The supported architecture is an arbitrary-depth feed-forward network with
continuous-time alpha PSPs and one relevant spike per neuron. Input perturbations are finite
spike-time shifts ``d * shift_step`` with ``sum(abs(d)) <= budget``.

Unlike the cumulative-IF verifier, a LIF PSP is not monotone in time.  This
module therefore bounds the complete PSP over each continuous time cell.  A
multiple-choice knapsack selects exactly one shift option per input while
respecting the shared L1 budget.  The output layer uses ordinary interval
propagation over the resulting hidden first-spike intervals.

Returning ``robust=True`` is a sound sufficient certificate.  ``False`` means
unknown, not necessarily adversarial.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy remains the reference backend.
    torch = None

from utils.lif_ttfs_net import lif_ttfs_forward


Array = np.ndarray


@dataclass(frozen=True)
class LIFSpikeBounds:
    """First-spike interval for one layer.

    ``earliest`` is a lower bound on every possible finite spike time.
    ``latest`` is an upper bound when a spike is guaranteed; infinity means
    that the neuron may remain silent. ``possible_latest`` bounds any finite
    spike even when silence is possible.
    """

    earliest: Array
    latest: Array
    possible_latest: Array
    tail_cap: float


@dataclass(frozen=True)
class LIFBCIBPResult:
    robust: bool
    prediction: int
    hidden: LIFSpikeBounds
    output: LIFSpikeBounds
    budget: int
    shift_step: float
    reason: str
    layers: tuple[LIFSpikeBounds, ...] = ()


def canonical_shift_options(
    input_time: float,
    budget: int,
    shift_step: float,
    input_min: float,
    input_max: float,
) -> tuple[tuple[int, int, float], ...]:
    """Return unique ``(shift, cost, time)`` options for one input.

    Clipping can make several integer shifts denote the same perturbed time.
    Keeping only the cheapest representative preserves the threat set while
    preventing duplicate BnB leaves and duplicate MCKP choices.
    """
    if budget < 0 or shift_step <= 0.0 or input_min > input_max:
        raise ValueError("invalid shift domain")
    unique: dict[bytes, tuple[int, int, float]] = {}
    for shift in range(-budget, budget + 1):
        time = float(np.clip(input_time + shift_step * shift, input_min, input_max))
        option = (int(shift), abs(int(shift)), time)
        key = np.float64(time).tobytes()
        previous = unique.get(key)
        if previous is None or (option[1], abs(option[0]), option[0]) < (
            previous[1], abs(previous[0]), previous[0]
        ):
            unique[key] = option
    return tuple(sorted(unique.values(), key=lambda item: (item[1], item[0])))


def _alpha_value(age: Array | float, tau: float) -> Array:
    age_arr = np.asarray(age, dtype=np.float64)
    scaled = age_arr / tau
    return np.where(age_arr >= 0.0, scaled * np.exp(-scaled), 0.0)


def alpha_psp_interval(age_lo: Array, age_hi: Array, tau: float = 1.0) -> tuple[Array, Array]:
    """Exact range of ``(age/tau) exp(-age/tau)`` on ``[age_lo, age_hi]``.

    The only interior extremum is the maximum at ``age=tau``.  Returned
    floating-point bounds are rounded outwards by one representable value.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    lo = np.asarray(age_lo, dtype=np.float64)
    hi = np.asarray(age_hi, dtype=np.float64)
    lo, hi = np.broadcast_arrays(lo, hi)
    if np.any(lo > hi):
        raise ValueError("age_lo must not exceed age_hi")

    causal_hi = np.maximum(hi, 0.0)
    causal_lo = np.maximum(lo, 0.0)
    value_lo = _alpha_value(causal_lo, tau)
    value_hi = _alpha_value(causal_hi, tau)
    maximum = np.maximum(value_lo, value_hi)
    includes_peak = (causal_lo <= tau) & (causal_hi >= tau) & (hi >= 0.0)
    maximum = np.where(includes_peak, 1.0 / math.e, maximum)
    maximum = np.where(hi < 0.0, 0.0, maximum)

    minimum = np.minimum(value_lo, value_hi)
    minimum = np.where((lo <= 0.0) | (hi < 0.0), 0.0, minimum)
    return np.nextafter(minimum, -np.inf), np.nextafter(maximum, np.inf)


def _weighted_bounds(weights: Array, kernel_lo: Array, kernel_hi: Array) -> tuple[Array, Array]:
    positive = weights >= 0.0
    lower = np.where(positive, weights * kernel_lo, weights * kernel_hi)
    upper = np.where(positive, weights * kernel_hi, weights * kernel_lo)
    return lower, upper


def input_voltage_bounds_mckp(
    weights: Array,
    input_times: Array,
    *,
    cell_start: float,
    cell_end: float,
    budget: int,
    shift_step: float,
    input_min: float,
    input_max: float,
    tau: float = 1.0,
) -> tuple[Array, Array]:
    """Budget-coupled voltage bounds for a first LIF layer on one time cell.

    This is a multiple-choice knapsack: input ``p`` chooses one integer shift
    in ``[-budget, budget]`` (including zero), with cost ``abs(shift)``.
    The DP is vectorized over all postsynaptic neurons.
    """
    weights = np.asarray(weights, dtype=np.float64)
    input_times = np.asarray(input_times, dtype=np.float64).reshape(-1)
    if weights.ndim != 2 or weights.shape[1] != input_times.size:
        raise ValueError("weights must have shape (n_post, n_inputs)")
    if budget < 0 or shift_step <= 0 or input_min > input_max:
        raise ValueError("invalid budget, shift_step, or input range")
    if cell_start > cell_end:
        raise ValueError("cell_start must not exceed cell_end")

    shifts = np.arange(-budget, budget + 1, dtype=np.int32)
    costs = np.abs(shifts)
    option_times = np.clip(
        input_times[:, None] + shift_step * shifts[None, :], input_min, input_max
    )
    kernel_lo, kernel_hi = alpha_psp_interval(
        cell_start - option_times, cell_end - option_times, tau
    )

    n_post = weights.shape[0]
    neg_inf = np.full((budget + 1, n_post), -np.inf, dtype=np.float64)
    pos_inf = np.full((budget + 1, n_post), np.inf, dtype=np.float64)
    dp_upper = neg_inf.copy()
    dp_lower = pos_inf.copy()
    dp_upper[0] = 0.0
    dp_lower[0] = 0.0

    for p in range(input_times.size):
        next_upper = neg_inf.copy()
        next_lower = pos_inf.copy()
        w = weights[:, p]
        option_lower, option_upper = _weighted_bounds(
            w[:, None], kernel_lo[p][None, :], kernel_hi[p][None, :]
        )
        # Clipping can map several shifts to the same time. Keeping them is
        # harmless; the DP naturally rejects a duplicate with larger cost.
        for option, cost in enumerate(costs):
            c = int(cost)
            if c > budget:
                continue
            for used in range(c, budget + 1):
                prev = used - c
                np.maximum(
                    next_upper[used], dp_upper[prev] + option_upper[:, option],
                    out=next_upper[used],
                )
                np.minimum(
                    next_lower[used], dp_lower[prev] + option_lower[:, option],
                    out=next_lower[used],
                )
        dp_upper, dp_lower = next_upper, next_lower

    lower = np.min(dp_lower, axis=0)
    upper = np.max(dp_upper, axis=0)
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _tail_cap(weights: Array, latest_presynaptic: float, threshold: float, tau: float) -> float:
    """Find a time after which even the positive-only PSP sum is subthreshold."""
    positive_sum = np.maximum(np.asarray(weights, dtype=np.float64), 0.0).sum(axis=1)
    if not np.any(positive_sum > 0.0):
        return latest_presynaptic + tau
    age = tau
    for _ in range(80):
        upper = positive_sum * float(_alpha_value(age, tau))
        if np.all(np.nextafter(upper, np.inf) < threshold):
            return latest_presynaptic + age
        age *= 1.35
    raise RuntimeError("failed to establish a finite LIF tail cap")


def _extract_spike_bounds(
    n_neurons: int,
    start_time: float,
    tail_cap: float,
    time_step: float,
    threshold: float,
    interval_bound: Callable[[float, float], tuple[Array, Array]],
) -> LIFSpikeBounds:
    """Convert continuous voltage envelopes into sound first-spike intervals."""
    if time_step <= 0 or tail_cap < start_time:
        raise ValueError("invalid time partition")
    count = max(1, int(math.ceil((tail_cap - start_time) / time_step)))
    grid = np.linspace(start_time, tail_cap, count + 1, dtype=np.float64)
    earliest = np.full(n_neurons, np.inf, dtype=np.float64)
    latest = np.full(n_neurons, np.inf, dtype=np.float64)

    for left, right in zip(grid[:-1], grid[1:]):
        _, cell_upper = interval_bound(float(left), float(right))
        can_cross = np.isinf(earliest) & (cell_upper >= threshold)
        earliest[can_cross] = left

        point_lower, _ = interval_bound(float(right), float(right))
        # Strict comparison is intentionally conservative under floating point.
        must_have_crossed = np.isinf(latest) & (point_lower > threshold)
        latest[must_have_crossed] = right

    possible_latest = np.where(np.isfinite(earliest), tail_cap, np.inf)
    return LIFSpikeBounds(earliest, latest, possible_latest, float(tail_cap))


def _output_voltage_bounds(
    weights: Array,
    presynaptic: LIFSpikeBounds,
    cell_start: float,
    cell_end: float,
    tau: float,
) -> tuple[Array, Array]:
    """Sign-aware output voltage bounds from hidden first-spike intervals."""
    weights = np.asarray(weights, dtype=np.float64)
    n_pre = weights.shape[1]
    total_lower = np.zeros(weights.shape[0], dtype=np.float64)
    total_upper = np.zeros(weights.shape[0], dtype=np.float64)

    for p in range(n_pre):
        spike_lo = presynaptic.earliest[p]
        if not np.isfinite(spike_lo):
            continue  # this hidden neuron can never spike
        guaranteed = np.isfinite(presynaptic.latest[p])
        spike_hi = presynaptic.latest[p] if guaranteed else presynaptic.possible_latest[p]
        kernel_lo, kernel_hi = alpha_psp_interval(
            np.asarray(cell_start - spike_hi), np.asarray(cell_end - spike_lo), tau
        )
        if not guaranteed:
            kernel_lo = np.asarray(0.0)  # silence is also feasible
        lower, upper = _weighted_bounds(weights[:, p], kernel_lo, kernel_hi)
        total_lower += lower
        total_upper += upper

    return np.nextafter(total_lower, -np.inf), np.nextafter(total_upper, np.inf)


def _alpha_psp_interval_torch(age_lo, age_hi, tau: float):
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is unavailable")
    lo, hi = torch.broadcast_tensors(age_lo, age_hi)
    causal_lo = torch.clamp_min(lo, 0.0)
    causal_hi = torch.clamp_min(hi, 0.0)
    scaled_lo = causal_lo / tau
    scaled_hi = causal_hi / tau
    value_lo = scaled_lo * torch.exp(-scaled_lo)
    value_hi = scaled_hi * torch.exp(-scaled_hi)
    maximum = torch.maximum(value_lo, value_hi)
    includes_peak = (causal_lo <= tau) & (causal_hi >= tau) & (hi >= 0.0)
    maximum = torch.where(includes_peak, torch.full_like(maximum, 1.0 / math.e), maximum)
    maximum = torch.where(hi < 0.0, torch.zeros_like(maximum), maximum)
    minimum = torch.minimum(value_lo, value_hi)
    minimum = torch.where((lo <= 0.0) | (hi < 0.0), torch.zeros_like(minimum), minimum)
    neg_inf = torch.full_like(minimum, -torch.inf)
    pos_inf = torch.full_like(maximum, torch.inf)
    return torch.nextafter(minimum, neg_inf), torch.nextafter(maximum, pos_inf)


def input_voltage_bounds_cells(
    weights: Array,
    input_times: Array,
    *,
    cell_starts: Array,
    cell_ends: Array,
    budget: int,
    shift_step: float,
    input_min: float,
    input_max: float,
    mutable_indices: Sequence[int] | Array | None = None,
    coupled: bool = True,
    tau: float = 1.0,
    backend: str = "numpy",
    device: str | None = None,
) -> tuple[Array, Array]:
    """Bound first-layer voltages for many time cells at once."""
    weights = np.asarray(weights, dtype=np.float64)
    input_times = np.asarray(input_times, dtype=np.float64).reshape(-1)
    starts = np.asarray(cell_starts, dtype=np.float64).reshape(-1)
    ends = np.asarray(cell_ends, dtype=np.float64).reshape(-1)
    if weights.ndim != 2 or weights.shape[1] != input_times.size:
        raise ValueError("weights must have shape (n_post, n_inputs)")
    if starts.shape != ends.shape or np.any(starts > ends):
        raise ValueError("invalid time cells")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    mutable = np.ones(input_times.size, dtype=bool)
    if mutable_indices is not None:
        mutable[:] = False
        mutable[np.asarray(mutable_indices, dtype=np.int64)] = True

    options: list[tuple[tuple[int, int, float], ...]] = []
    for index, value in enumerate(input_times):
        if mutable[index] and budget:
            options.append(
                canonical_shift_options(value, budget, shift_step, input_min, input_max)
            )
        else:
            options.append(((0, 0, float(value)),))

    if backend not in {"numpy", "torch"}:
        raise ValueError("backend must be numpy or torch")
    if backend == "torch":
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is unavailable")
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w_all = torch.as_tensor(weights, dtype=torch.float64, device=target)
        s_all = torch.as_tensor(starts, dtype=torch.float64, device=target)
        e_all = torch.as_tensor(ends, dtype=torch.float64, device=target)
        n_cells, n_post = starts.size, weights.shape[0]
        if coupled:
            dp_lo = torch.full((n_cells, budget + 1, n_post), torch.inf,
                               dtype=torch.float64, device=target)
            dp_hi = torch.full((n_cells, budget + 1, n_post), -torch.inf,
                               dtype=torch.float64, device=target)
            dp_lo[:, 0] = 0.0
            dp_hi[:, 0] = 0.0
        else:
            total_lo = torch.zeros((n_cells, n_post), dtype=torch.float64, device=target)
            total_hi = torch.zeros_like(total_lo)

        for index, choices in enumerate(options):
            costs = sorted({choice[1] for choice in choices})
            times_t = torch.as_tensor([choice[2] for choice in choices],
                                      dtype=torch.float64, device=target)
            k_lo, k_hi = _alpha_psp_interval_torch(
                s_all[:, None] - times_t[None, :],
                e_all[:, None] - times_t[None, :],
                tau,
            )
            weight = w_all[:, index][None, None, :]
            opt_lo = torch.where(weight >= 0.0, weight * k_lo[:, :, None],
                                 weight * k_hi[:, :, None])
            opt_hi = torch.where(weight >= 0.0, weight * k_hi[:, :, None],
                                 weight * k_lo[:, :, None])
            by_cost = {}
            choice_costs = torch.as_tensor([choice[1] for choice in choices],
                                           dtype=torch.int64, device=target)
            for cost in costs:
                mask = choice_costs == cost
                by_cost[cost] = (
                    opt_lo[:, mask].amin(dim=1),
                    opt_hi[:, mask].amax(dim=1),
                )
            if coupled:
                next_lo = torch.full_like(dp_lo, torch.inf)
                next_hi = torch.full_like(dp_hi, -torch.inf)
                for cost, (contrib_lo, contrib_hi) in by_cost.items():
                    if cost > budget:
                        continue
                    candidate_lo = dp_lo[:, : budget + 1 - cost] + contrib_lo[:, None]
                    candidate_hi = dp_hi[:, : budget + 1 - cost] + contrib_hi[:, None]
                    next_lo[:, cost:] = torch.minimum(next_lo[:, cost:], candidate_lo)
                    next_hi[:, cost:] = torch.maximum(next_hi[:, cost:], candidate_hi)
                dp_lo, dp_hi = next_lo, next_hi
            else:
                total_lo += torch.stack([item[0] for item in by_cost.values()]).amin(dim=0)
                total_hi += torch.stack([item[1] for item in by_cost.values()]).amax(dim=0)

        if coupled:
            result_lo = dp_lo.amin(dim=1)
            result_hi = dp_hi.amax(dim=1)
        else:
            result_lo, result_hi = total_lo, total_hi
        result_lo = torch.nextafter(result_lo, torch.full_like(result_lo, -torch.inf))
        result_hi = torch.nextafter(result_hi, torch.full_like(result_hi, torch.inf))
        return result_lo.cpu().numpy(), result_hi.cpu().numpy()

    n_cells, n_post = starts.size, weights.shape[0]
    if coupled:
        dp_lo = np.full((n_cells, budget + 1, n_post), np.inf, dtype=np.float64)
        dp_hi = np.full((n_cells, budget + 1, n_post), -np.inf, dtype=np.float64)
        dp_lo[:, 0] = 0.0
        dp_hi[:, 0] = 0.0
    else:
        total_lo = np.zeros((n_cells, n_post), dtype=np.float64)
        total_hi = np.zeros_like(total_lo)

    for index, choices in enumerate(options):
        choice_times = np.asarray([choice[2] for choice in choices], dtype=np.float64)
        choice_costs = np.asarray([choice[1] for choice in choices], dtype=np.int32)
        kernel_lo, kernel_hi = alpha_psp_interval(
            starts[:, None] - choice_times[None, :],
            ends[:, None] - choice_times[None, :],
            tau,
        )
        weight = weights[:, index][None, None, :]
        opt_lo = np.where(weight >= 0.0, weight * kernel_lo[:, :, None],
                          weight * kernel_hi[:, :, None])
        opt_hi = np.where(weight >= 0.0, weight * kernel_hi[:, :, None],
                          weight * kernel_lo[:, :, None])
        by_cost = {
            int(cost): (
                np.min(opt_lo[:, choice_costs == cost], axis=1),
                np.max(opt_hi[:, choice_costs == cost], axis=1),
            )
            for cost in np.unique(choice_costs)
        }
        if coupled:
            next_lo = np.full_like(dp_lo, np.inf)
            next_hi = np.full_like(dp_hi, -np.inf)
            for cost, (contrib_lo, contrib_hi) in by_cost.items():
                if cost > budget:
                    continue
                candidate_lo = dp_lo[:, : budget + 1 - cost] + contrib_lo[:, None]
                candidate_hi = dp_hi[:, : budget + 1 - cost] + contrib_hi[:, None]
                next_lo[:, cost:] = np.minimum(next_lo[:, cost:], candidate_lo)
                next_hi[:, cost:] = np.maximum(next_hi[:, cost:], candidate_hi)
            dp_lo, dp_hi = next_lo, next_hi
        else:
            total_lo += np.min(np.stack([item[0] for item in by_cost.values()]), axis=0)
            total_hi += np.max(np.stack([item[1] for item in by_cost.values()]), axis=0)

    if coupled:
        result_lo = np.min(dp_lo, axis=1)
        result_hi = np.max(dp_hi, axis=1)
    else:
        result_lo, result_hi = total_lo, total_hi
    return np.nextafter(result_lo, -np.inf), np.nextafter(result_hi, np.inf)


def _extract_spike_bounds_batched(
    n_neurons: int,
    start_time: float,
    tail_cap: float,
    time_step: float,
    min_time_step: float,
    threshold: float,
    interval_bound: Callable[[Array, Array], tuple[Array, Array]],
) -> LIFSpikeBounds:
    """Convert batched sound voltage cells into first-spike bounds."""
    if time_step <= 0.0 or min_time_step <= 0.0 or min_time_step > time_step:
        raise ValueError("invalid adaptive time partition")
    count = max(1, int(math.ceil((tail_cap - start_time) / time_step)))
    grid = np.linspace(start_time, tail_cap, count + 1, dtype=np.float64)
    starts, ends = grid[:-1], grid[1:]

    while np.any((ends - starts) > min_time_step * (1.0 + 1e-12)):
        refinable = (ends - starts) > min_time_step * (1.0 + 1e-12)
        _, upper = interval_bound(starts, ends)
        split = refinable & np.any(upper >= threshold, axis=1)
        if not np.any(split):
            break
        new_starts: list[float] = []
        new_ends: list[float] = []
        for left, right, do_split in zip(starts, ends, split):
            if do_split:
                middle = (float(left) + float(right)) / 2.0
                new_starts.extend((float(left), middle))
                new_ends.extend((middle, float(right)))
            else:
                new_starts.append(float(left))
                new_ends.append(float(right))
        starts = np.asarray(new_starts, dtype=np.float64)
        ends = np.asarray(new_ends, dtype=np.float64)

    cell_count = starts.size
    combined_lower, combined_upper = interval_bound(
        np.concatenate((starts, ends)),
        np.concatenate((ends, ends)),
    )
    upper = combined_upper[:cell_count]
    point_lower = combined_lower[cell_count:]
    earliest = np.full(n_neurons, np.inf, dtype=np.float64)
    latest = np.full(n_neurons, np.inf, dtype=np.float64)
    for neuron in range(n_neurons):
        possible = np.flatnonzero(upper[:, neuron] >= threshold)
        if possible.size:
            earliest[neuron] = starts[possible[0]]
        guaranteed = np.flatnonzero(point_lower[:, neuron] > threshold)
        if guaranteed.size:
            latest[neuron] = ends[guaranteed[0]]
    possible_latest = np.where(np.isfinite(earliest), tail_cap, np.inf)
    return LIFSpikeBounds(earliest, latest, possible_latest, float(tail_cap))


def _output_voltage_bounds_cells(
    weights: Array,
    presynaptic: LIFSpikeBounds,
    cell_starts: Array,
    cell_ends: Array,
    tau: float,
) -> tuple[Array, Array]:
    weights = np.asarray(weights, dtype=np.float64)
    starts = np.asarray(cell_starts, dtype=np.float64).reshape(-1)
    ends = np.asarray(cell_ends, dtype=np.float64).reshape(-1)
    lower = np.zeros((starts.size, weights.shape[0]), dtype=np.float64)
    upper = np.zeros_like(lower)
    for index in range(weights.shape[1]):
        spike_lo = presynaptic.earliest[index]
        if not np.isfinite(spike_lo):
            continue
        guaranteed = np.isfinite(presynaptic.latest[index])
        spike_hi = (
            presynaptic.latest[index]
            if guaranteed
            else presynaptic.possible_latest[index]
        )
        kernel_lo, kernel_hi = alpha_psp_interval(
            starts - spike_hi, ends - spike_lo, tau
        )
        if not guaranteed:
            kernel_lo = np.zeros_like(kernel_lo)
        weight = weights[:, index][None, :]
        lower += np.where(weight >= 0.0, weight * kernel_lo[:, None],
                          weight * kernel_hi[:, None])
        upper += np.where(weight >= 0.0, weight * kernel_hi[:, None],
                          weight * kernel_lo[:, None])
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _classification_is_separated(output: LIFSpikeBounds, prediction: int) -> tuple[bool, str]:
    if prediction < 0 or prediction >= output.earliest.size:
        return False, "baseline has no valid output spike"
    target_latest = output.latest[prediction]
    if not np.isfinite(target_latest):
        return False, "target output is not guaranteed to spike"
    for competitor, competitor_earliest in enumerate(output.earliest):
        if competitor == prediction:
            continue
        safe = (
            competitor_earliest > target_latest
            if competitor < prediction
            else competitor_earliest >= target_latest
        )
        if not safe:
            return False, f"class {competitor} may tie or precede the target"
    return True, "all competitor first-spike bounds are separated"


def lif_bcibp_bound_node(
    input_times: Array,
    weights_list: Sequence[Array],
    *,
    budget: int,
    shift_step: float,
    input_min: float,
    input_max: float,
    mutable_indices: Sequence[int] | Array | None = None,
    coupled: bool = True,
    tau: float = 1.0,
    threshold: float = 1.0,
    time_step: float = 0.05,
    min_time_step: float | None = None,
    prediction: int | None = None,
    backend: str = "numpy",
    device: str | None = None,
) -> LIFBCIBPResult:
    """Sound subtree certificate for an arbitrary-depth feed-forward LIF SNN."""
    if not weights_list:
        raise ValueError("weights_list must not be empty")
    times = np.asarray(input_times, dtype=np.float64).reshape(-1)
    weights = [np.asarray(value, dtype=np.float64) for value in weights_list]
    previous_width = times.size
    for value in weights:
        if value.ndim != 2 or value.shape[1] != previous_width:
            raise ValueError("incompatible LIF network shapes")
        previous_width = value.shape[0]
    if prediction is None:
        prediction = lif_ttfs_forward(weights, times, tau_syn=tau, threshold=threshold)
    if min_time_step is None:
        min_time_step = time_step

    first_tail = _tail_cap(weights[0], input_max, threshold, tau)

    def first_bound(starts: Array, ends: Array) -> tuple[Array, Array]:
        return input_voltage_bounds_cells(
            weights[0],
            times,
            cell_starts=starts,
            cell_ends=ends,
            budget=budget,
            shift_step=shift_step,
            input_min=input_min,
            input_max=input_max,
            mutable_indices=mutable_indices,
            coupled=coupled,
            tau=tau,
            backend=backend,
            device=device,
        )

    layer_bounds: list[LIFSpikeBounds] = [
        _extract_spike_bounds_batched(
            weights[0].shape[0],
            input_min,
            first_tail,
            time_step,
            min_time_step,
            threshold,
            first_bound,
        )
    ]

    for layer_weights in weights[1:]:
        previous = layer_bounds[-1]
        finite = previous.earliest[np.isfinite(previous.earliest)]
        if not finite.size:
            silent = np.full(layer_weights.shape[0], np.inf, dtype=np.float64)
            layer_bounds.append(
                LIFSpikeBounds(silent.copy(), silent.copy(), silent.copy(), previous.tail_cap)
            )
            continue
        layer_start = float(np.min(finite))
        layer_tail = _tail_cap(layer_weights, previous.tail_cap, threshold, tau)

        def layer_bound(starts: Array, ends: Array, w=layer_weights, pre=previous):
            return _output_voltage_bounds_cells(w, pre, starts, ends, tau)

        layer_bounds.append(
            _extract_spike_bounds_batched(
                layer_weights.shape[0],
                layer_start,
                layer_tail,
                time_step,
                min_time_step,
                threshold,
                layer_bound,
            )
        )

    robust, reason = _classification_is_separated(layer_bounds[-1], int(prediction))
    return LIFBCIBPResult(
        robust=robust,
        prediction=int(prediction),
        hidden=layer_bounds[0],
        output=layer_bounds[-1],
        budget=int(budget),
        shift_step=float(shift_step),
        reason=reason,
        layers=tuple(layer_bounds),
    )
def _legacy_lif_bcibp_prove_robust(
    input_times: Array,
    weights_list: Sequence[Array],
    *,
    budget: int,
    shift_step: float,
    input_min: float,
    input_max: float,
    tau: float = 1.0,
    threshold: float = 1.0,
    time_step: float = 0.05,
    prediction: int | None = None,
) -> LIFBCIBPResult:
    """Run sound LIF BC-IBP for an input-hidden-output TTFS network."""
    if len(weights_list) != 2:
        raise NotImplementedError("the initial LIF BC-IBP supports one hidden layer")
    input_times = np.asarray(input_times, dtype=np.float64).reshape(-1)
    w_hidden = np.asarray(weights_list[0], dtype=np.float64)
    w_output = np.asarray(weights_list[1], dtype=np.float64)
    if w_hidden.shape[1] != input_times.size or w_output.shape[1] != w_hidden.shape[0]:
        raise ValueError("incompatible LIF network shapes")
    if prediction is None:
        prediction = lif_ttfs_forward(
            weights_list, input_times, tau_syn=tau, threshold=threshold
        )

    latest_input = float(input_max)
    hidden_tail = _tail_cap(w_hidden, latest_input, threshold, tau)

    def hidden_bound(left: float, right: float) -> tuple[Array, Array]:
        return input_voltage_bounds_mckp(
            w_hidden, input_times, cell_start=left, cell_end=right,
            budget=budget, shift_step=shift_step, input_min=input_min,
            input_max=input_max, tau=tau,
        )

    hidden = _extract_spike_bounds(
        w_hidden.shape[0], input_min, hidden_tail, time_step, threshold, hidden_bound
    )

    finite_hidden = hidden.earliest[np.isfinite(hidden.earliest)]
    if finite_hidden.size == 0:
        output = LIFSpikeBounds(
            np.full(w_output.shape[0], np.inf), np.full(w_output.shape[0], np.inf),
            np.full(w_output.shape[0], np.inf), hidden_tail,
        )
        return LIFBCIBPResult(False, int(prediction), hidden, output, budget, shift_step,
                              "no output spike can be guaranteed")

    output_start = float(np.min(finite_hidden))
    output_tail = _tail_cap(w_output, hidden_tail, threshold, tau)

    def output_bound(left: float, right: float) -> tuple[Array, Array]:
        return _output_voltage_bounds(w_output, hidden, left, right, tau)

    output = _extract_spike_bounds(
        w_output.shape[0], output_start, output_tail, time_step,
        threshold, output_bound,
    )

    if prediction < 0 or prediction >= w_output.shape[0]:
        return LIFBCIBPResult(False, int(prediction), hidden, output, budget, shift_step,
                              "baseline has no valid output spike")
    target_latest = output.latest[prediction]
    if not np.isfinite(target_latest):
        return LIFBCIBPResult(False, int(prediction), hidden, output, budget, shift_step,
                              "target output is not guaranteed to spike")

    for competitor in range(w_output.shape[0]):
        if competitor == prediction:
            continue
        competitor_earliest = output.earliest[competitor]
        # NumPy argmin gives a tie to the lower output index.
        if competitor < prediction:
            safe = competitor_earliest > target_latest
        else:
            safe = competitor_earliest >= target_latest
        if not safe:
            return LIFBCIBPResult(
                False, int(prediction), hidden, output, budget, shift_step,
                f"class {competitor} may tie or precede the target",
            )
    return LIFBCIBPResult(True, int(prediction), hidden, output, budget, shift_step,
                          "all competitor first-spike bounds are separated")

def lif_bcibp_prove_robust(
    input_times: Array,
    weights_list: Sequence[Array],
    *,
    budget: int,
    shift_step: float,
    input_min: float,
    input_max: float,
    tau: float = 1.0,
    threshold: float = 1.0,
    time_step: float = 0.05,
    prediction: int | None = None,
) -> LIFBCIBPResult:
    """Backward-compatible root wrapper around the batched node verifier."""
    return lif_bcibp_bound_node(
        input_times,
        weights_list,
        budget=budget,
        shift_step=shift_step,
        input_min=input_min,
        input_max=input_max,
        mutable_indices=None,
        coupled=True,
        tau=tau,
        threshold=threshold,
        time_step=time_step,
        min_time_step=time_step,
        prediction=prediction,
    )
