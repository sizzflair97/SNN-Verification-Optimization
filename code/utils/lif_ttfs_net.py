"""Independent NumPy forward pass for Goeltz-style LIF-TTFS networks.

The dynamics match :mod:`goltz_lif_ttfs`: current-based LIF neurons with
``tau_mem == tau_syn`` and one relevant spike per neuron. For every possible
causal prefix of sorted presynaptic spikes, the first threshold crossing is
computed with the principal real Lambert-W branch (Goeltz et al., 2021,
Eq. 19). The earliest causally consistent crossing is emitted.

This module intentionally does not call the PyTorch training model. It is an
independent reference forward for checking exported weights before the same
semantics are used in a verifier.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


Array = np.ndarray


def lambertw0_real(z: Array, iterations: int = 12) -> Array:
    """Principal real Lambert-W branch for an array in ``[-1/e, 0)``."""
    z = np.asarray(z)
    if z.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        z = z.astype(np.float64)
    finfo = np.finfo(z.dtype)
    branch = -1.0 / math.e
    z_safe = np.clip(z, branch + 16 * finfo.eps, -16 * finfo.eps)

    q = np.maximum(1.0 + math.e * z_safe, 0.0)
    p = np.sqrt(2.0 * q)
    near_branch = -1.0 + p - p**2 / 3.0 + 11.0 * p**3 / 72.0
    near_zero = z_safe - z_safe**2 + 1.5 * z_safe**3
    w = np.where(z_safe < -0.25, near_branch, near_zero)

    for _ in range(iterations):
        ew = np.exp(w)
        residual = w * ew - z_safe
        wp1 = w + 1.0
        denominator = ew * wp1 - (w + 2.0) * residual / (2.0 * wp1)
        denominator = np.where(
            np.abs(denominator) < 32 * finfo.eps,
            np.asarray(32 * finfo.eps, dtype=z.dtype),
            denominator,
        )
        w = w - residual / denominator
    return w


def alpha_psp(age: Array, tau_syn: float = 1.0) -> Array:
    """Normalized alpha-shaped PSP ``(age/tau) exp(-age/tau)``."""
    age = np.asarray(age)
    scaled = age / tau_syn
    return np.where(age >= 0.0, scaled * np.exp(-scaled), 0.0)


@dataclass(frozen=True)
class LIFLayerForwardResult:
    """First-spike outputs of one LIF layer."""

    times: Array
    spiked: Array
    causal_count: Array


def lif_first_spike_layer(
    input_times: Array,
    weights: Array,
    input_spiked: Array | None = None,
    *,
    tau_syn: float = 1.0,
    g_leak: float = 1.0,
    threshold: float = 1.0,
    no_spike_time: float = 10.0,
) -> LIFLayerForwardResult:
    """Evaluate a fully-connected first-spike CuBa-LIF layer.

    ``input_times`` is ``(batch, n_pre)`` or ``(n_pre,)`` and ``weights`` is
    ``(n_post, n_pre)``. The returned ``causal_count`` is zero for silent
    neurons and otherwise records the selected causal-prefix length.
    """
    if tau_syn <= 0 or g_leak <= 0 or threshold <= 0:
        raise ValueError("tau_syn, g_leak, and threshold must be positive")

    input_times = np.asarray(input_times)
    squeeze_batch = input_times.ndim == 1
    if squeeze_batch:
        input_times = input_times[None, :]
    if input_times.ndim != 2:
        raise ValueError("input_times must have shape (batch, n_pre) or (n_pre,)")
    weights = np.asarray(weights)
    if weights.ndim != 2 or weights.shape[1] != input_times.shape[1]:
        raise ValueError(
            f"expected weights shape (n_post, {input_times.shape[1]}), got {weights.shape}"
        )

    dtype = np.result_type(input_times.dtype, weights.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    input_times = input_times.astype(dtype, copy=False)
    weights = weights.astype(dtype, copy=False)
    if input_spiked is None:
        input_spiked = np.ones(input_times.shape, dtype=bool)
    else:
        input_spiked = np.asarray(input_spiked, dtype=bool)
        if squeeze_batch and input_spiked.ndim == 1:
            input_spiked = input_spiked[None, :]
        if input_spiked.shape != input_times.shape:
            raise ValueError("input_spiked must have the same shape as input_times")

    batch, n_pre = input_times.shape
    n_post = weights.shape[0]
    finfo = np.finfo(dtype)
    sort_key = np.where(input_spiked, input_times, np.inf)
    order = np.argsort(sort_key, axis=1, kind="stable")
    sorted_times = np.take_along_axis(input_times, order, axis=1)
    sorted_active = np.take_along_axis(input_spiked, order, axis=1)
    safe_times = np.where(sorted_active, sorted_times, 0.0)

    weight_view = np.broadcast_to(weights[None, :, :], (batch, n_post, n_pre))
    sorted_weights = np.take_along_axis(weight_view, order[:, None, :], axis=2)
    sorted_weights = sorted_weights * sorted_active[:, None, :]

    scaled_time = safe_times / tau_syn
    exp_time = np.exp(scaled_time)
    a1 = np.cumsum(sorted_weights * exp_time[:, None, :], axis=2)
    b = np.cumsum(sorted_weights * (scaled_time * exp_time)[:, None, :], axis=2)

    stable_a = np.abs(a1) > 64 * finfo.eps
    safe_a = np.where(stable_a, a1, 1.0)
    ratio = b / safe_a
    z = -(g_leak * threshold / safe_a) * np.exp(np.clip(ratio, -40.0, 40.0))
    domain = (z >= (-1.0 / math.e)) & (z < 0.0) & (a1 > 0.0) & stable_a
    candidates = tau_syn * (ratio - lambertw0_real(z))

    next_input = np.concatenate(
        (sorted_times[:, 1:], np.full_like(sorted_times[:, :1], np.inf)), axis=1
    )
    causal = (candidates > sorted_times[:, None, :]) & (
        candidates <= next_input[:, None, :]
    )
    valid = domain & causal & sorted_active[:, None, :] & np.isfinite(candidates)
    masked = np.where(valid, candidates, np.inf)
    selected_index = np.argmin(masked, axis=2)
    first_time = np.take_along_axis(masked, selected_index[:, :, None], axis=2)[:, :, 0]
    spiked = np.isfinite(first_time)
    times = np.where(spiked, first_time, np.asarray(no_spike_time, dtype=dtype))
    causal_count = np.where(spiked, selected_index + 1, 0)

    if squeeze_batch:
        return LIFLayerForwardResult(times[0], spiked[0], causal_count[0])
    return LIFLayerForwardResult(times, spiked, causal_count)


def lif_ttfs_forward_batch(
    weights_list: Sequence[Array],
    input_times: Array,
    *,
    tau_syn: float = 1.0,
    g_leak: float = 1.0,
    threshold: float = 1.0,
    no_spike_time: float = 10.0,
    layer_results_return: list[LIFLayerForwardResult] | None = None,
) -> Array:
    """Evaluate a batch and return lowest-index earliest-output predictions."""
    times = np.asarray(input_times)
    squeeze_batch = times.ndim == 1
    if squeeze_batch:
        times = times[None, :]
    active = np.ones(times.shape, dtype=bool)
    results: list[LIFLayerForwardResult] = []
    for weights in weights_list:
        result = lif_first_spike_layer(
            times,
            weights,
            active,
            tau_syn=tau_syn,
            g_leak=g_leak,
            threshold=threshold,
            no_spike_time=no_spike_time,
        )
        results.append(result)
        times, active = result.times, result.spiked
    if not results:
        raise ValueError("weights_list must contain at least one layer")
    if layer_results_return is not None:
        layer_results_return[:] = results
    # Earliest *actual* output spike wins. If the complete output layer is
    # silent, return the Fast&Deep reject label instead of letting argmin pick
    # class zero from equal no-spike sentinels.
    output = results[-1]
    masked_times = np.where(output.spiked, output.times, np.inf)
    prediction = np.argmin(masked_times, axis=1)
    prediction = np.where(np.any(output.spiked, axis=1), prediction, -1)
    return prediction[0] if squeeze_batch else prediction


def lif_ttfs_forward(
    weights_list: Sequence[Array],
    input_times: Array,
    **kwargs: object,
) -> int:
    """Single-sample convenience wrapper around :func:`lif_ttfs_forward_batch`."""
    input_times = np.asarray(input_times)
    if input_times.ndim != 1:
        raise ValueError("lif_ttfs_forward expects one flattened input-time vector")
    return int(lif_ttfs_forward_batch(weights_list, input_times, **kwargs))


def load_lif_ttfs_weights(model_dir: str | os.PathLike[str], *, best: bool = True) -> list[Array]:
    """Load sequential exported NumPy weights from a Goeltz training run."""
    root = Path(model_dir)
    prefix = "weights_best" if best and (root / "weights_best_0.npy").exists() else "weights"
    weights: list[Array] = []
    index = 0
    while (path := root / f"{prefix}_{index}.npy").exists():
        weights.append(np.load(path))
        index += 1
    if not weights:
        raise FileNotFoundError(f"no {prefix}_*.npy weights found in {root}")
    return weights
