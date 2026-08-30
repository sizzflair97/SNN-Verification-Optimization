"""Continuous-time first-spike LIF layers following Goeltz et al. (2021).

This module implements the ``tau_m == tau_s`` current-based LIF case from

    Fast and energy-efficient neuromorphic deep learning with first-spike
    times, Nature Machine Intelligence 3, 823--835 (2021).

For a causal set C of presynaptic spikes, the membrane voltage is a sum of
alpha-shaped postsynaptic potentials.  The first threshold crossing has the
closed form in Eq. (19) of the paper and uses the principal real branch of
the Lambert W function.  The implementation below is made exclusively from
PyTorch operations, so autograd gives the exact, piecewise derivatives in
Eqs. (36)--(40) away from causal-set boundaries and tangencies.

This is deliberately separate from ``utils/mnist_net.py``.  That verifier
implements discrete-time cumulative IF dynamics; weights produced by this
module are not compatible with it until a matching LIF verifier is added.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def lambertw0(z: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    """Principal real Lambert-W branch on ``[-1/e, 0)``.

    A branch-point series supplies the initial value near ``-1/e`` and
    Halley's method performs the refinement.  Keeping the iterations in the
    PyTorch graph lets autograd differentiate through W without SciPy or a
    custom backward implementation.
    """
    finfo = torch.finfo(z.dtype)
    branch = -1.0 / math.e
    z_safe = z.clamp(min=branch + 16 * finfo.eps, max=-16 * finfo.eps)

    q = (1.0 + math.e * z_safe).clamp_min(0.0)
    p = torch.sqrt(2.0 * q)
    near_branch = -1.0 + p - p.square() / 3.0 + 11.0 * p.pow(3) / 72.0
    near_zero = z_safe - z_safe.square() + 1.5 * z_safe.pow(3)
    w = torch.where(z_safe < -0.25, near_branch, near_zero)

    for _ in range(iterations):
        ew = torch.exp(w)
        f = w * ew - z_safe
        wp1 = w + 1.0
        denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1)
        denom = torch.where(
            denom.abs() < 32 * finfo.eps,
            torch.full_like(denom, 32 * finfo.eps),
            denom,
        )
        w = w - f / denom
    return w


@dataclass
class SpikeTimeLayerResult:
    """Output of a :class:`GoeltzLIFLayer`."""

    times: torch.Tensor
    spiked: torch.Tensor


class GoeltzLIFLayer(nn.Module):
    """A fully-connected CuBa-LIF first-spike layer.

    Parameters are normalized as in the software MNIST experiment in the
    paper: ``g_leak = threshold = tau_syn = tau_mem = 1`` by default.  Only
    one spike per presynaptic neuron is relevant.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tau_syn: float = 1.0,
        g_leak: float = 1.0,
        threshold: float = 1.0,
        weight_mean: float = 0.05,
        weight_std: float = 0.8,
        no_spike_time: float = 10.0,
    ) -> None:
        super().__init__()
        if tau_syn <= 0 or g_leak <= 0 or threshold <= 0:
            raise ValueError("tau_syn, g_leak, and threshold must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.tau_syn = float(tau_syn)
        self.g_leak = float(g_leak)
        self.threshold = float(threshold)
        self.no_spike_time = float(no_spike_time)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=weight_mean, std=weight_std)

    def forward(
        self,
        input_times: torch.Tensor,
        input_spiked: torch.Tensor | None = None,
    ) -> SpikeTimeLayerResult:
        """Calculate the first threshold crossing for every output neuron.

        ``input_times`` has shape ``(batch, in_features)``.  ``input_spiked``
        marks which entries are actual spikes; absent spikes are excluded
        from every candidate causal set.
        """
        if input_times.ndim != 2 or input_times.shape[1] != self.in_features:
            raise ValueError(
                f"expected input_times shape (batch, {self.in_features}), "
                f"got {tuple(input_times.shape)}"
            )
        if input_spiked is None:
            input_spiked = torch.ones_like(input_times, dtype=torch.bool)
        elif input_spiked.shape != input_times.shape:
            raise ValueError("input_spiked must have the same shape as input_times")

        batch, n_pre = input_times.shape
        dtype = input_times.dtype
        device = input_times.device
        finfo = torch.finfo(dtype)

        # Inactive inputs sort to the end but use a harmless finite value in
        # exponentials.  Their gathered weights are multiplied by zero.
        inf = torch.full((), float("inf"), dtype=dtype, device=device)
        sort_key = torch.where(input_spiked, input_times, inf)
        _, order = torch.sort(sort_key, dim=1)
        sorted_times = input_times.gather(1, order)
        sorted_active = input_spiked.gather(1, order)
        safe_times = torch.where(sorted_active, sorted_times, torch.zeros_like(sorted_times))

        gather_index = order[:, None, :].expand(batch, self.out_features, n_pre)
        sorted_weights = self.weight[None, :, :].expand(batch, -1, -1).gather(2, gather_index)
        sorted_weights = sorted_weights * sorted_active[:, None, :].to(dtype)

        scaled_t = safe_times / self.tau_syn
        exp_t = torch.exp(scaled_t)
        a1 = torch.cumsum(sorted_weights * exp_t[:, None, :], dim=2)
        b = torch.cumsum(sorted_weights * (scaled_t * exp_t)[:, None, :], dim=2)

        stable_a = a1.abs() > 64 * finfo.eps
        safe_a = torch.where(stable_a, a1, torch.ones_like(a1))
        ratio = b / safe_a
        # Extreme ratios only occur for invalid/nearly-cancelling causal sets;
        # clamping prevents them from contaminating the graph with inf/nan.
        z = -(self.g_leak * self.threshold / safe_a) * torch.exp(ratio.clamp(-40.0, 40.0))
        domain = (z >= (-1.0 / math.e)) & (z < 0.0) & (a1 > 0.0) & stable_a
        w0 = lambertw0(z)
        candidates = self.tau_syn * (ratio - w0)

        last_input = sorted_times[:, None, :]
        next_input = torch.cat(
            (sorted_times[:, 1:], torch.full_like(sorted_times[:, :1], float("inf"))),
            dim=1,
        )[:, None, :]
        causal = (candidates > last_input) & (candidates <= next_input)
        valid = domain & causal & sorted_active[:, None, :] & torch.isfinite(candidates)

        masked = torch.where(valid, candidates, torch.full_like(candidates, float("inf")))
        first_time = masked.min(dim=2).values
        spiked = torch.isfinite(first_time)
        times = torch.where(
            spiked,
            first_time,
            torch.full_like(first_time, self.no_spike_time),
        )
        return SpikeTimeLayerResult(times=times, spiked=spiked)


class GoeltzLIFNetwork(nn.Module):
    """Feed-forward hierarchy of exact first-spike LIF layers."""

    def __init__(
        self,
        dims: list[int] | tuple[int, ...],
        *,
        tau_syn: float = 1.0,
        threshold: float = 1.0,
        no_spike_time: float = 10.0,
        weight_means: tuple[float, float] = (0.05, 0.15),
        weight_std: float = 0.8,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output sizes")
        layers = []
        for index, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(
                GoeltzLIFLayer(
                    n_in,
                    n_out,
                    tau_syn=tau_syn,
                    threshold=threshold,
                    no_spike_time=no_spike_time,
                    weight_mean=weight_means[0 if index == 0 else 1],
                    weight_std=weight_std,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.dims = tuple(dims)
        self.tau_syn = float(tau_syn)

    def forward(self, input_times: torch.Tensor) -> list[SpikeTimeLayerResult]:
        results: list[SpikeTimeLayerResult] = []
        times = input_times
        active = torch.ones_like(times, dtype=torch.bool)
        for layer in self.layers:
            result = layer(times, active)
            results.append(result)
            times, active = result.times, result.spiked
        return results

