"""Numerical checks for the analytical Goeltz LIF-TTFS layer."""
from __future__ import annotations

import math

import numpy as np
import torch

from goltz_lif_ttfs import GoeltzLIFLayer, lambertw0
from utils.lif_ttfs_net import (
    lambertw0_real,
    lif_first_spike_layer,
    lif_ttfs_forward_batch,
)


def test_lambertw0_residual() -> None:
    z = torch.tensor([-1.0 / math.e + 1e-5, -0.3, -0.1, -1e-3], dtype=torch.float64)
    w = lambertw0(z, iterations=16)
    torch.testing.assert_close(w * torch.exp(w), z, rtol=1e-9, atol=1e-9)
    assert bool((w >= -1.0).all())


def test_numpy_lambertw0_residual() -> None:
    z = np.array([-1.0 / math.e + 1e-5, -0.3, -0.1, -1e-3], dtype=np.float64)
    w = lambertw0_real(z, iterations=16)
    np.testing.assert_allclose(w * np.exp(w), z, rtol=1e-9, atol=1e-9)


def test_single_psp_matches_dense_threshold_search() -> None:
    layer = GoeltzLIFLayer(1, 1, weight_std=0.0, no_spike_time=10.0).double()
    with torch.no_grad():
        layer.weight.fill_(3.0)
    input_time = torch.tensor([[0.2]], dtype=torch.float64)
    result = layer(input_time)
    assert bool(result.spiked.item())

    grid = torch.linspace(0.2, 1.2, 200_001, dtype=torch.float64)
    age = grid - 0.2
    voltage = 3.0 * age * torch.exp(-age)
    dense_crossing = grid[(voltage >= 1.0).nonzero()[0, 0]]
    assert abs(float(result.times.item() - dense_crossing)) < 1e-5


def test_weight_gradient_matches_finite_difference() -> None:
    layer = GoeltzLIFLayer(2, 1, weight_std=0.0, no_spike_time=10.0).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.2, 1.4]], dtype=torch.float64))
    times = torch.tensor([[0.1, 0.4]], dtype=torch.float64)
    output = layer(times).times.sum()
    output.backward()
    analytical = layer.weight.grad.detach().clone()

    epsilon = 1e-6
    finite = torch.zeros_like(analytical)
    with torch.no_grad():
        for index in range(2):
            original = float(layer.weight[0, index])
            layer.weight[0, index] = original + epsilon
            plus = float(layer(times).times)
            layer.weight[0, index] = original - epsilon
            minus = float(layer(times).times)
            layer.weight[0, index] = original
            finite[0, index] = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(analytical, finite, rtol=2e-4, atol=2e-5)


def test_multiple_psps_match_dense_voltage_simulation() -> None:
    layer = GoeltzLIFLayer(3, 1, weight_std=0.0, no_spike_time=10.0).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.5, -0.5, 1.8]], dtype=torch.float64))
    input_times = torch.tensor([[0.1, 0.3, 0.6]], dtype=torch.float64)
    result = layer(input_times)
    assert bool(result.spiked.item())

    grid = torch.linspace(0.0, 4.0, 400_001, dtype=torch.float64)
    ages = grid[:, None] - input_times[0]
    kernels = torch.where(ages >= 0.0, ages * torch.exp(-ages), torch.zeros_like(ages))
    voltage = kernels @ layer.weight.detach()[0]
    dense_crossing = grid[(voltage >= 1.0).nonzero()[0, 0]]
    assert abs(float(result.times.item() - dense_crossing)) < 1e-5


def test_numpy_layer_matches_torch_layer() -> None:
    layer = GoeltzLIFLayer(5, 4, weight_std=0.0, no_spike_time=8.0).double()
    weights = torch.tensor(
        [[2.0, -0.4, 1.2, 0.7, -0.1],
         [0.1, 0.2, 0.3, 0.4, 0.5],
         [-1.0, -0.5, -0.2, -0.1, -0.3],
         [1.5, 1.0, -0.8, 0.6, 0.2]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(weights)
    times = torch.tensor(
        [[0.1, 0.4, 0.2, 0.8, 0.5], [0.7, 0.2, 0.5, 0.1, 0.9]],
        dtype=torch.float64,
    )
    active = torch.tensor(
        [[True, True, True, True, False], [True, False, True, True, True]]
    )
    torch_result = layer(times, active)
    numpy_result = lif_first_spike_layer(
        times.numpy(), weights.numpy(), active.numpy(), no_spike_time=8.0
    )
    np.testing.assert_array_equal(numpy_result.spiked, torch_result.spiked.numpy())
    np.testing.assert_allclose(
        numpy_result.times, torch_result.times.detach().numpy(), rtol=1e-9, atol=1e-9
    )


def test_numpy_network_tie_breaks_by_lowest_output_index() -> None:
    first = np.full((2, 3), 3.0, dtype=np.float64)
    # Both output neurons receive identical, sufficiently strong input and
    # therefore emit at the same finite spike time.
    second = np.full((2, 2), 2.0, dtype=np.float64)
    inputs = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float64)
    predictions = lif_ttfs_forward_batch([first, second], inputs)
    np.testing.assert_array_equal(predictions, np.array([0, 0]))


def test_numpy_network_all_silent_outputs_choose_lowest_index() -> None:
    first = np.full((2, 3), -1.0, dtype=np.float64)
    second = np.ones((2, 2), dtype=np.float64)
    inputs = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float64)
    predictions = lif_ttfs_forward_batch([first, second], inputs)
    np.testing.assert_array_equal(predictions, np.array([0, 0]))


def test_crossing_after_deadline_is_terminal_no_spike() -> None:
    layer = GoeltzLIFLayer(
        1, 1, weight_std=0.0, no_spike_time=0.3
    ).double()
    with torch.no_grad():
        layer.weight.fill_(3.0)
    input_time = torch.tensor([[0.2]], dtype=torch.float64)
    torch_result = layer(input_time)
    numpy_result = lif_first_spike_layer(
        input_time.numpy(),
        layer.weight.detach().numpy(),
        no_spike_time=0.3,
    )
    assert not bool(torch_result.spiked.item())
    assert float(torch_result.times.item()) == 0.3
    assert not bool(numpy_result.spiked.item())
    assert float(numpy_result.times.item()) == 0.3


def test_silent_neuron_uses_no_spike_sentinel() -> None:
    layer = GoeltzLIFLayer(2, 1, weight_std=0.0, no_spike_time=7.0).double()
    with torch.no_grad():
        layer.weight.fill_(-1.0)
    result = layer(torch.tensor([[0.1, 0.2]], dtype=torch.float64))
    assert not bool(result.spiked.item())
    assert float(result.times.item()) == 7.0
