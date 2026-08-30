"""Soundness regressions for continuous-time LIF BC-IBP.

Run directly with ``python test_lif_bnb_ibp.py``; no pytest dependency is
required.  Tiny networks are exhaustively enumerated over the complete finite
input-shift domain.
"""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from lif_bnb_ibp import (
    alpha_psp_interval,
    input_voltage_bounds_mckp,
    lif_bcibp_prove_robust,
)
from utils.lif_ttfs_net import lif_first_spike_layer, lif_ttfs_forward


def enumerate_shifts(n_inputs: int, budget: int):
    for shifts in itertools.product(range(-budget, budget + 1), repeat=n_inputs):
        if sum(abs(value) for value in shifts) <= budget:
            yield np.asarray(shifts, dtype=np.int32)


class LIFBCIBPTests(unittest.TestCase):
    def test_all_silent_terminal_tie_certifies_only_class_zero(self) -> None:
        input_times = np.asarray([0.2, 0.7], dtype=np.float64)
        weights = [
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((3, 2), dtype=np.float64),
        ]
        common = dict(
            budget=1,
            shift_step=0.2,
            input_min=0.0,
            input_max=1.0,
            no_spike_time=2.0,
            time_step=0.05,
        )

        class_zero = lif_bcibp_prove_robust(
            input_times, weights, prediction=0, **common
        )
        class_one = lif_bcibp_prove_robust(
            input_times, weights, prediction=1, **common
        )

        self.assertTrue(class_zero.robust)
        self.assertFalse(class_one.robust)
        self.assertTrue(np.all(np.isinf(class_zero.output.earliest)))

    def test_alpha_interval_contains_dense_curve(self) -> None:
        cases = [(-2.0, -0.1), (-0.2, 0.4), (0.2, 0.8), (0.3, 1.4), (1.3, 3.0)]
        for lo, hi in cases:
            lower, upper = alpha_psp_interval(np.asarray(lo), np.asarray(hi))
            ages = np.linspace(lo, hi, 100_001)
            values = np.where(ages >= 0.0, ages * np.exp(-ages), 0.0)
            self.assertLessEqual(float(lower), float(values.min()))
            self.assertGreaterEqual(float(upper), float(values.max()))

    def test_mckp_point_bounds_match_exhaustive_options(self) -> None:
        times = np.asarray([1.0, 2.0, 0.0])
        weights = np.asarray([[2.0, 1.5, 1.0], [-0.7, 0.8, 1.2]])
        budget = 2
        lower, upper = input_voltage_bounds_mckp(
            weights, times, cell_start=3.0, cell_end=3.0,
            budget=budget, shift_step=1.0, input_min=0.0, input_max=3.0,
        )
        exact = []
        for shifts in enumerate_shifts(3, budget):
            shifted = np.clip(times + shifts, 0.0, 3.0)
            age = 3.0 - shifted
            kernel = np.where(age >= 0.0, age * np.exp(-age), 0.0)
            exact.append(weights @ kernel)
        exact_arr = np.asarray(exact)
        np.testing.assert_allclose(lower, exact_arr.min(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(upper, exact_arr.max(axis=0), rtol=1e-12, atol=1e-12)

    def test_layer_and_network_bounds_contain_exhaustive_forward(self) -> None:
        input_times = np.asarray([0.2, 0.7, 1.1], dtype=np.float64)
        w_hidden = np.asarray(
            [[3.0, 2.0, 1.5], [2.5, -0.4, 2.2]], dtype=np.float64
        )
        w_output = np.asarray(
            [[3.0, 1.5], [1.2, 3.0]], dtype=np.float64
        )
        weights = [w_hidden, w_output]
        budget = 1
        step = 0.4
        baseline = lif_ttfs_forward(weights, input_times)
        bound = lif_bcibp_prove_robust(
            input_times, weights, budget=budget, shift_step=step,
            input_min=0.0, input_max=1.5, time_step=0.025,
            prediction=baseline,
        )

        predictions = []
        for shifts in enumerate_shifts(input_times.size, budget):
            perturbed = np.clip(input_times + step * shifts, 0.0, 1.5)
            hidden = lif_first_spike_layer(perturbed, w_hidden)
            output = lif_first_spike_layer(hidden.times, w_output, hidden.spiked)
            for index, result in enumerate((hidden, output)):
                interval = bound.hidden if index == 0 else bound.output
                for neuron, spiked in enumerate(result.spiked):
                    if spiked:
                        self.assertGreaterEqual(
                            result.times[neuron] + 1e-10, interval.earliest[neuron]
                        )
                        self.assertLessEqual(
                            result.times[neuron], interval.possible_latest[neuron] + 1e-10
                        )
                        if np.isfinite(interval.latest[neuron]):
                            self.assertLessEqual(
                                result.times[neuron], interval.latest[neuron] + 1e-10
                            )
                    else:
                        self.assertFalse(np.isfinite(interval.latest[neuron]))
            predictions.append(lif_ttfs_forward(weights, perturbed))

        if bound.robust:
            self.assertEqual(set(predictions), {baseline})

    def test_psp_peak_is_not_missed_between_grid_points(self) -> None:
        # The peak at age=1 lies strictly inside the cell. Endpoint-only
        # sampling would miss it, while the interval upper must include 1/e.
        lower, upper = alpha_psp_interval(np.asarray(0.6), np.asarray(1.4))
        self.assertLess(float(lower), 1.0 / math.e)
        self.assertGreaterEqual(float(upper), 1.0 / math.e)


if __name__ == "__main__":
    unittest.main(verbosity=2)
