"""Soundness and completeness tests for the full continuous-time LIF BnB."""
from __future__ import annotations

import itertools
import unittest

import numpy as np

from lif_bnb import (
    LIFModelSpec,
    LIFThreatSpec,
    LIFVerifyConfig,
    METHODS,
    verify_lif_ttfs,
)
from lif_bnb_ibp import (
    canonical_shift_options,
    input_voltage_bounds_cells,
    lif_bcibp_bound_node,
)
from utils.lif_ttfs_net import lif_ttfs_forward_batch


def enumerate_shifts(n_inputs: int, budget: int):
    for values in itertools.product(range(-budget, budget + 1), repeat=n_inputs):
        if sum(abs(value) for value in values) <= budget:
            yield np.asarray(values, dtype=np.int32)


class FullLIFBnBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.asarray([0.2, 0.7, 1.1], dtype=np.float64)
        self.weights = (
            np.asarray([[3.0, 2.0, 1.5], [2.5, -0.4, 2.2]], dtype=np.float64),
            np.asarray([[3.0, 1.5], [1.2, 3.0]], dtype=np.float64),
        )
        self.model = LIFModelSpec(
            self.weights, input_min=0.0, input_max=1.5
        )

    def test_canonical_options_keep_minimum_clipped_cost(self) -> None:
        options = canonical_shift_options(0.0, 3, 0.4, 0.0, 1.0)
        by_time = {value[2]: value for value in options}
        self.assertEqual(by_time[0.0], (0, 0, 0.0))
        self.assertEqual(by_time[0.4], (1, 1, 0.4))
        self.assertEqual(by_time[0.8], (2, 2, 0.8))
        self.assertEqual(by_time[1.0], (3, 3, 1.0))

    def test_batched_cell_mckp_contains_complete_enumeration(self) -> None:
        starts = np.asarray([0.0, 0.5, 1.0, 1.5])
        ends = starts + 0.35
        lower, upper = input_voltage_bounds_cells(
            self.weights[0],
            self.times,
            cell_starts=starts,
            cell_ends=ends,
            budget=2,
            shift_step=0.4,
            input_min=0.0,
            input_max=1.5,
        )
        for shift in enumerate_shifts(3, 2):
            perturbed = np.clip(self.times + 0.4 * shift, 0.0, 1.5)
            for cell, (left, right) in enumerate(zip(starts, ends)):
                dense = np.linspace(left, right, 1001)
                age = dense[:, None] - perturbed[None, :]
                kernel = np.where(age >= 0.0, age * np.exp(-age), 0.0)
                voltage = kernel @ self.weights[0].T
                self.assertTrue(np.all(voltage >= lower[cell] - 1e-12))
                self.assertTrue(np.all(voltage <= upper[cell] + 1e-12))

    def test_multilayer_bounds_contain_all_exact_spikes(self) -> None:
        extra = np.asarray([[2.4, 1.3], [1.1, 2.6]], dtype=np.float64)
        weights = (*self.weights, extra)
        prediction = int(lif_ttfs_forward_batch(weights, self.times))
        certificate = lif_bcibp_bound_node(
            self.times,
            weights,
            budget=1,
            shift_step=0.4,
            input_min=0.0,
            input_max=1.5,
            prediction=prediction,
            time_step=0.05,
            min_time_step=0.025,
        )
        for shift in enumerate_shifts(3, 1):
            perturbed = np.clip(self.times + 0.4 * shift, 0.0, 1.5)
            from utils.lif_ttfs_net import LIFLayerForwardResult
            layers: list[LIFLayerForwardResult] = []
            lif_ttfs_forward_batch(weights, perturbed, layer_results_return=layers)
            for exact, bound in zip(layers, certificate.layers):
                for index, spiked in enumerate(exact.spiked[0]):
                    if spiked:
                        self.assertGreaterEqual(exact.times[0, index] + 1e-10, bound.earliest[index])
                        self.assertLessEqual(exact.times[0, index], bound.possible_latest[index] + 1e-10)
                        if np.isfinite(bound.latest[index]):
                            self.assertLessEqual(exact.times[0, index], bound.latest[index] + 1e-10)
                    else:
                        self.assertFalse(np.isfinite(bound.latest[index]))

    def test_all_bnb_variants_equal_exhaustive(self) -> None:
        threat = LIFThreatSpec(2, 0.4, 0.0, 1.5)
        oracle = verify_lif_ttfs(
            self.model,
            self.times,
            threat,
            LIFVerifyConfig(method="exhaustive", timeout_s=30.0),
        )
        self.assertIn(oracle.verdict, {"robust", "not_robust"})
        for method in sorted(METHODS - {"exhaustive"}):
            with self.subTest(method=method):
                result = verify_lif_ttfs(
                    self.model,
                    self.times,
                    threat,
                    LIFVerifyConfig(
                        method=method,
                        bound_every=1,
                        timeout_s=30.0,
                        time_step=0.05,
                        min_time_step=0.05,
                    ),
                )
                self.assertEqual(result.verdict, oracle.verdict)
                if result.verdict == "not_robust":
                    self.assertLessEqual(result.witness["l1_cost"], threat.budget)
                    replay = np.asarray(result.witness["perturbed_times"])
                    replay_prediction = int(lif_ttfs_forward_batch(self.weights, replay))
                    self.assertEqual(replay_prediction, result.witness["prediction"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

