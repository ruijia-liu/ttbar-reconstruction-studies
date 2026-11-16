"""Numerical tests for paired regression metrics and leakage-aware responses."""
from pathlib import Path
import sys
import unittest

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "tmp/ttbar4_packages"))

import numpy as np

from ttbar_metrics import paired_histogram, regression_metrics, response_matrix


class RegressionMetricTests(unittest.TestCase):
    def test_known_residuals_and_full_quantile_width(self):
        # Residuals -2,-1,0,1,2 have MAE 1.2, RMSE sqrt(2), and zero bias.
        result = regression_metrics([-2, -1, 0, 1, 2], [0] * 5, bootstrap=0)
        self.assertEqual(result["n_events"], 5)
        self.assertAlmostEqual(result["mae"], 1.2)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2))
        self.assertEqual(result["bias"], 0)
        self.assertAlmostEqual(result["central68_width"], 2.7304)
        self.assertEqual(result["bootstrap_se"], {})
        shifted = regression_metrics([3, 4], [1, 2], bootstrap=0)
        self.assertEqual(shifted["bias"], 2)
        self.assertEqual(shifted["central68_width"], 0)

    def test_bootstrap_is_paired_and_reproducible(self):
        truth = np.arange(31, dtype=float)
        pred = truth - 2
        first = regression_metrics(truth, pred, bootstrap=20, seed=42)
        self.assertEqual(first, regression_metrics(truth, pred, bootstrap=20, seed=42))
        # Independent truth/pred resampling would destroy the constant residual.
        self.assertTrue(all(value == 0 for value in first["bootstrap_se"].values()))
        self.assertEqual(first["bootstrap_ci95"]["bias"], [2, 2])

    def test_bootstrap_standard_error_matches_manual_paired_indices(self):
        residual = np.array([-3., 1., 2., 7.])
        result = regression_metrics(residual, np.zeros(4), bootstrap=15, seed=7)
        rng = np.random.default_rng(7)
        means = [residual[rng.integers(0, 4, size=4)].mean() for _ in range(15)]
        self.assertAlmostEqual(result["bootstrap_se"]["bias"], np.std(means, ddof=1))
        np.testing.assert_allclose(result["bootstrap_ci95"]["bias"], np.quantile(means, [0.025, 0.975]))


class BinnedMetricTests(unittest.TestCase):
    def test_response_orientation_boundaries_and_prediction_leakage(self):
        truth = [0, .5, 1, 2, 3, -1, 4]
        pred = [-.1, .5, 3, 4, 2, 1, 1]
        result = response_matrix(truth, pred, [0, 1, 2, 3])
        # Columns are prediction underflow, [0,1), [1,2), [2,3], overflow.
        expected = np.array([[1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(result["full_counts"], expected)
        np.testing.assert_array_equal(result["row_counts"], [2, 1, 2])
        np.testing.assert_allclose(result["full_probs"].sum(axis=1), 1)
        np.testing.assert_allclose(result["central_probs"].sum(axis=1), [.5, 1, .5])
        self.assertAlmostEqual(result["diagonal_mean"], 1 / 3)
        self.assertEqual(result["n_nonempty_rows"], 3)
        self.assertEqual(result["range_counts"]["truth_out_of_range"], 2)
        self.assertEqual(result["range_counts"]["pred_out_of_range"], 2)
        self.assertAlmostEqual(result["range_fractions"]["truth_out_of_range"], 2 / 7)

    def test_empty_response_rows_and_no_in_range_truth(self):
        result = response_matrix([.1, .2], [.1, -.1], [0, 1, 2])
        self.assertTrue(np.isnan(result["full_probs"][1]).all())
        self.assertEqual(result["n_nonempty_rows"], 1)
        self.assertEqual(result["diagonal_mean"], .5)
        empty = response_matrix([-1, 3], [.5, 1.5], [0, 1, 2])
        self.assertEqual(empty["n_nonempty_rows"], 0)
        self.assertTrue(np.isnan(empty["diagonal_mean"]))
        self.assertTrue(np.isnan(empty["full_probs"]).all())

    def test_histogram_agrees_with_numpy_and_keeps_common_normalization(self):
        edges = np.array([0, 1, 3])
        truth = np.array([0, 1, 3, -1])
        pred = np.array([.2, .5, 4, -.2])
        result = paired_histogram(truth, pred, edges)
        np.testing.assert_array_equal(result["truth_counts"], np.histogram(truth, edges)[0])
        np.testing.assert_array_equal(result["pred_counts"], np.histogram(pred, edges)[0])
        self.assertAlmostEqual(np.sum(result["truth_density"] * np.diff(edges)), .75)
        self.assertAlmostEqual(np.sum(result["pred_density"] * np.diff(edges)), .5)
        np.testing.assert_allclose(result["truth_density_se"], np.sqrt([1, 2]) / (4 * np.diff(edges)))
        np.testing.assert_array_equal(result["ratio"], [2, 0])

    def test_identical_paired_data_have_zero_ratio_error(self):
        values = [0, .2, 1, 1.5, 1.8, -1, 4]
        result = paired_histogram(values, values, [0, 1, 2, 2.5])
        np.testing.assert_array_equal(result["ratio"][:2], [1, 1])
        np.testing.assert_array_equal(result["ratio_se"][:2], [0, 0])
        self.assertTrue(np.isnan(result["ratio"][2]))
        self.assertTrue(np.isnan(result["ratio_se"][2]))

    def test_ratio_error_accounts_for_pair_covariance(self):
        # First bin has T=2, P=3, C=1; variance = 3/4 + 9/8 - 6/8.
        result = paired_histogram([.2, .3, 1.2, 1.3], [.2, 1.2, .3, .4], [0, 1, 2])
        np.testing.assert_array_equal(result["same_bin_counts"], [1, 0])
        self.assertAlmostEqual(result["ratio_se"][0] ** 2, 1.125)
        self.assertAlmostEqual(result["ratio_se"][1] ** 2, .375)


class ValidationTests(unittest.TestCase):
    def test_bad_inputs_rejected_by_all_functions(self):
        calls = [
            lambda t, p: regression_metrics(t, p, bootstrap=0),
            lambda t, p: response_matrix(t, p, [0, 1]),
            lambda t, p: paired_histogram(t, p, [0, 1]),
        ]
        cases = [
            ([], []), ([1], [1, 2]), ([[1]], [[1]]),
            ([np.nan], [1]), ([1], [np.inf]), ([None], [1]),
            (np.ma.array([1], mask=[True]), [1]), ([1 + 2j], [1]),
        ]
        for call in calls:
            for truth, pred in cases:
                with self.subTest(call=call, truth=truth, pred=pred):
                    with self.assertRaises(ValueError):
                        call(truth, pred)

    def test_invalid_edges_and_bootstrap_counts(self):
        for edges in ([0], [0, 0], [1, 0], [0, np.inf], [[0, 1]]):
            for function in (response_matrix, paired_histogram):
                with self.assertRaises(ValueError):
                    function([.5], [.5], edges)
        for count in (-1, 1, 1.5, True):
            with self.assertRaises(ValueError):
                regression_metrics([1], [1], bootstrap=count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
