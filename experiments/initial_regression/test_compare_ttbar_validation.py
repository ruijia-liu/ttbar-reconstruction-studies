"""Small numerical and input-integrity tests for the paired VAL comparison."""
from pathlib import Path
import sys
import tempfile
import unittest

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "tmp/ttbar4_packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from compare_ttbar_validation import (
    LABELS, TARGETS, compare_one, endpoint_summary, load_predictions,
    require_aligned_truth,
)


class PairedComparisonTests(unittest.TestCase):
    def test_identical_predictions_have_exact_zero_paired_intervals(self):
        truth = np.array([-5., -2., 0., 1., 7.])
        pred = np.array([-3., -1., .3, .9, 4.])
        result = compare_one(truth, pred, pred.copy(), bootstrap=12, seed=71)
        for metric in ("mae", "rmse", "wasserstein1"):
            self.assertEqual(result["difference_emd_minus_mse"][metric], 0.)
            self.assertEqual(result["difference_bootstrap_se"][metric], 0.)
            self.assertEqual(result["difference_bootstrap_ci95"][metric], [0., 0.])
        self.assertGreater(result["models"][LABELS[0]]["mae"], 0.)

    def test_bootstrap_differences_use_the_same_event_indices_for_both_models(self):
        truth = np.array([-4., -1., 0., 2., 9.])
        baseline = np.array([-3., -.5, -2., 3., 5.])
        variant = np.array([-3.8, 1., -.3, 2.2, 8.])
        count, seed = 17, 127
        result = compare_one(truth, baseline, variant, bootstrap=count, seed=seed)

        def independent_scores(t, p):
            return np.array([
                sum(abs(float(a) - float(b)) for a, b in zip(t, p)) / len(t),
                np.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(t, p)) / len(t)),
                sum(abs(a - b) for a, b in zip(sorted(t), sorted(p))) / len(t),
            ])

        rng = np.random.default_rng(seed)
        expected = []
        for _ in range(count):
            indices = rng.integers(0, len(truth), len(truth))
            expected.append(independent_scores(truth[indices], variant[indices])
                            - independent_scores(truth[indices], baseline[indices]))
        expected = np.asarray(expected)
        for index, metric in enumerate(("mae", "rmse", "wasserstein1")):
            self.assertAlmostEqual(result["difference_bootstrap_se"][metric],
                                   expected[:, index].std(ddof=1))
            np.testing.assert_allclose(result["difference_bootstrap_ci95"][metric],
                                       np.quantile(expected[:, index], [.025, .975]), atol=1e-14)
        self.assertEqual(result, compare_one(truth, baseline, variant, bootstrap=count, seed=seed))

    def test_global_w1_does_not_certify_eventwise_reconstruction(self):
        truth = np.array([-3., -1., 2., 6.])
        result = compare_one(truth, truth.copy(), truth[::-1], bootstrap=0)
        base, variant = (result["models"][label] for label in LABELS)
        self.assertEqual(base["wasserstein1"], 0.)
        self.assertEqual(variant["wasserstein1"], 0.)
        self.assertEqual(base["mae"], 0.)
        self.assertEqual(variant["mae"], 6.)
        self.assertEqual(result["difference_emd_minus_mse"]["mae"], 6.)
        self.assertIsNone(result["percentage_change_from_mse"]["mae"])
        self.assertEqual(result["difference_bootstrap_ci95"], {})

    def test_reordered_or_changed_truth_refuses_a_paired_comparison(self):
        truth = {name: np.array([index + .1, index + .2, index + .3])
                 for index, name in enumerate(TARGETS)}
        require_aligned_truth(truth, {name: value.copy() for name, value in truth.items()})
        reordered = {name: value[::-1] for name, value in truth.items()}
        with self.assertRaisesRegex(ValueError, "value or order"):
            require_aligned_truth(truth, reordered)
        changed = {name: value.copy() for name, value in truth.items()}
        changed[TARGETS[-1]][1] += .01
        with self.assertRaisesRegex(ValueError, TARGETS[-1]):
            require_aligned_truth(truth, changed)

    def test_invalid_pairs_and_bootstrap_counts_are_rejected(self):
        for truth, baseline, variant in [([], [], []), ([1], [1, 2], [1]),
                                        ([1, np.nan], [1, 2], [1, 2]),
                                        ([1, 2], [1, 2], [1, np.inf]),
                                        ([[1, 2]], [[1, 2]], [[1, 2]])]:
            with self.subTest(truth=truth, baseline=baseline, variant=variant):
                with self.assertRaises(ValueError):
                    compare_one(truth, baseline, variant, bootstrap=0)
        for count in (-1, 1, True, 2.5):
            with self.subTest(bootstrap=count):
                with self.assertRaises(ValueError):
                    compare_one([0, 1], [0, 1], [0, 1], bootstrap=count)


class EndpointTests(unittest.TestCase):
    def test_physical_boundaries_are_inclusive_and_endpoint_bins_explicit(self):
        truth = np.array([-1., -.95, -.9, 0., .9, .95, 1.])
        pred = np.array([-1.01, -1., -.9, 0., .9, 1., 1.01])
        result = endpoint_summary(truth, pred)
        self.assertEqual(result["predictions_outside_physical_range"], 2)
        neg, pos = result["negative_endpoint"], result["positive_endpoint"]
        self.assertEqual((neg["truth_count"], neg["prediction_count"]), (2, 1))
        self.assertEqual(neg["predicted_to_truth_count_ratio"], .5)
        self.assertFalse(neg["right_edge_inclusive"])
        self.assertEqual((pos["truth_count"], pos["prediction_count"]), (3, 2))
        self.assertAlmostEqual(pos["predicted_to_truth_count_ratio"], 2 / 3)
        self.assertTrue(pos["right_edge_inclusive"])
        empty = endpoint_summary(np.array([0.]), np.array([1.]))
        self.assertIsNone(empty["positive_endpoint"]["predicted_to_truth_count_ratio"])


class PredictionLoadingTests(unittest.TestCase):
    def setUp(self):
        scratch = WORKSPACE / "tmp/ttbar4_test_runs"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="compare_validation_", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "validation.parquet"

    def write_predictions(self, *, rows=3, missing_target=None, missing_prediction=None,
                          nonfinite_prediction=None, parent_null=False):
        targets, predictions = [], []
        for index, name in enumerate(TARGETS):
            t = [float(index + i) for i in range(rows)]
            p = [float(index + i) + .1 for i in range(rows)]
            if name == missing_target:
                t[1] = None
            if name == missing_prediction:
                p[1] = None
            if name == nonfinite_prediction:
                p[1] = float("inf")
            targets.append(pa.array(t, type=pa.float64()))
            predictions.append(pa.array(p, type=pa.float64()))
        mask = pa.array([False, True] + [False] * (rows - 2)) if parent_null else None
        target_struct = pa.StructArray.from_arrays(targets, names=TARGETS, mask=mask)
        pred_struct = pa.StructArray.from_arrays(predictions, names=TARGETS)
        pq.write_table(pa.table({"targets": target_struct, "predictions": pred_struct}), self.path)

    def test_expected_rows_and_values_are_loaded_in_order(self):
        self.write_predictions()
        truth, pred = load_predictions(self.path, expected_rows=3)
        self.assertEqual(list(truth), TARGETS)
        for index, name in enumerate(TARGETS):
            np.testing.assert_array_equal(truth[name], np.arange(3) + index)
            np.testing.assert_allclose(pred[name], np.arange(3) + index + .1)
            self.assertEqual(truth[name].dtype, np.float64)

    def test_incorrect_count_missing_values_and_nonfinite_outputs_are_rejected(self):
        self.write_predictions(rows=2)
        with self.assertRaisesRegex(ValueError, "Expected 3 validation events"):
            load_predictions(self.path, expected_rows=3)
        for option in ({"missing_target": TARGETS[0]}, {"missing_prediction": TARGETS[1]},
                       {"nonfinite_prediction": TARGETS[2]}, {"parent_null": True}):
            with self.subTest(option=option):
                self.write_predictions(**option)
                with self.assertRaises(ValueError):
                    load_predictions(self.path, expected_rows=3)


if __name__ == "__main__":
    unittest.main()
