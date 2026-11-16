"""Regression demonstration for nullable Parquet targets and float32 training.

Uses the real converter's synthetic physics fixture. No Torch/GPU is required:
the failure mechanism is a MaskedArray promotion before torch.from_numpy.
"""
from pathlib import Path
import sys
import tempfile
import unittest

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "tmp/ttbar4_packages"))

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_ttbar_four import TARGETS, convert_root_chunk
from test_prepare_ttbar_four import example_records


def old_target_matrix(targets):
    """Current upstream load_arrays expression, preserved for reproduction."""
    return np.stack(
        [ak.to_numpy(targets[name]).astype(np.float32) for name in TARGETS],
        axis=1,
    )


def checked_float32_matrix(targets):
    """Proposed loader boundary: reject nulls, then use plain float32 arrays."""
    return np.stack(
        [np.asarray(ak.to_numpy(targets[name], allow_missing=False), dtype=np.float32)
         for name in TARGETS],
        axis=1,
    )


def normalize(values, mean, std):
    return (values - mean) / (std + 1e-8)


class TargetDtypeTests(unittest.TestCase):
    def setUp(self):
        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
        reco, truth = example_records()
        self.events = convert_root_chunk(ak.Array(reco), ak.Array(truth), "truth_parent_pdgId")

    def test_nullable_parquet_reproduces_old_float64_promotion(self):
        temp_root = WORKSPACE / "tmp/ttbar4_test_runs"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            path = Path(folder) / "nullable_targets.parquet"
            # Match the converter's plain Arrow writer, including nullability.
            pq.write_table(ak.to_arrow_table(self.events, extensionarray=False), path)
            targets = ak.from_parquet(path).targets

        self.assertFalse(bool(ak.any(ak.is_none(targets.ttbar_mass))))
        matrix = old_target_matrix(targets)
        self.assertIsInstance(matrix, np.ma.MaskedArray)
        self.assertEqual(matrix.dtype, np.dtype("float32"))
        mean, std = matrix.mean(0), matrix.std(0)
        self.assertEqual(mean.dtype, np.dtype("float32"))
        self.assertEqual(std.dtype, np.dtype("float32"))
        # This demonstrates the observed NumPy masked-array scalar promotion:
        # casting only before np.stack is insufficient.
        self.assertEqual((std + 1e-8).dtype, np.dtype("float64"))
        self.assertEqual(normalize(matrix, mean, std).dtype, np.dtype("float64"))

        fixed = checked_float32_matrix(targets)
        self.assertIs(type(fixed), np.ndarray)
        self.assertEqual(fixed.dtype, np.dtype("float32"))
        self.assertEqual(normalize(fixed, fixed.mean(0), fixed.std(0)).dtype,
                         np.dtype("float32"))
        # Plain-array conversion does not alter any of the already-float32 data.
        np.testing.assert_array_equal(fixed, matrix.data)

    def test_train_stats_and_validation_targets_remain_float32(self):
        train = checked_float32_matrix(self.events.targets[:5])
        validation = checked_float32_matrix(self.events.targets[5:])
        mean, std = train.mean(0), train.std(0)
        original_mean, original_std = mean.copy(), std.copy()
        train_scaled = normalize(train, mean, std)
        validation_scaled = normalize(validation, mean, std)
        for value in (train, validation, mean, std, train_scaled, validation_scaled):
            self.assertIs(type(value), np.ndarray)
            self.assertEqual(value.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(value).all())
        self.assertEqual(train_scaled.shape, (5, 4))
        self.assertEqual(validation_scaled.shape, (2, 4))
        reference = ((validation.astype(np.float64) - mean.astype(np.float64)) /
                     (std.astype(np.float64) + 1e-8))
        np.testing.assert_allclose(validation_scaled, reference, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(mean, original_mean)
        np.testing.assert_array_equal(std, original_std)

    def test_genuinely_missing_truth_is_rejected_instead_of_unmasked(self):
        with_missing = ak.with_field(
            self.events.targets,
            ak.mask(self.events.targets.cos_phi, np.array([True] * 6 + [False])),
            "cos_phi",
        )
        with self.assertRaises(ValueError):
            checked_float32_matrix(with_missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
