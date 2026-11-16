"""Exercise the actual Parquet writer with noncontiguous NumPy inputs.

Run beside evaluate_density.py: ``python test_prediction_export.py -v``.
These tests require NumPy, Awkward, PyArrow and SciPy, but no Torch or GPU.
The original writer reproduced ``ValueError: ndarray is not contiguous`` with
Awkward 2.14.0 / PyArrow 26.0.1 on the C-order column views used below.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import awkward as ak
import numpy as np


HERE = Path(__file__).resolve().parent


class PredictionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = importlib.util.spec_from_file_location(
            '_prediction_export_evaluation', HERE / 'evaluate_density.py'
        )
        if specification is None or specification.loader is None:
            raise RuntimeError('Cannot import the retry evaluation module')
        cls.evaluation = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.evaluation)

    def setUp(self):
        self.truth = np.array([
            [350.0 + i * 31.25, -1.5 + i * .125, -.8 + i * .09, .7 - i * .075]
            for i in range(17)
        ], dtype=np.float64)
        self.prediction = (self.truth + np.array([2.5, .015, -.012, .008])).astype(np.float32)
        self.row_index = np.arange(100, 117, dtype=np.int64)

    def assert_round_trip(self, destination, prediction, truth, row_index):
        # Copies are the oracle: the writer must preserve dtype, values and row order.
        expected_prediction = np.array(prediction, copy=True)
        expected_truth = np.array(truth, copy=True)
        expected_rows = np.array(row_index, copy=True)
        self.evaluation.write_predictions(destination, prediction, truth, row_index)
        actual = ak.from_parquet(destination)
        self.assertEqual(len(actual), len(expected_rows))
        self.assertEqual(set(actual.fields), {'row_index', 'targets', 'predictions'})
        actual_rows = ak.to_numpy(actual['row_index'])
        self.assertEqual(actual_rows.dtype, expected_rows.dtype)
        np.testing.assert_array_equal(actual_rows, expected_rows)
        for record, expected in [('targets', expected_truth), ('predictions', expected_prediction)]:
            self.assertEqual(actual[record].fields, self.evaluation.KEYS)
            for column, key in enumerate(self.evaluation.KEYS):
                values = ak.to_numpy(actual[record][key])
                self.assertEqual(values.dtype, expected[:, column].dtype, f'{record}.{key}')
                np.testing.assert_array_equal(values, expected[:, column])
        # Export must not modify caller-owned arrays or row indices either.
        np.testing.assert_array_equal(prediction, expected_prediction)
        np.testing.assert_array_equal(truth, expected_truth)
        np.testing.assert_array_equal(row_index, expected_rows)

    def test_c_order_strided_columns_round_trip(self):
        self.assertTrue(self.prediction.flags.c_contiguous)
        self.assertFalse(self.prediction[:, 0].flags.c_contiguous)
        self.assertFalse(self.truth[:, 0].flags.c_contiguous)
        with tempfile.TemporaryDirectory(prefix='_export_c_', dir=HERE) as directory:
            self.assert_round_trip(Path(directory) / 'predictions.parquet',
                                   self.prediction, self.truth, self.row_index)

    def test_fortran_order_arrays_round_trip(self):
        prediction = np.asfortranarray(self.prediction)
        truth = np.asfortranarray(self.truth)
        self.assertTrue(prediction.flags.f_contiguous and truth.flags.f_contiguous)
        self.assertTrue(prediction[:, 0].flags.c_contiguous)
        with tempfile.TemporaryDirectory(prefix='_export_f_', dir=HERE) as directory:
            self.assert_round_trip(Path(directory) / 'predictions.parquet',
                                   prediction, truth, self.row_index)

    def test_readonly_memmap_arrays_round_trip(self):
        with tempfile.TemporaryDirectory(prefix='_export_mmap_', dir=HERE) as directory:
            directory = Path(directory)
            np.save(directory / 'prediction.npy', self.prediction)
            np.save(directory / 'truth.npy', self.truth)
            np.save(directory / 'rows.npy', self.row_index)
            prediction = np.load(directory / 'prediction.npy', mmap_mode='r')
            truth = np.load(directory / 'truth.npy', mmap_mode='r')
            rows = np.load(directory / 'rows.npy', mmap_mode='r')
            try:
                self.assertFalse(prediction.flags.writeable or truth.flags.writeable or rows.flags.writeable)
                self.assert_round_trip(directory / 'predictions.parquet', prediction, truth, rows)
            finally:
                # Release Windows mapping handles before TemporaryDirectory cleanup.
                del prediction, truth, rows

    def test_noncontiguous_row_index_preserves_identifiers(self):
        rows = np.arange(1000, 1034, dtype=np.int64)[::2]
        self.assertFalse(rows.flags.c_contiguous)
        with tempfile.TemporaryDirectory(prefix='_export_rows_', dir=HERE) as directory:
            self.assert_round_trip(Path(directory) / 'predictions.parquet',
                                   self.prediction, self.truth, rows)

    def test_rows_and_columns_both_strided_round_trip(self):
        prediction_storage = np.zeros((34, 8), dtype=np.float32)
        truth_storage = np.zeros((34, 8), dtype=np.float64)
        prediction_storage[::2, ::2] = self.prediction
        truth_storage[::2, ::2] = self.truth
        prediction, truth = prediction_storage[::2, ::2], truth_storage[::2, ::2]
        self.assertFalse(prediction.flags.c_contiguous or prediction.flags.f_contiguous)
        self.assertFalse(truth.flags.c_contiguous or truth.flags.f_contiguous)
        with tempfile.TemporaryDirectory(prefix='_export_strided_', dir=HERE) as directory:
            self.assert_round_trip(Path(directory) / 'predictions.parquet',
                                   prediction, truth, self.row_index)

    def test_negative_strides_preserve_reverse_order(self):
        with tempfile.TemporaryDirectory(prefix='_export_reverse_', dir=HERE) as directory:
            self.assert_round_trip(Path(directory) / 'predictions.parquet',
                                   self.prediction[::-1], self.truth[::-1], self.row_index[::-1])

    def test_single_mean_and_median_outputs_preserve_their_dtypes(self):
        generator = np.random.default_rng(4204)
        draws = (self.prediction[:, None, :] + generator.normal(0, .01, (17, 32, 4))).astype(np.float32)
        modes = {
            'single_draw': draws[:, 0, :],
            'mean32': draws.mean(axis=1, dtype=np.float64),
            'median32': np.median(draws.astype(np.float64), axis=1),
        }
        self.assertEqual(modes['single_draw'].dtype, np.dtype('float32'))
        self.assertEqual(modes['mean32'].dtype, np.dtype('float64'))
        self.assertEqual(modes['median32'].dtype, np.dtype('float64'))
        with tempfile.TemporaryDirectory(prefix='_export_modes_', dir=HERE) as directory:
            for mode, prediction in modes.items():
                with self.subTest(mode=mode):
                    self.assert_round_trip(Path(directory) / f'predictions_{mode}.parquet',
                                           prediction, self.truth, self.row_index)


if __name__ == '__main__':
    unittest.main()
