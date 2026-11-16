"""CPU callback-logic tests, not a substitute for actual Lightning GPU smoke.

Only when pytorch_lightning itself is absent, provide its empty Callback base
class so the local CPU environment can exercise our lifecycle/selection logic.
The Slurm smoke uses the actual installed Lightning package and full datasets.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

try:
    import pytorch_lightning  # noqa: F401
except ModuleNotFoundError as exc:
    # Do not mask a broken installed Lightning dependency.
    if exc.name != "pytorch_lightning":
        raise
    _lightning_stub = types.ModuleType("pytorch_lightning")
    _lightning_stub.Callback = type("Callback", (), {})
    sys.modules["pytorch_lightning"] = _lightning_stub
    try:
        from training_control import BudgetSelection
    finally:
        del sys.modules["pytorch_lightning"]
else:
    from training_control import BudgetSelection

from evaluation import compute_metrics


class FakeTrainer:
    """Exercise checkpoint requests and atomic file replacement only."""

    def __init__(self):
        self.current_epoch = 0
        self.global_step = 2
        self.num_training_batches = 2
        self.sanity_checking = False
        self.should_stop = False
        self.saved = []

    def save_checkpoint(self, path):
        path = Path(path)
        self.saved.append((self.current_epoch, path.name))
        path.write_text(json.dumps({"epoch": self.current_epoch}), encoding="utf-8")


class TrainingControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vyper-callback-test-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        # Sixty events cover every production truth bin and both mass tails.
        # Pattern differences avoid identical coordinates across observables.
        self.truth = np.column_stack((
            np.tile([340, 410, 470, 625, 1200], 12),
            np.tile([-1.7, -.75, -.25, .25, .75, 1.7], 10),
            np.tile([-.75, -.25, .25, .75], 15),
            np.tile([.75, .25, -.25, -.75], 15),
        )).astype(np.float32)
        sign = np.where(np.arange(60) % 2, 1., -1.).astype(np.float32)
        self.first_prediction = self.truth + np.array([50., .1, .06, .06], dtype=np.float32)
        self.first_prediction += sign[:, None] * np.array([1., .01, .01, .01], dtype=np.float32)
        baseline = compute_metrics(self.first_prediction, self.truth)
        self.plan = dict(train_rows=12, val_rows=60,
                         training_and_setup_budget_minutes=1,
                         baseline_metrics=baseline)
        self.callback = BudgetSelection(time.monotonic(), self.out, self.plan,
                                        self.truth, torch.zeros(4), torch.ones(4))
        self.trainer = FakeTrainer()
        self.module = types.SimpleNamespace(validation_predictions=[], validation_targets=[], adaptive=None)

    def prepare_epoch(self, prediction=None, truth=None, batches=(6, 6)):
        self.callback.on_train_epoch_start(self.trainer, self.module)
        for i, rows in enumerate(batches):
            self.callback.on_train_batch_end(self.trainer, self.module, None,
                                             {"target": torch.zeros(rows, 4)}, i)
        p = self.first_prediction if prediction is None else prediction
        t = self.truth if truth is None else truth
        self.module.validation_predictions = [torch.as_tensor(p[:30]).clone(), torch.as_tensor(p[30:]).clone()]
        self.module.validation_targets = [torch.as_tensor(t[:30]).clone(), torch.as_tensor(t[30:]).clone()]

    def finish_epoch(self):
        # Avoid printing full diagnostics for each intentionally tiny test.
        with contextlib.redirect_stdout(io.StringIO()):
            self.callback.on_validation_end(self.trainer, self.module)

    def assert_no_selection_or_checkpoint(self):
        self.assertEqual(self.callback.completed, 0)
        self.assertEqual(self.callback.history, [])
        self.assertEqual(self.callback.best, {})
        self.assertEqual(self.trainer.saved, [])
        self.assertEqual(list((self.out / "checkpoints").iterdir()), [])
        self.assertFalse((self.out / "checkpoint_selection.json").exists())

    def test_partial_epoch_excluded_when_row_count_or_batch_count_wrong(self):
        # First case has all expected batches but too few events. The second
        # has every expected event in one batch but is missing a train batch.
        for batches in ((6, 5), (12,)):
            with self.subTest(batches=batches):
                self.prepare_epoch(batches=batches)
                with patch("training_control.compute_metrics", side_effect=AssertionError("partial epoch evaluated")):
                    self.finish_epoch()
                self.assertTrue(self.callback.partial)
                self.assert_no_selection_or_checkpoint()
                self.assertEqual(self.module.validation_predictions, [])
                self.assertEqual(self.module.validation_targets, [])

    def test_incomplete_validation_fails_without_checkpoint(self):
        self.prepare_epoch(prediction=self.first_prediction[:-1], truth=self.truth[:-1])
        with self.assertRaisesRegex(RuntimeError, "Incomplete validation"):
            self.finish_epoch()
        self.assert_no_selection_or_checkpoint()

    def test_validation_event_permutation_fails_without_checkpoint(self):
        # Same dimensions and multiset of truth values must not hide wrong
        # event ordering. This is more than a row-count check.
        permuted = self.truth.copy()
        permuted[[0, 1]] = permuted[[1, 0]]
        self.prepare_epoch(truth=permuted)
        with self.assertRaisesRegex(RuntimeError, "row alignment"):
            self.finish_epoch()
        self.assert_no_selection_or_checkpoint()

    def test_eligible_primary_survives_better_mse_and_shape_with_worse_trace(self):
        self.prepare_epoch()
        self.finish_epoch()
        self.assertEqual(self.callback.completed, 1)
        self.assertTrue(self.callback.best["primary"]["selection"]["eligible"])
        self.assertEqual(set(self.callback.best), {"primary", "score", "mse"})
        self.assertEqual(len(self.trainer.saved), 3)
        initial_score = self.callback.best["score"]["selection"]["joint_score"]
        initial_mse = self.callback.best["mse"]["mse"]

        # This epoch has exactly the truth marginals, but exchanges the cosK
        # predictions of two events across bins. Thus excellent shape and MSE
        # cannot compensate for failing one of the protected Trace metrics.
        later = self.truth.copy()
        later[[0, 3], 2] = later[[3, 0], 2]
        self.trainer.current_epoch = 1
        self.trainer.global_step = 4
        self.prepare_epoch(prediction=later)
        self.finish_epoch()

        self.assertEqual(self.callback.completed, 2)
        self.assertFalse(self.callback.history[-1]["selection"]["eligible"])
        self.assertFalse(self.callback.history[-1]["selection"]["checks"]["cos_K_plus/trace"])
        self.assertLess(self.callback.history[-1]["selection"]["joint_score"], initial_score)
        self.assertLess(self.callback.history[-1]["mse"], initial_mse)
        self.assertEqual(self.callback.best["primary"]["epoch"], 0)
        self.assertEqual(self.callback.best["score"]["epoch"], 1)
        self.assertEqual(self.callback.best["mse"]["epoch"], 1)
        self.assertEqual(len(self.trainer.saved), 5)
        for label, epoch in (("primary", 0), ("score", 1), ("mse", 1)):
            restored = json.loads((self.out / "checkpoints" / (label + ".ckpt")).read_text(encoding="utf-8"))
            self.assertEqual(restored["epoch"], epoch)
        self.assertEqual(list((self.out / "checkpoints").glob("*.tmp")), [])
        saved_selection = json.loads((self.out / "checkpoint_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_selection["primary"]["epoch"], 0)

    def test_budget_expiry_requests_stop_and_partial_epoch_stays_excluded(self):
        self.callback.started = time.monotonic() - 61.0
        self.prepare_epoch(batches=(6,))
        self.assertTrue(self.trainer.should_stop)
        self.finish_epoch()
        self.assert_no_selection_or_checkpoint()

    def test_sanity_validation_cannot_create_a_selected_checkpoint(self):
        # Lightning sanity validation can run without any train batches.
        self.trainer.sanity_checking = True
        self.finish_epoch()
        self.assert_no_selection_or_checkpoint()


if __name__ == "__main__":
    unittest.main()
