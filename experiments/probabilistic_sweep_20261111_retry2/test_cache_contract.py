"""Small CPU checks for frozen feature extraction and density evaluation.

Run with ``python test_cache_contract.py --repo /path/to/HepSetReg``.
The local archived teacher source is used only when --repo is omitted.
No physics data, checkpoints, network connection, or GPU is required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE_HASHES = {
    "head": "9ccac7577cd580c4fd9c827bff15b47473f58c75fb341365efa47ecbe874855f",
    "backbone": "990cc0ba49e8aa3ae99000ade27e54b98ca2cb431c9a693ac85e73b60d69653b",
}
REPO = os.environ.get("PROB_SWEEP_REPO")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trusted_source(name):
    if REPO:
        path = Path(REPO) / "src/hepsetreg/backbones" / f"{name}.py"
    else:
        path = (HERE.parent / "PHYS30881 S5 Project/tmp/teacher_data_audit/github_inventory"
                / f"src__hepsetreg__backbones__{name}.py")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_HASHES[name]:
        raise AssertionError(f"Teacher source SHA256 mismatch: {path}")
    return load_file(f"_cache_contract_{name}", path)


class FrozenCacheContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone_module = trusted_source("backbone")
        cls.head_module = trusted_source("head")
        cls.cache = load_file("_cache_contract_cache", HERE / "cache_features.py")

    def setUp(self):
        torch.manual_seed(1941)
        backbone = self.backbone_module.TransformerBackbone(
            n_global_features=6, n_jet_features=9, n_lepton_features=10,
            d_model=16, nhead=4, num_layers=2, dropout=0.5,
        )
        head = self.head_module.RegressionHead(
            d_model=16, output_dim=4, n_layers=2, start_neurons=16, dropout=0.5,
        )
        self.model = self.head_module.EventRegressor(backbone, head)
        self.model.eval().requires_grad_(False)
        self.batch = {
            "global": torch.randn(7, 6),
            "jet": torch.randn(7, 5, 9),
            "jet_mask": torch.tensor([[1, 1, 0, 0, 0]] * 7, dtype=torch.bool),
            "lepton": torch.randn(7, 2, 10),
            "lepton_mask": torch.ones(7, 2, dtype=torch.bool),
            "target": torch.randn(7, 4),
        }

    def extract(self, batch=None):
        return self.cache.extract_batch(self.model, batch or self.batch, torch.device("cpu"))

    def test_embedding_then_head_matches_teacher_forward(self):
        h, pred = self.extract()
        with torch.inference_mode():
            direct = self.model(self.batch).numpy()
            embedding = self.model.backbone(self.batch).numpy()
        np.testing.assert_allclose(h, embedding, rtol=0, atol=0)
        np.testing.assert_allclose(pred, direct, rtol=0, atol=0)
        self.assertEqual(h.shape, (7, 16))
        self.assertEqual(pred.shape, (7, 4))
        self.assertEqual(h.dtype, np.float32)
        self.assertEqual(pred.dtype, np.float32)
        self.assertTrue(np.isfinite(h).all() and np.isfinite(pred).all())

    def test_target_poisoning_and_nonreco_fields_are_excluded(self):
        seen = []
        hook = self.model.backbone.register_forward_pre_hook(
            lambda module, args: seen.append(set(args[0]))
        )
        try:
            first = self.extract()
            poisoned = dict(self.batch)
            poisoned["target"] = torch.full((7, 4), float("nan"))
            poisoned["truth_mass"] = torch.full((7,), float("inf"))
            poisoned["row_index"] = torch.arange(7)
            second = self.extract(poisoned)
        finally:
            hook.remove()
        expected = {"global", "jet", "jet_mask", "lepton", "lepton_mask"}
        self.assertEqual(seen, [expected, expected])
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_eval_disables_dropout_and_keeps_weights_frozen(self):
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        first = self.extract()
        second = self.extract()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        for parameter in self.model.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(all(not module.training for module in self.model.modules()))

    def test_training_mode_is_rejected(self):
        self.model.train()
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            self.extract()

    def test_masked_padding_does_not_change_features(self):
        first = self.extract()
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["jet"][~changed["jet_mask"]] = 10000.0
        second = self.extract(changed)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b, rtol=0, atol=2e-6)

    def test_nondivisible_batches_preserve_every_row_and_order(self):
        full = self.extract()
        pieces = [self.extract({key: value[start:start + 3] for key, value in self.batch.items()})
                  for start in range(0, 7, 3)]
        for index in range(2):
            joined = np.concatenate([part[index] for part in pieces], axis=0)
            self.assertEqual(joined.shape, full[index].shape)
            np.testing.assert_allclose(joined, full[index], rtol=1e-6, atol=2e-6)

    def test_loss_module_checkpoint_namespace_is_strictly_loadable(self):
        state = {f"model.{key}": value.clone() for key, value in self.model.state_dict().items()}
        state["directions"] = torch.randn(32, 4)
        stripped = {key.removeprefix("model."): value for key, value in state.items()
                    if key.startswith("model.")}
        result = self.model.load_state_dict(stripped, strict=True)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        del stripped["head.net.0.weight"]
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict(stripped, strict=True)

    def test_physical_transform_known_values_preserves_original_truth(self):
        truth = np.array([[np.e, -0.4, 0.0, np.tanh(0.5)],
                          [np.e ** 2, 0.6, -1.0, 1.0]], dtype=np.float64)
        original = truth.copy()
        transformed, report = self.cache.physical_transform(truth)
        np.testing.assert_allclose(transformed[0], [1.0, -0.4, 0.0, 0.5], atol=1e-14)
        np.testing.assert_allclose(transformed[1, :2], [2.0, 0.6], atol=1e-14)
        np.testing.assert_array_equal(truth, original)
        self.assertEqual(report['atanh_endpoint_adjustments'], [1, 1])
        self.assertFalse(report['original_truth_modified'])
        self.assertTrue(np.isfinite(transformed).all())

    def test_invalid_physical_truth_is_rejected(self):
        valid = np.array([[500.0, 0.2, 0.1, -0.5]])
        for column, value in [(0, 0.0), (0, -1.0), (1, np.nan), (2, 1.001)]:
            broken = valid.copy()
            broken[0, column] = value
            with self.assertRaises((AssertionError, ValueError)):
                self.cache.physical_transform(broken)


class DensityEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_file("_cache_contract_evaluation", HERE / "evaluate_density.py")

    def setUp(self):
        self.transforms = dict(keys=list(self.evaluation.KEYS),
                               transform=['log', 'identity', 'atanh', 'atanh'],
                               mean=[np.log(500.0), 0.0, 0.0, 0.0],
                               std=[0.2, 0.8, 0.9, 0.9], atanh_epsilon=1e-7)
        # Every fixed truth bin is represented, including the high-mass row.
        self.truth = np.array([[340, -1.7, -.75, .75],
                               [410, -.75, -.25, .25],
                               [470, -.25, .25, -.25],
                               [600, .25, .75, -.75],
                               [1000, .75, -.75, .75],
                               [1500, 1.7, .75, -.75]], dtype=np.float64)

    def test_inverse_transform_known_values_and_sample_dimensions(self):
        u = np.zeros((2, 3, 4), dtype=np.float64)
        u[1, 2] = [1.0, -1.0, 1.0, -1.0]
        physical = self.evaluation.inverse_transform(u, self.transforms)
        np.testing.assert_allclose(physical[0], [[500., 0., 0., 0.]] * 3, atol=1e-12)
        np.testing.assert_allclose(physical[1, 2],
                                   [500 * np.exp(.2), -.8, np.tanh(.9), -np.tanh(.9)],
                                   rtol=1e-13, atol=1e-13)
        self.assertEqual(physical.shape, (2, 3, 4))
        self.assertEqual(physical.dtype, np.float64)

    def test_inverse_transform_rejects_overflow_without_clipping(self):
        u = np.zeros((1, 4))
        u[0, 0] = 10000.
        with self.assertRaises(FloatingPointError):
            self.evaluation.inverse_transform(u, self.transforms)

    def test_trace_keeps_outside_prediction_in_truth_denominator(self):
        prediction = self.truth.copy()
        prediction[0, 0] = 2500.0
        result = self.evaluation.compute_metrics(prediction, self.truth)['observables']['ttbar_mass']
        self.assertAlmostEqual(result['local_trace'], .8)
        self.assertEqual(result['truth_bin_counts'], [1, 1, 1, 1, 2])
        self.assertEqual(result['prediction_outside_given_truth_bin'], [1, 0, 0, 0, 0])
        self.assertEqual(sum(result['row_probabilities'][0]), 0.)
        # Empty rows are explicitly rejected, not silently omitted from Trace.
        with self.assertRaises(ValueError):
            self.evaluation.compute_metrics(self.truth[1:], self.truth[1:])

    def test_resolution_is_full_interquantile_width(self):
        prediction = self.truth.copy()
        prediction[:, 0] += np.arange(6)
        result = self.evaluation.compute_metrics(prediction, self.truth)['observables']['ttbar_mass']
        self.assertAlmostEqual(result['res'], 3.413, places=12)
        self.assertAlmostEqual(result['q1587'], .7935, places=12)
        self.assertAlmostEqual(result['q8413'], 4.2065, places=12)

    def test_score_has_twelve_components_and_correct_directions(self):
        baseline = dict(observables={key: dict(res=2., local_trace=.5, w1=4.)
                                     for key in self.evaluation.KEYS})
        improved = dict(observables={key: dict(res=1., local_trace=1., w1=2.)
                                     for key in self.evaluation.KEYS})
        equal = self.evaluation.selection_score(baseline, baseline)
        better = self.evaluation.selection_score(improved, baseline)
        self.assertEqual(equal['score'], 0.)
        self.assertEqual(len(better['components']), 12)
        self.assertEqual(better['count'], 12)
        self.assertAlmostEqual(better['score'], -np.log(2.))
        self.assertAlmostEqual(better['geometric_ratio'], .5)
        self.assertTrue(all(value['log_ratio'] < 0 for value in better['components'].values()))
        worsened = dict(observables={key: value.copy() for key, value in baseline['observables'].items()})
        worsened['observables']['cos_phi']['w1'] = 8.
        self.assertAlmostEqual(self.evaluation.selection_score(worsened, baseline)['score'], np.log(2.) / 12)

    def test_eligibility_requires_all_eight_resolution_and_trace_bounds(self):
        with patch.object(sys, 'path', [str(HERE), *sys.path]):
            runner = load_file('_cache_contract_runner', HERE / 'run_density.py')
        reference = {key: dict(res=2., local_trace=.5, w1=4.) for key in self.evaluation.KEYS}
        equal = dict(observables={key: value.copy() for key, value in reference.items()})
        qualified, checks = runner.eligible(equal, reference)
        self.assertTrue(qualified)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(checks.values()))
        for key in self.evaluation.KEYS:
            for metric, worse_value in [('res', 2.001), ('local_trace', .4999)]:
                worse = dict(observables={name: value.copy() for name, value in reference.items()})
                worse['observables'][key][metric] = worse_value
                qualified, checks = runner.eligible(worse, reference)
                self.assertFalse(qualified, f'{key}/{metric}')
                self.assertFalse(checks[f'{key}/{metric}'])
        # Shape remains an explicit separate score component, not a hidden eligibility cut.
        equal['observables']['cos_phi']['w1'] = 1e9
        self.assertTrue(runner.eligible(equal, reference)[0])

    def test_full_evaluation_keeps_independent_sample_sets_and_physical_summaries(self):
        class RandomHead(torch.nn.Module):
            def sample(self, condition, num_samples, generator=None):
                return torch.randn(len(condition), num_samples, 4,
                                   device=condition.device, generator=generator)

        condition = np.zeros((6, 132), dtype=np.float32)
        point = self.truth + np.arange(1, 7)[:, None] * np.array([.1, .01, .01, .01])
        cache = dict(transforms=self.transforms,
                     arrays=dict(val_truth=self.truth, val_point=point,
                                 val_condition=condition, row_index_val=np.arange(6)))
        saved = {}

        def capture(path, prediction, truth, row_index):
            saved[Path(path).stem] = np.array(prediction, copy=True)
            np.testing.assert_array_equal(truth, self.truth)
            np.testing.assert_array_equal(row_index, np.arange(6))

        head = RandomHead()
        rng_before = torch.get_rng_state().clone()
        # Keep generated artifacts inside this explicitly authorized workspace.
        with tempfile.TemporaryDirectory(prefix='_density_contract_', dir=HERE) as directory:
            with patch.object(self.evaluation, 'write_predictions', side_effect=capture):
                result = self.evaluation.evaluate_full(
                    head, cache, directory, torch.device('cpu'), sample_batch_size=2,
                    single_draw_seed=4203, posterior_seed=4204,
                )
            posterior = np.load(Path(directory) / 'posterior_samples32.npy')
            replay = self.evaluation.sample_single(head, condition, self.transforms,
                                                   torch.device('cpu'), seed=4203, batch_size=2)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        np.testing.assert_array_equal(saved['predictions_single_draw'], replay)
        self.assertFalse(np.array_equal(replay, posterior[:, 0, :]))
        np.testing.assert_array_equal(saved['predictions_mean32'], posterior.mean(axis=1, dtype=np.float64))
        np.testing.assert_array_equal(saved['predictions_median32'], np.median(posterior.astype(np.float64), axis=1))
        # For positive mass, this also excludes inverse(mean(log(m))) as the reported mean.
        geometric_mass = np.exp(np.log(posterior[:, :, 0].astype(np.float64)).mean(axis=1))
        self.assertTrue(np.all(saved['predictions_mean32'][:, 0] > geometric_mass))
        self.assertTrue(result['sampling']['single_draw_is_separate_from_pool'])
        self.assertEqual(result['posterior_pool32']['samples_per_event'], 32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo")
    args, remaining = parser.parse_known_args()
    REPO = args.repo or REPO
    unittest.main(argv=[sys.argv[0], *remaining])
