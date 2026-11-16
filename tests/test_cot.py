"""Numerical and gradient checks for the fixed-group projection objective."""

import json
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from ttbar_reco import GroupDefinition, assign_groups, conditional_ot_loss, fit_groups, make_directions


class ConditionalOTTests(unittest.TestCase):
    def test_reference_json_roundtrip_and_fixed_edges(self):
        reference = fit_groups(np.arange(100, dtype=np.float32))
        restored = GroupDefinition.from_dict(json.loads(json.dumps(reference.to_dict())))
        self.assertEqual(reference, restored)
        self.assertEqual(reference.counts, (60, 30, 10))
        edges = torch.tensor(restored.internal_edges)
        self.assertEqual(assign_groups(edges, restored).tolist(), [1, 2])
        before = restored.to_dict()
        self.assertEqual(assign_groups([-100.0, 1000.0], restored).tolist(), [0, 2])
        self.assertEqual(restored.to_dict(), before)

    def test_tied_training_quantiles_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_groups(np.ones(100))
        with self.assertRaises(ValueError):
            fit_groups(np.arange(10), quantiles=[np.nan])

    def test_reference_cannot_be_relabelled_as_validation(self):
        state = fit_groups(np.arange(100)).to_dict()
        state['fit_split'] = 'validation'
        with self.assertRaises(ValueError):
            GroupDefinition.from_dict(state)

    def test_directions_are_fixed_unit_vectors_with_private_rng(self):
        before = torch.random.get_rng_state().clone()
        directions = make_directions()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.testing.assert_close(directions.norm(dim=1), torch.ones(32))
        torch.testing.assert_close(directions, make_directions())

    def test_translation_cost_and_prediction_gradient(self):
        target = torch.arange(24, dtype=torch.float32).reshape(6, 4).requires_grad_()
        shift = torch.tensor([1.0, 2.0, 3.0, 4.0])
        prediction = (target.detach() + shift).requires_grad_()
        value = conditional_ot_loss(prediction, target, torch.zeros(6, dtype=torch.long), [1.0], torch.eye(4))
        torch.testing.assert_close(value, shift.square().mean())
        value.backward()
        torch.testing.assert_close(prediction.grad, 2 * shift.expand_as(prediction) / prediction.numel())
        self.assertIsNone(target.grad)

    def test_cross_group_exchange_is_detected(self):
        target = torch.tensor([[0.] * 4, [1.] * 4, [10.] * 4, [11.] * 4])
        prediction = target[[2, 3, 0, 1]]
        global_cost = conditional_ot_loss(prediction, target, torch.zeros(4, dtype=torch.long), [1.], torch.eye(4))
        grouped_cost = conditional_ot_loss(prediction, target, torch.tensor([0, 0, 1, 1]), [.5, .5], torch.eye(4))
        self.assertEqual(global_cost.item(), 0.)
        self.assertEqual(grouped_cost.item(), 100.)

    def test_within_group_permutation_does_not_replace_paired_loss(self):
        target = torch.tensor([[0.] * 4, [1.] * 4, [10.] * 4, [11.] * 4])
        prediction = target[[1, 0, 3, 2]].clone()
        original = prediction.clone()
        cost = conditional_ot_loss(prediction, target, torch.tensor([0, 0, 1, 1]), [.5, .5], torch.eye(4))
        self.assertEqual(cost.item(), 0.)
        self.assertGreater((prediction - target).square().mean().item(), 0.)
        torch.testing.assert_close(prediction, original)

    def test_missing_singleton_groups_keep_training_weights(self):
        target = torch.zeros(3, 4)
        prediction = torch.ones(3, 4, requires_grad=True)
        cost = conditional_ot_loss(prediction, target, torch.tensor([0, 0, 1]), [.2, .3, .5], torch.eye(4))
        self.assertAlmostEqual(cost.item(), .2, places=6)
        cost.backward()
        self.assertEqual(prediction.grad[2].abs().sum().item(), 0.)

    def test_all_singletons_produce_differentiable_zero(self):
        prediction = torch.ones(3, 4, requires_grad=True)
        cost = conditional_ot_loss(prediction, torch.zeros_like(prediction), torch.arange(3), [.6, .3, .1], make_directions())
        self.assertEqual(cost.item(), 0.)
        cost.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_identical_tied_samples_zero_and_finite_gradient(self):
        prediction = torch.zeros(6, 4, requires_grad=True)
        value = conditional_ot_loss(prediction, prediction.detach().clone(), torch.zeros(6, dtype=torch.long), [1.], make_directions())
        value.backward()
        self.assertEqual(value.item(), 0.)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_mixed_precision_loss_accumulates_in_float32(self):
        prediction = torch.ones(4, 4, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            cost = conditional_ot_loss(prediction, torch.zeros_like(prediction), torch.zeros(4, dtype=torch.long), [1.], torch.eye(4))
        self.assertEqual(cost.dtype, torch.float32)
        self.assertEqual(cost.item(), 1.)
        cost.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_invalid_weights_and_group_ids_rejected(self):
        p = torch.zeros(4, 4)
        for weights in ([.5], [float('nan')], [-1.]):
            with self.assertRaises(ValueError):
                conditional_ot_loss(p, p, torch.zeros(4, dtype=torch.long), weights, torch.eye(4))
        with self.assertRaises(ValueError):
            conditional_ot_loss(p, p, torch.ones(4, dtype=torch.long), [1.], torch.eye(4))


class OriginalStudyParityTests(unittest.TestCase):
    def test_random_forward_gradient_and_group_reference_match_recorded_study(self):
        path = Path(__file__).resolve().parents[1] / 'experiments' / 'methods_sweep_20261112' / 'ot_proxy.py'
        spec = importlib.util.spec_from_file_location('recorded_study_ot', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch.testing.assert_close(make_directions(), original.make_directions(), rtol=0, atol=0)
        # Include ordinary batches, absent groups, singleton groups and no valid group.
        patterns = ((128, None), (31, None), (4, [0, 0, 1, 1]),
                    (3, [0, 0, 1]), (3, [0, 1, 2]))
        for seed in range(20):
            generator = torch.Generator().manual_seed(101 + seed)
            mass = np.random.default_rng(seed).lognormal(6.2, .3, size=1003).astype(np.float32)
            reference = fit_groups(mass)
            self.assertEqual(reference.to_dict(), original.fit_groups(mass))
            torch.testing.assert_close(
                assign_groups(torch.from_numpy(mass), reference),
                original.assign_groups(torch.from_numpy(mass), reference.internal_edges),
                rtol=0, atol=0,
            )
            directions = make_directions(seed=4201 + seed)
            for size, labels in patterns:
                with self.subTest(seed=seed, size=size, labels=labels):
                    values = torch.randn(size, 4, generator=generator)
                    target = torch.randn(size, 4, generator=generator)
                    groups = (torch.randint(3, (size,), generator=generator)
                              if labels is None else torch.tensor(labels))
                    prediction = values.clone().requires_grad_()
                    recorded_prediction = values.clone().requires_grad_()
                    actual = conditional_ot_loss(prediction, target, groups, reference.probabilities, directions)
                    expected = original.conditional_ot_loss(
                        recorded_prediction, target, groups, reference.probabilities, directions,
                    )
                    actual.backward()
                    expected.backward()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(prediction.grad, recorded_prediction.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
