"""Numerical-property checks; run with ``python test_losses.py`` (CPU is enough)."""
from __future__ import annotations

import math
import unittest

import torch

from losses import emd_loss, make_directions, point_loss, sliced_wasserstein_loss


class LossProperties(unittest.TestCase):
    def setUp(self):
        self.directions = make_directions()
        self.pred = torch.tensor([[1.0, -2.0, 0.2, 0.8], [0.0, 3.0, -0.7, -0.1], [-1.0, 0.5, 0.6, -0.4]])
        self.target = torch.tensor([[0.3, -1.0, 0.1, 0.5], [0.2, 1.0, -0.5, -0.3], [-0.4, 0.2, 0.7, -0.8]])
        self.functions = {
            "mse": lambda p, t: point_loss(p, t, "mse"),
            "huber_05": lambda p, t: point_loss(p, t, "huber", delta=0.5),
            "huber_1": lambda p, t: point_loss(p, t, "huber", delta=1.0),
            "logcosh": lambda p, t: point_loss(p, t, "logcosh"),
            "emd": emd_loss,
            "sw1": lambda p, t: sliced_wasserstein_loss(p, t, self.directions),
        }

    def test_finite_scalar_and_gradients(self):
        for name, function in self.functions.items():
            with self.subTest(name=name):
                pred = self.pred.clone().requires_grad_()
                loss = function(pred, self.target)
                self.assertEqual(loss.ndim, 0)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreaterEqual(loss.item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())

    def test_equal_inputs_have_zero_loss_and_gradient(self):
        for name, function in self.functions.items():
            with self.subTest(name=name):
                pred = self.pred.clone().requires_grad_()
                loss = function(pred, self.pred)
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_distributions_ignore_event_permutation_but_point_losses_do_not(self):
        permutation = torch.tensor([2, 0, 1])
        target = self.pred[permutation]
        self.assertEqual(emd_loss(self.pred, target).item(), 0.0)
        # BLAS may round identical rows differently after a row permutation.
        self.assertLess(sliced_wasserstein_loss(self.pred, target, self.directions).item(), 1e-6)
        for kind in ("mse", "huber", "logcosh"):
            self.assertGreater(point_loss(self.pred, target, kind).item(), 0)

    def test_sw1_can_distinguish_equal_marginals_with_different_joint_structure(self):
        pred = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])
        target = torch.tensor([[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]])
        self.assertEqual(emd_loss(pred, target).item(), 0.0)
        direction = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        loss = sliced_wasserstein_loss(pred, target, direction)
        self.assertAlmostEqual(loss.item(), math.sqrt(2), places=6)
        torch.testing.assert_close(loss, sliced_wasserstein_loss(pred, target, 3 * direction))

    def test_huber_quadratic_and_small_delta_limits(self):
        # Standard Huber, not SmoothL1: the quadratic branch is 0.5*MSE.
        mse = point_loss(self.pred, self.target, "mse")
        torch.testing.assert_close(point_loss(self.pred, self.target, "huber", delta=100), 0.5 * mse)
        # Huber(delta)/delta approaches MAE as delta tends to zero.
        delta = 1e-4
        scaled_huber = point_loss(self.pred, self.target, "huber", delta=delta) / delta
        mae = (self.pred - self.target).abs().mean()
        self.assertLessEqual(abs(scaled_huber.item() - mae.item()), delta)

    def test_logcosh_extreme_residuals_remain_finite_and_have_bounded_gradients(self):
        pred = torch.tensor([[-1e20, -1e4, 0.0, 1e4, 1e20]], requires_grad=True)
        loss = point_loss(pred, torch.zeros_like(pred), "logcosh")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        torch.testing.assert_close(pred.grad, pred.detach().sign() / pred.numel())
        # At large residual, scaled log-cosh approaches scale*|r|-scale^2*ln(2).
        large = torch.full((2, 4), 100.0)
        value = point_loss(large, torch.zeros_like(large), "logcosh", scale=2)
        self.assertAlmostEqual(value.item(), 200.0 - 4.0 * math.log(2.0), places=4)

    def test_directions_are_unit_fixed_and_do_not_change_global_rng(self):
        state = torch.random.get_rng_state().clone()
        first = make_directions(32, 4, seed=4201)
        second = make_directions(32, 4, seed=4201)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(first.shape, (32, 4))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(first.device.type, "cpu")
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, make_directions(32, 4, seed=4202)))
        torch.testing.assert_close(torch.linalg.vector_norm(first, dim=1), torch.ones(32))
        module = torch.nn.Module()
        module.register_buffer("sw_directions", first)
        self.assertIn("sw_directions", module.state_dict())
        self.assertEqual(list(module.parameters()), [])

    def test_autocast_does_not_reduce_loss_arithmetic_precision(self):
        for name, function in self.functions.items():
            with self.subTest(name=name):
                pred = self.pred.to(torch.bfloat16).requires_grad_()
                target = self.target.to(torch.bfloat16)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    loss = function(pred, target)
                self.assertEqual(loss.dtype, torch.float32)
                torch.testing.assert_close(loss, function(pred.float(), target.float()))
                loss.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_shapes_and_parameters_are_rejected(self):
        for function in self.functions.values():
            with self.assertRaises(ValueError):
                function(self.pred, self.target[:, :2])
            with self.assertRaises(ValueError):
                function(self.pred.flatten(), self.target.flatten())
            with self.assertRaises(ValueError):
                function(self.pred[:0], self.target[:0])
            with self.assertRaises(TypeError):
                function(self.pred.long(), self.target.long())
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                point_loss(self.pred, self.target, "huber", delta=value)
            with self.assertRaises(ValueError):
                point_loss(self.pred, self.target, "logcosh", scale=value)
        with self.assertRaises(ValueError):
            point_loss(self.pred, self.target, "unknown")
        for directions in (torch.ones(32, 3), torch.empty(0, 4), torch.zeros(1, 4), torch.full((1, 4), float("nan"))):
            with self.assertRaises(ValueError):
                sliced_wasserstein_loss(self.pred, self.target, directions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
