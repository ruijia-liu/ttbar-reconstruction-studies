"""CPU checks; --build-table regenerates the quadrature table deterministically."""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import k1
import torch

from adaptive_loss import ALPHA_MAX, ALPHA_MIN, AdaptiveRobustLoss


def rho_numpy(x, alpha):
    beta = 2.0 - alpha
    return (beta / alpha) * np.expm1(0.5 * alpha * np.log1p(x * x / beta))


def reference_partition(alpha):
    """Integrate Z and dZ/dalpha over the entire real line in float64."""
    beta = 2.0 - alpha

    def integrand(x, derivative=False):
        logarithm = math.log1p(x * x / beta)
        exponent = 0.5 * alpha * logarithm
        excess = math.expm1(exponent)
        rho = beta / alpha * excess
        if rho > 740.0:
            return 0.0
        density = math.exp(-rho)
        if not derivative:
            return density
        d_logarithm = (x * x / (beta + x * x)) / beta
        d_rho = (-2.0 / (alpha * alpha)) * excess + (beta / alpha) * math.exp(exponent) * (
            0.5 * logarithm + 0.5 * alpha * d_logarithm
        )
        return -density * d_rho

    half_z, error_z = quad(integrand, 0.0, np.inf, epsabs=2e-11, epsrel=2e-11, limit=400)
    half_dz, error_dz = quad(lambda x: integrand(x, True), 0.0, np.inf,
                            epsabs=2e-11, epsrel=2e-11, limit=400)
    if max(error_z, error_dz) > 2e-8:
        raise RuntimeError("Quadrature did not achieve required precision")
    return math.log(2.0 * half_z), half_dz / half_z


def build_table():
    alpha = np.linspace(ALPHA_MIN, ALPHA_MAX, 361, dtype=np.float64)
    result = np.asarray([reference_partition(float(a)) for a in alpha])
    np.savez(Path(__file__).with_name("barron_partition.npz"),
             alpha=alpha, log_z=result[:, 0], d_log_z=result[:, 1])
    print("Built 361-knot partition table from full-domain quadrature")


class AdaptiveLossTests(unittest.TestCase):
    def test_initialisation_does_not_change_rng(self):
        before = torch.random.get_rng_state().clone()
        module = AdaptiveRobustLoss()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.testing.assert_close(module.alpha(), torch.ones(4), rtol=0, atol=1e-7)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 4)

    def test_alpha_one_matches_pseudo_huber_and_analytic_partition(self):
        residual = torch.tensor([-100.0, -3.0, -0.01, 0.0, 0.01, 3.0, 100.0], dtype=torch.float64)
        actual = AdaptiveRobustLoss.rho(residual, torch.tensor(1.0, dtype=torch.float64))
        expected = torch.sqrt(1.0 + residual.square()) - 1.0
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-14)
        # Z(1)=2*e*K_1(1), independently known special case.
        value, _ = reference_partition(1.0)
        self.assertAlmostEqual(value, math.log(2.0 * math.e * k1(1.0)), places=10)

    def test_partition_values_and_gradients_all_knots_and_midpoints(self):
        module = AdaptiveRobustLoss()
        knots = np.linspace(ALPHA_MIN, ALPHA_MAX, 361)
        values = np.sort(np.concatenate((knots, (knots[:-1] + knots[1:]) / 2.0)))
        alpha = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        estimated = module.log_partition(alpha)
        gradient, = torch.autograd.grad(estimated.sum(), alpha)
        references = np.asarray([reference_partition(float(a)) for a in values])
        value_error = np.max(np.abs(estimated.detach().numpy() - references[:, 0]))
        gradient_error = np.max(np.abs(gradient.detach().numpy() - references[:, 1]))
        print(f"partition checks: max abs value error={value_error:.3g}, gradient error={gradient_error:.3g}")
        self.assertLess(value_error, 3e-7)
        self.assertLess(gradient_error, 3e-6)

    def test_local_curvature_and_zero_prediction_gradient(self):
        for raw in (-6.0, 0.0, 6.0):
            module = AdaptiveRobustLoss(dimensions=1)
            module.raw_alpha.data.fill_(raw)
            prediction = torch.zeros(1, 1, requires_grad=True)
            value = module(prediction, torch.zeros_like(prediction))
            first, = torch.autograd.grad(value, prediction, create_graph=True)
            second, = torch.autograd.grad(first.sum(), prediction)
            torch.testing.assert_close(first, torch.zeros_like(first), rtol=0, atol=0)
            torch.testing.assert_close(second, torch.full_like(second, 2.0), rtol=1e-6, atol=1e-6)

    def test_normalisation_prevents_trivial_downward_alpha_gradient(self):
        module = AdaptiveRobustLoss()
        prediction = torch.zeros(32, 4, requires_grad=True)
        module(prediction, torch.zeros_like(prediction)).backward()
        # At zero residual rho has no alpha gradient. logZ decreases with
        # alpha, favouring larger alpha instead of unconditionally relaxing it.
        self.assertTrue(bool((module.raw_alpha.grad < 0).all()))
        self.assertTrue(bool(torch.isfinite(module.raw_alpha.grad).all()))

    def test_forward_and_gradients_match_numeric_normalised_likelihood(self):
        module = AdaptiveRobustLoss()
        module.raw_alpha.data.copy_(torch.tensor([-1.0, -0.3, 0.4, 1.0]))
        prediction = torch.tensor([[0.1, 1.0, -3.0, 20.0], [-0.5, -1.0, 0.2, 5.0]], requires_grad=True)
        value = module(prediction, torch.zeros_like(prediction))
        alpha = module.alpha().detach().numpy().astype(float)
        expected = 2.0 * np.mean(rho_numpy(prediction.detach().numpy().astype(float), alpha)
                                 + np.array([reference_partition(a)[0] for a in alpha]))
        self.assertAlmostEqual(value.item(), expected, delta=3e-6)
        value.backward()
        numeric = []
        raw = module.raw_alpha.detach().numpy().astype(float)
        residual = prediction.detach().numpy().astype(float)
        for j in range(4):
            def objective(offset):
                changed = raw.copy()
                changed[j] += offset
                a = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / (1.0 + np.exp(-changed))
                return 2.0 * np.mean(rho_numpy(residual, a)
                                     + np.array([reference_partition(x)[0] for x in a]))
            numeric.append((objective(1e-4) - objective(-1e-4)) / 2e-4)
        np.testing.assert_allclose(module.raw_alpha.grad.numpy(), numeric, rtol=1e-4, atol=2e-6)

    def test_extreme_residual_and_bfloat16_autocast_gradients(self):
        for raw in (-20.0, 0.0, 20.0):
            module = AdaptiveRobustLoss()
            module.raw_alpha.data.fill_(raw)
            pred = torch.tensor([[0.0, 1e-6, -1000.0, 1e6]], requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                value = module(pred, torch.zeros_like(pred))
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(value)))
            value.backward()
            self.assertTrue(bool(torch.isfinite(pred.grad).all()))
            self.assertTrue(bool(torch.isfinite(module.raw_alpha.grad).all()))
        module = AdaptiveRobustLoss()
        value = module(torch.ones(3, 4, dtype=torch.bfloat16), torch.zeros(3, 4, dtype=torch.bfloat16))
        self.assertEqual(value.dtype, torch.float32)

    def test_checkpoint_restores_loss_and_alpha(self):
        module = AdaptiveRobustLoss()
        module.raw_alpha.data.copy_(torch.tensor([-2.0, -1.0, 0.0, 2.0]))
        memory = io.BytesIO()
        torch.save(module.state_dict(), memory)
        memory.seek(0)
        restored = AdaptiveRobustLoss()
        restored.load_state_dict(torch.load(memory, weights_only=True))
        pred = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 7.0
        torch.testing.assert_close(module(pred, torch.zeros_like(pred)), restored(pred, torch.zeros_like(pred)), rtol=0, atol=0)
        self.assertEqual(module.diagnostics(), restored.diagnostics())

    def test_shape_contract(self):
        module = AdaptiveRobustLoss()
        for pred, target in ((torch.zeros(0, 4), torch.zeros(0, 4)),
                             (torch.zeros(2, 3), torch.zeros(2, 3)),
                             (torch.zeros(2, 4), torch.zeros(3, 4))):
            with self.assertRaises(ValueError):
                module(pred, target)
        with self.assertRaises(TypeError):
            module(torch.ones(2, 4, dtype=torch.int64), torch.zeros(2, 4, dtype=torch.int64))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-table", action="store_true")
    args, rest = parser.parse_known_args()
    if args.build_table:
        build_table()
    else:
        unittest.main(argv=[__file__] + rest)
