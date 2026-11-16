"""Numerical properties for conditional densities; CPU torch is sufficient."""
from __future__ import annotations

import math
import unittest

import torch

from density_heads import ConditionalAffineCoupling, ConditionalGaussian, ConditionalRealNVP, make_head


class DensityProperties(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(712)
        self.condition = torch.randn(7, 9)
        self.target = torch.randn(7, 4)

    def nontrivial(self, head):
        # Zero-last-layer initialization is intentionally condition independent;
        # perturb it when checking non-identity transforms/conditional response.
        with torch.no_grad():
            networks = [layer.net for layer in head.layers] if isinstance(head, ConditionalRealNVP) else [head.net]
            for net in networks:
                net[-1].weight.normal_(mean=0, std=.04)
                net[-1].bias.add_(torch.randn_like(net[-1].bias) * .04)
        return head

    def test_initial_joint_standard_normal_density(self):
        expected = -.5 * (4 * math.log(2 * math.pi) + self.target.square().sum(dim=-1))
        for kind in ['gaussian', 'flow']:
            with self.subTest(kind=kind):
                head = make_head(kind, condition_dim=9, hidden_dim=24)
                torch.testing.assert_close(head.log_prob(self.target, self.condition), expected, atol=2e-6, rtol=1e-6)
                self.assertEqual(head.log_prob(self.target, self.condition).shape, (7,))

    def test_gaussian_cholesky_matches_torch_multivariate_normal(self):
        head = self.nontrivial(ConditionalGaussian(condition_dim=9, hidden_dim=24))
        location, scale_tril = head.distribution_parameters(self.condition)
        expected = torch.distributions.MultivariateNormal(location, scale_tril=scale_tril).log_prob(self.target)
        torch.testing.assert_close(head.log_prob(self.target, self.condition), expected, atol=3e-6, rtol=1e-6)
        self.assertTrue((scale_tril.diagonal(dim1=-2, dim2=-1) > 0).all())
        self.assertTrue(torch.equal(scale_tril.triu(diagonal=1), torch.zeros_like(scale_tril)))
        self.assertGreater(location.abs().sum().item(), 0)

    def test_coupling_inverse_and_jacobian_for_both_masks(self):
        for mask in [(1, 0, 1, 0), (0, 1, 0, 1)]:
            with self.subTest(mask=mask):
                layer = ConditionalAffineCoupling(9, hidden_dim=24, mask=mask)
                with torch.no_grad():
                    layer.net[-1].weight.normal_(0, .08)
                    layer.net[-1].bias.normal_(0, .1)
                transformed, logdet = layer(self.target, self.condition)
                recovered, inverse_logdet = layer.inverse(transformed, self.condition)
                torch.testing.assert_close(recovered, self.target, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(logdet + inverse_logdet, torch.zeros(7), atol=2e-6, rtol=0)
                kept = torch.tensor(mask, dtype=torch.bool)
                torch.testing.assert_close(transformed[:, kept], self.target[:, kept], atol=0, rtol=0)
                x, c = self.target[0].detach().requires_grad_(), self.condition[:1]
                jacobian = torch.autograd.functional.jacobian(lambda v: layer(v[None], c)[0][0], x)
                sign, numerical = torch.linalg.slogdet(jacobian)
                self.assertGreater(sign.item(), 0)
                torch.testing.assert_close(numerical, logdet[0], atol=3e-6, rtol=2e-6)

    def test_full_flow_inverse_density_and_logdet(self):
        head = self.nontrivial(ConditionalRealNVP(9, 24, num_layers=6))
        base, logdet = head.to_base(self.target, self.condition)
        reconstructed, inverse_logdet = head.from_base(base, self.condition)
        torch.testing.assert_close(reconstructed, self.target, atol=5e-6, rtol=3e-6)
        torch.testing.assert_close(logdet + inverse_logdet, torch.zeros(7), atol=3e-6, rtol=0)
        expected = -.5 * (4 * math.log(2 * math.pi) + base.square().sum(dim=-1)) + logdet
        torch.testing.assert_close(head.log_prob(self.target, self.condition), expected)
        x, c = self.target[0].detach().requires_grad_(), self.condition[:1]
        jacobian = torch.autograd.functional.jacobian(lambda v: head.to_base(v[None], c)[0][0], x)
        torch.testing.assert_close(torch.linalg.slogdet(jacobian)[1], logdet[0], atol=5e-6, rtol=3e-6)

    def test_sampling_shapes_reproducibility_and_reparameterized_gradients(self):
        for kind in ['gaussian', 'flow']:
            with self.subTest(kind=kind):
                head = make_head(kind, condition_dim=9, hidden_dim=24)
                first = head.sample(self.condition, 3, torch.Generator().manual_seed(12))
                second = head.sample(self.condition, 3, torch.Generator().manual_seed(12))
                self.assertEqual(first.shape, (7, 3, 4))
                self.assertEqual(first.dtype, torch.float32)
                torch.testing.assert_close(first, second, atol=0, rtol=0)
                first.square().mean().backward()
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_nll_updates_parameters_and_responds_to_condition(self):
        c_other = self.condition.clone()
        c_other[:, 0] += 3
        for kind in ['gaussian', 'flow']:
            with self.subTest(kind=kind):
                head = self.nontrivial(make_head(kind, condition_dim=9, hidden_dim=24))
                self.assertFalse(torch.allclose(head.log_prob(self.target, self.condition), head.log_prob(self.target, c_other)))
                before = [p.detach().clone() for p in head.parameters()]
                optimizer = torch.optim.Adam(head.parameters(), lr=.001)
                optimizer.zero_grad()
                loss = -head.log_prob(self.target, self.condition).mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
                optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, head.parameters())))

    def test_optional_anchor_uses_same_u_coordinates_only(self):
        anchored_target = self.target + self.condition[:, -4:]
        expected = -.5 * (4 * math.log(2 * math.pi) + self.target.square().sum(dim=-1))
        for kind in ['gaussian', 'flow']:
            with self.subTest(kind=kind):
                head = make_head(kind, condition_dim=9, hidden_dim=24, anchor_last=True)
                torch.testing.assert_close(head.log_prob(anchored_target, self.condition), expected, atol=2e-6, rtol=1e-6)
                unanchored = make_head(kind, condition_dim=9, hidden_dim=24, anchor_last=False)
                first = head.sample(self.condition, 2, torch.Generator().manual_seed(8))
                second = unanchored.sample(self.condition, 2, torch.Generator().manual_seed(8))
                torch.testing.assert_close(first, second + self.condition[:, None, -4:], atol=1e-6, rtol=1e-6)

    def test_autocast_keeps_joint_density_and_samples_float32(self):
        for kind in ['gaussian', 'flow']:
            with self.subTest(kind=kind):
                head = make_head(kind, condition_dim=9, hidden_dim=24)
                c, y = self.condition.bfloat16(), self.target.bfloat16()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    log_prob = head.log_prob(y, c)
                    samples = head.sample(c, 2)
                self.assertEqual(log_prob.dtype, torch.float32)
                self.assertEqual(samples.dtype, torch.float32)
                torch.testing.assert_close(log_prob, head.log_prob(y.float(), c.float()))

    def test_sampling_correlation_orientation(self):
        head = ConditionalGaussian(9, hidden_dim=8)
        # Packed Cholesky entries ordered (00,10,11,20,21,22,30,31,32,33).
        with torch.no_grad():
            head.net[-1].bias[4 + 1] = .6
        c = torch.zeros(1, 9)
        draws = head.sample(c, 80000, torch.Generator().manual_seed(71)).detach()[0]
        _, scale_tril = head.distribution_parameters(c)
        covariance = scale_tril[0] @ scale_tril[0].T
        torch.testing.assert_close(torch.cov(draws.T), covariance, atol=.025, rtol=.025)

    def test_shape_and_configuration_validation(self):
        for kind in ['gaussian', 'flow']:
            head = make_head(kind, condition_dim=9, hidden=24)
            with self.assertRaises(ValueError): head.log_prob(self.target[:, :3], self.condition)
            with self.assertRaises(ValueError): head.log_prob(self.target, self.condition[:, :8])
            with self.assertRaises(ValueError): head.sample(self.condition, 0)
            with self.assertRaises(TypeError): head.log_prob(self.target.long(), self.condition)
        with self.assertRaises(ValueError): make_head('unknown')
        with self.assertRaises(ValueError): make_head('gaussian', hidden=16, hidden_dim=32)
        with self.assertRaises(ValueError): ConditionalAffineCoupling(9, mask=(1, 1, 1, 1))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
