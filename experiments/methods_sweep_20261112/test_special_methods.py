"""Small CPU checks for the mathematical and autograd contracts of two arms."""
import unittest

import torch
from torch import nn

from flowalign import VelocityField, field_training_loss, path_consistency_loss
from ranksim import TrueRanker, rank_normalised, rank_similarity_loss


def rng(seed=741):
    return torch.Generator(device="cpu").manual_seed(seed)


class EndpointOracle(nn.Module):
    def forward(self, point, time, condition):
        return (condition[:, :4] - point) / (1.0 - time)


class LinearOracle(nn.Module):
    def forward(self, point, time, condition):
        return 0.5 * point + condition[:, :4]


class RankSimTests(unittest.TestCase):
    def test_correct_neighbor_order_is_best(self):
        target = torch.tensor([0.0, 0.11, 0.36, 0.91])
        features = torch.stack((target.cos(), target.sin()), dim=1)
        good = rank_similarity_loss(features, target)
        bad = rank_similarity_loss(features[[0, 3, 1, 2]], target)
        self.assertEqual(good.item(), 0.0)
        self.assertGreater(bad.item(), good.item())

    def test_official_black_box_backward(self):
        x = torch.tensor([[0.2, 0.3, 0.1]], requires_grad=True)
        incoming = torch.tensor([[0.0, -0.3, 0.2]])
        ranks = TrueRanker.apply(x, 2.0)
        (ranks * incoming).sum().backward()
        expected = (rank_normalised(x.detach() + 2.0 * incoming)
                    - rank_normalised(x.detach())) / (2.0 + 1e-8)
        torch.testing.assert_close(x.grad, expected)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_trainable_features_receive_finite_gradient(self):
        features = torch.randn((64, 12), generator=rng(), requires_grad=True)
        target = torch.randn((64,), generator=rng(23))
        loss = rank_similarity_loss(features, target)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_duplicates_and_singleton_do_not_invent_rank_signal(self):
        for count in (0, 1, 5):
            features = torch.randn((count, 3), generator=rng(), requires_grad=True)
            value = rank_similarity_loss(features, torch.ones(count))
            value.backward()
            self.assertEqual(value.item(), 0)
            self.assertEqual(features.grad.abs().sum().item(), 0)

    def test_loss_reduction_matches_anchor_sum(self):
        features = torch.randn((9, 5), generator=rng(), requires_grad=True)
        target = torch.arange(9).float() ** 1.4
        normalized = torch.nn.functional.normalize(features, dim=1)
        similarity = normalized @ normalized.T
        expected = torch.tensor(0.0)
        for row in range(9):
            predicted = rank_normalised(similarity[row:row + 1])
            truth = rank_normalised(-(target[row] - target).abs()[None, :])
            expected += (predicted - truth).square().mean()
        torch.testing.assert_close(rank_similarity_loss(features, target), expected)


class FlowAlignTests(unittest.TestCase):
    def setUp(self):
        self.base = torch.randn((32, 4), generator=rng())
        self.target = torch.randn((32, 4), generator=rng(12))
        self.condition = torch.cat((self.target, torch.zeros(32, 128)), dim=1)

    def test_oracle_correct_path_zero_wrong_endpoint_positive(self):
        oracle = EndpointOracle()
        train = field_training_loss(oracle, self.condition, self.base,
                                    self.target, rng())
        correct = path_consistency_loss(oracle, self.condition, self.base,
                                        self.target, rng())
        wrong = path_consistency_loss(oracle, self.condition, self.base,
                                      self.target + 0.2, rng())
        self.assertLess(train.item(), 1e-10)
        self.assertLess(correct.item(), 1e-10)
        self.assertGreater(wrong.item(), 0.01)

    def test_frozen_field_keeps_candidate_gradient(self):
        with torch.random.fork_rng():
            torch.manual_seed(83)
            field = VelocityField().requires_grad_(False).eval()
        candidate = self.base.clone().requires_grad_()
        loss = path_consistency_loss(field, self.condition, self.base,
                                     candidate, rng())
        loss.backward()
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(candidate.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in field.parameters()))

    def test_gradient_includes_field_input_derivative(self):
        candidate = self.base.clone().requires_grad_()
        loss = path_consistency_loss(LinearOracle(), self.condition,
                                     self.base, candidate, rng())
        actual, = torch.autograd.grad(loss, candidate)
        generator = rng()
        noise = torch.randn(self.base.shape, generator=generator)
        time = torch.rand((len(self.base), 1), generator=generator)
        start = self.base + 0.05 * noise
        point = (1.0 - time) * start + time * candidate.detach()
        residual = 0.5 * point + self.target - (candidate.detach() - start)
        expected = 2 * residual * (0.5 * time - 1.0) / candidate.numel()
        torch.testing.assert_close(actual, expected)

    def test_private_rng_and_reproducibility(self):
        global_before = torch.random.get_rng_state().clone()
        generator = rng()
        saved = generator.get_state().clone()
        first = path_consistency_loss(LinearOracle(), self.condition,
                                      self.base, self.target, generator)
        generator.set_state(saved)
        repeat = path_consistency_loss(LinearOracle(), self.condition,
                                       self.base, self.target, generator)
        torch.testing.assert_close(first, repeat, rtol=0, atol=0)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))

    def test_field_training_updates_parameters_but_not_source(self):
        with torch.random.fork_rng():
            torch.manual_seed(83)
            field = VelocityField()
        condition = self.condition.clone().requires_grad_()
        base = self.base.clone().requires_grad_()
        target = self.target.clone().requires_grad_()
        loss = field_training_loss(field, condition, base, target, rng())
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in field.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in field.parameters()), 0)
        self.assertIsNone(condition.grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(target.grad)

    def test_unfrozen_stage2_and_missing_rng_rejected(self):
        with torch.random.fork_rng():
            field = VelocityField()
        with self.assertRaises(ValueError):
            path_consistency_loss(field, self.condition, self.base, self.target, rng())
        with self.assertRaises(ValueError):
            field_training_loss(field, self.condition, self.base, self.target, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
