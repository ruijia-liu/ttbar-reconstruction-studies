"""Mathematical and row-alignment checks; CPU torch is sufficient."""
from __future__ import annotations

import pickle
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from gated_loss import WeightedDataset, gated_loss


class CachedDataset(Dataset):
    def __init__(self, n=9):
        self.rows = torch.arange(n)
        self.target_tensor = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
        self.max_jets = 6
        self.stats = {"training_only": True}
        self.batch_fetches = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"row_index": self.rows[index], "target": self.target_tensor[index]}

    def __getitems__(self, indices):
        self.batch_fetches += 1
        return {"row_index": self.rows[indices], "target": self.target_tensor[indices]}


class OrdinaryDataset(Dataset):
    def __init__(self, n=9):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"row_index": torch.tensor(index), "target": torch.full((4,), float(index))}


class ListBatchDataset(OrdinaryDataset):
    def __getitems__(self, indices):
        return [self[int(i)] for i in indices]


def identity(batch):
    return batch


class GatedLossProperties(unittest.TestCase):
    def setUp(self):
        self.pred = torch.tensor([[2., -.2, .8, -.9], [-1., 1., -.7, .3], [.1, -2., 1., .2]])
        self.target = torch.tensor([[.3, .1, -.2, -.1], [.1, .2, -.3, -.1], [.2, -.4, .1, -.2]])

    def expected(self, gate, delta=.5):
        elementwise = 2 * F.huber_loss(self.pred, self.target, delta=.5, reduction="none")
        hmass = 2 * F.huber_loss(self.pred[:, 0], self.target[:, 0], delta=delta, reduction="none")
        elementwise[:, 0] = (1-gate)*hmass + gate*(self.pred[:, 0]-self.target[:, 0]).square()
        return elementwise.mean()

    def test_zero_gate_recovers_previous_four_output_huber(self):
        expected = 2 * F.huber_loss(self.pred, self.target, delta=.5)
        torch.testing.assert_close(gated_loss(self.pred, self.target, 0.), expected)

    def test_one_gate_changes_only_mass_to_mse(self):
        torch.testing.assert_close(gated_loss(self.pred, self.target, 1.), self.expected(1.))
        pred = self.pred.clone().requires_grad_()
        gated_loss(pred, self.target, 1.).backward()
        residual = self.pred-self.target
        gradient = 2*residual.clamp(-.5, .5)/residual.numel()
        gradient[:, 0] = 2*residual[:, 0]/residual.numel()
        torch.testing.assert_close(pred.grad, gradient)

    def test_scalar_vector_column_and_mass_delta_formulas(self):
        for gate in [.5, torch.tensor([0., .3, 1.])]:
            with self.subTest(gate=gate):
                torch.testing.assert_close(gated_loss(self.pred, self.target, gate), self.expected(gate))
        gate = torch.tensor([0., .3, 1.])
        torch.testing.assert_close(gated_loss(self.pred, self.target, gate[:, None]), self.expected(gate))
        # Nondefault delta changes mass only; the other outputs stay Huber_0.5.
        torch.testing.assert_close(gated_loss(self.pred, self.target, gate, delta=1.), self.expected(gate, delta=1.))

    def test_gate_is_detached_and_prediction_gradients_are_finite(self):
        pred = self.pred.clone().requires_grad_()
        gate = torch.tensor([.1, .5, .9], requires_grad=True)
        original = gate.detach().clone()
        result = gated_loss(pred, self.target, gate)
        self.assertEqual(result.ndim, 0)
        self.assertEqual(result.dtype, torch.float32)
        result.backward()
        self.assertIsNone(gate.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(torch.equal(gate.detach(), original))

    def test_equal_inputs_zero_loss_and_gradient(self):
        pred = self.pred.clone().requires_grad_()
        result = gated_loss(pred, self.pred, torch.tensor([0., .5, 1.]))
        self.assertEqual(result.item(), 0.)
        result.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_autocast_still_uses_float32_loss(self):
        pred = self.pred.to(torch.bfloat16).requires_grad_()
        target = self.target.to(torch.bfloat16)
        gate = torch.tensor([0., .5, 1.], dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = gated_loss(pred, target, gate)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, gated_loss(pred.float(), target.float(), gate.float()))
        result.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_shapes_and_parameters(self):
        for pred, target in [(self.pred[:, :3], self.target[:, :3]), (self.pred, self.target[:2]),
                             (self.pred.flatten(), self.target.flatten()), (self.pred[:0], self.target[:0])]:
            with self.assertRaises(ValueError):
                gated_loss(pred, target, .5)
        with self.assertRaises(TypeError):
            gated_loss(self.pred.long(), self.target, .5)
        with self.assertRaises(TypeError):
            gated_loss(self.pred.tolist(), self.target, .5)
        with self.assertRaises(TypeError):
            gated_loss(self.pred, self.target, 0.5+0.1j)
        for gate in [torch.ones(2), torch.ones(3, 2), torch.ones(1, 3), -0.01, 1.01, float("nan"), float("inf")]:
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                gated_loss(self.pred, self.target, gate)
        for delta in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                gated_loss(self.pred, self.target, .5, delta=delta)


class DatasetProperties(unittest.TestCase):
    def setUp(self):
        self.base = CachedDataset()
        self.weights = torch.linspace(0, 1, len(self.base))

    def test_cache_proxy_and_metadata_do_not_modify_base(self):
        wrapper = WeightedDataset(self.base, self.weights)
        self.assertEqual(len(wrapper), len(self.base))
        self.assertEqual(wrapper.max_jets, self.base.max_jets)
        self.assertIs(wrapper.stats, self.base.stats)
        self.assertIs(wrapper.target_tensor, self.base.target_tensor)
        item = wrapper[3]
        self.assertNotIn("mass_gate", self.base[3])
        torch.testing.assert_close(item["target"], self.base[3]["target"])
        self.assertEqual(item["mass_gate"].dtype, torch.float32)
        self.assertFalse(item["mass_gate"].requires_grad)

    def test_source_and_returned_weights_cannot_change_fixed_values(self):
        source = self.weights.clone().requires_grad_()
        wrapper = WeightedDataset(self.base, source)
        with torch.no_grad():
            source.fill_(0.)
        torch.testing.assert_close(wrapper.weights, self.weights)
        wrapper.weights.fill_(0.)
        wrapper[3]["mass_gate"].fill_(0.)
        wrapper.__getitems__([3, 5])["mass_gate"].fill_(0.)
        torch.testing.assert_close(wrapper.weights, self.weights)
        self.assertFalse(wrapper.weights.requires_grad)

    def test_batched_fetch_preserves_shuffled_and_repeated_indices(self):
        wrapper = WeightedDataset(self.base, self.weights)
        index = [8, 0, 4, 2, 4]
        batch = wrapper.__getitems__(index)
        self.assertEqual(self.base.batch_fetches, 1)
        torch.testing.assert_close(batch["row_index"], torch.tensor(index))
        torch.testing.assert_close(batch["mass_gate"], self.weights[index])
        torch.testing.assert_close(batch["target"], self.base.target_tensor[index])
        batch = wrapper.__getitems__(torch.tensor(index, dtype=torch.uint8))
        torch.testing.assert_close(batch["row_index"], torch.tensor(index))
        torch.testing.assert_close(batch["mass_gate"], self.weights[index])

    def test_shuffled_dataloader_covers_all_rows_with_correct_weights(self):
        wrapper = WeightedDataset(self.base, self.weights)
        loader = DataLoader(wrapper, batch_size=4, shuffle=True, collate_fn=identity,
                            generator=torch.Generator().manual_seed(19))
        seen = []
        for batch in loader:
            index = batch["row_index"]
            seen += index.tolist()
            torch.testing.assert_close(batch["mass_gate"], self.weights[index])
        self.assertEqual(sorted(seen), list(range(len(wrapper))))
        self.assertNotEqual(seen, list(range(len(wrapper))))

    def test_ordinary_and_list_batch_datasets_work_with_default_collation(self):
        for base in [OrdinaryDataset(), ListBatchDataset()]:
            wrapper = WeightedDataset(base, self.weights)
            for batch in DataLoader(wrapper, batch_size=4, shuffle=True,
                                    generator=torch.Generator().manual_seed(11)):
                torch.testing.assert_close(batch["mass_gate"], self.weights[batch["row_index"]])

    def test_pickle_keeps_weights_and_cache_attributes(self):
        wrapper = pickle.loads(pickle.dumps(WeightedDataset(self.base, self.weights)))
        torch.testing.assert_close(wrapper.weights, self.weights)
        self.assertEqual(wrapper.max_jets, self.base.max_jets)

    def test_invalid_weights_and_batch_indices(self):
        for weights in [torch.zeros(8), torch.zeros(9, 1), torch.full((9,), -.1),
                        torch.full((9,), 1.1), torch.full((9,), float("nan"))]:
            with self.assertRaises(ValueError):
                WeightedDataset(self.base, weights)
        wrapper = WeightedDataset(self.base, self.weights)
        for indices in [[1., 2.], [[1, 2]], [True, False]]:
            with self.assertRaises(ValueError):
                wrapper.__getitems__(indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
