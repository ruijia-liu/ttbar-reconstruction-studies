"""CPU numerical checks for complete-bin, weighted differentiable migration loss."""
import copy
import json
import unittest

import numpy as np
import torch

from migration_loss import (KEYS, PHYSICAL_BINS, MigrationLoss, _bin_indices,
                            _log_probabilities, compute_bin_counts)


def module(counts=None, mean=None, std=None):
    if counts is None:
        # A shared total of 840 permits equal counts for 6,7,8 categories.
        counts = [[840//(len(PHYSICAL_BINS[key])+1)]*(len(PHYSICAL_BINS[key])+1) for key in KEYS]
    return MigrationLoss([0.]*4 if mean is None else mean,[1.]*4 if std is None else std,counts)


class MigrationLossTests(unittest.TestCase):
    def test_counting_all_boundaries_matches_histogram_with_overflow(self):
        # Float32 arrays and float32 edges match the training-label convention.
        rows = []
        for j,key in enumerate(KEYS):
            edges = np.asarray(PHYSICAL_BINS[key],dtype=np.float32)
            for value in np.concatenate(([edges[0]-10],edges,[edges[-1]+10])):
                row = np.array([450.,.2,.2,.2],dtype=np.float32)
                row[j] = value;rows.append(row)
        target = np.stack(rows)
        actual = compute_bin_counts(target,[0.]*4,[1.]*4,chunk_size=3)
        for j,key in enumerate(KEYS):
            edges = np.asarray(PHYSICAL_BINS[key],dtype=np.float32)
            finite = np.histogram(target[:,j],bins=edges)[0]
            expected = [int((target[:,j]<edges[0]).sum()),*finite.tolist(),int((target[:,j]>edges[-1]).sum())]
            self.assertEqual(actual[j],expected)
            self.assertEqual(sum(actual[j]),len(target))

    def test_exact_final_edge_is_last_interior_bin(self):
        loss = module()
        for j in range(4):
            edges = getattr(loss,'edges_'+str(j))
            labels = _bin_indices(edges,edges)
            expected = torch.arange(1,len(edges)+1)
            expected[-1] -= 1
            torch.testing.assert_close(labels,expected)

    def test_standardized_last_edge_matches_float32_preprocessing(self):
        mean = torch.tensor([489.3283,.017251,-.031239,.041923])
        effective_std = torch.tensor([170.4231,.498716,.578321,.563578])+1e-8
        loss = module(mean=mean,std=effective_std)
        for j,key in enumerate(KEYS):
            encoded = (torch.tensor(PHYSICAL_BINS[key],dtype=torch.float32)-mean[j])/effective_std[j]
            edges = getattr(loss,'edges_'+str(j))
            torch.testing.assert_close(encoded,edges,atol=0,rtol=0)
            self.assertEqual(int(_bin_indices(encoded[-1:],edges)[0]),len(edges)-1)

    def test_temperature_uses_physical_median_and_std(self):
        loss = module(mean=[500.,.1,-.2,.3],std=[100.,2.,.5,4.])
        torch.testing.assert_close(loss.physical_temperature,torch.tensor([8.,.05,.05,.05],dtype=torch.float64))
        torch.testing.assert_close(loss.temperature,torch.tensor([.08,.025,.1,.0125]))

    def test_normalized_full_interval_probabilities_include_both_tails(self):
        loss = module()
        for j in range(4):
            edges = getattr(loss,'edges_'+str(j))
            points = torch.tensor([float(edges[0])-10,float(edges[0]),float(edges.mean()),float(edges[-1]),float(edges[-1])+10])
            logp = _log_probabilities(points,edges,loss.temperature[j])
            torch.testing.assert_close(logp.exp().sum(dim=1),torch.ones(len(points)),atol=3e-7,rtol=3e-7)
            self.assertEqual(logp.shape[1],len(edges)+1)
            self.assertGreater(logp[0,0].exp().item(),.5)
            self.assertGreater(logp[-1,-1].exp().item(),.5)

    def test_weighted_scalar_matches_independent_cdf_difference(self):
        counts = [[300]+[10]*(len(PHYSICAL_BINS[key])-1)+[420-300-10*(len(PHYSICAL_BINS[key])-1)] for key in KEYS]
        loss = module(counts)
        pred = torch.tensor([[375.,-.45,.15,-.15],[755.,1.1,-.45,.55]])
        target = torch.tensor([[400.,-.2,.3,-.3],[800.,1.2,-.3,.7]])
        expected = 0.
        for j in range(4):
            edges = getattr(loss,'edges_'+str(j)).double()
            cdf = torch.sigmoid((edges[None,:]-pred[:,j,None].double())/loss.temperature[j].double())
            prob = torch.cat([cdf[:,:1],cdf[:,1:]-cdf[:,:-1],1-cdf[:,-1:]],dim=1)
            labels = _bin_indices(target[:,j],edges.float())
            weights = getattr(loss,'weights_'+str(j)).double()[labels]
            expected += (-torch.log(prob[torch.arange(len(pred)),labels])*weights).mean()/4
        self.assertAlmostEqual(loss(pred,target).item(),expected.item(),places=5)

    def test_gradients_point_toward_true_interval_and_overflow(self):
        loss = module()
        target = torch.tensor([[450.,.25,.25,.25],[450.,.25,.25,.25]])
        pred = torch.tensor([[350.,-1.5,-.9,-.9],[900.,1.5,.9,.9]],requires_grad=True)
        loss(pred,target).backward()
        self.assertTrue((pred.grad[0]<0).all())
        self.assertTrue((pred.grad[1]>0).all())
        target = torch.tensor([[250.,-3.,-2.,-2.],[2500.,3.,2.,2.]])
        pred = torch.tensor([[350.,0.,0.,0.],[1900.,0.,0.,0.]],requires_grad=True)
        loss(pred,target).backward()
        self.assertTrue((pred.grad[0]>0).all())
        self.assertTrue((pred.grad[1]<0).all())

    def test_extreme_wrong_predictions_have_finite_nonzero_gradients(self):
        loss = module()
        pred = torch.tensor([[-1e6]*4,[1e6]*4],requires_grad=True)
        target = torch.tensor([[450.,.25,.25,.25]]*2)
        value = loss(pred,target)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((pred.grad.abs()>0).all())

    def test_bfloat16_autocast_keeps_loss_float32_and_gradients(self):
        loss = module()
        pred = torch.tensor([[400.,-.1,.1,-.1],[600.,.2,-.2,.3]],dtype=torch.bfloat16,requires_grad=True)
        target = torch.tensor([[450.,.2,.2,.2],[650.,.4,-.4,.4]])
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            value = loss(pred,target)
        self.assertEqual(value.dtype,torch.float32)
        value.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((pred.grad.abs()>0).all())

    def test_zero_counts_capped_then_normalized_and_fixed(self):
        counts = [[100000,1]+[0]*(len(PHYSICAL_BINS[key])-1) for key in KEYS]
        loss = module(counts)
        before = copy.deepcopy(loss.state_dict())
        for j in range(4):
            c = getattr(loss,'counts_'+str(j)).double()
            w = getattr(loss,'weights_'+str(j)).double()
            self.assertTrue(torch.isfinite(w).all())
            self.assertTrue((w>0).all())
            self.assertLessEqual(float(getattr(loss,'capped_weights_'+str(j)).max()),5)
            self.assertAlmostEqual(float((c*w).sum()/c.sum()),1.,places=6)
            self.assertGreater(float(w.max()),5)
        pred = torch.tensor([[450.,.2,.2,.2]],requires_grad=True)
        target = pred.detach().clone().requires_grad_(True)
        loss(pred,target).backward()
        self.assertIsNone(target.grad)
        self.assertEqual(list(loss.parameters()),[])
        for name,value in before.items():
            torch.testing.assert_close(loss.state_dict()[name],value)
        json.dumps(loss.diagnostics(),allow_nan=False)

    def test_count_helper_standardized_inputs_and_chunking(self):
        mean,std = [500.,.1,-.2,.3],[100.,2.,.5,4.]
        target = torch.randn(103,4,generator=torch.Generator().manual_seed(19))
        first = compute_bin_counts(target,mean,std,chunk_size=7)
        second = compute_bin_counts(target,mean,std,chunk_size=1000)
        self.assertEqual(first,second)
        loss = module(first,mean,std)
        for j in range(4):
            expected = torch.bincount(_bin_indices(target[:,j],getattr(loss,'edges_'+str(j))),minlength=len(first[j])).tolist()
            self.assertEqual(first[j],expected)

    def test_validation_rejects_invalid_shapes_statistics_and_counts(self):
        loss = module()
        with self.assertRaises(ValueError):
            loss(torch.ones(2,3),torch.ones(2,3))
        with self.assertRaises(ValueError):
            loss(torch.full((2,4),float('nan')),torch.ones(2,4))
        with self.assertRaises(ValueError):
            module(std=[1.,0.,1.,1.])
        counts = [[60]*7,[60]*8,[60]*6,[60]*6]
        with self.assertRaises(ValueError):
            module(counts)
        with self.assertRaises(ValueError):
            compute_bin_counts(torch.empty(0,4),[0.]*4,[1.]*4)
        with self.assertRaises(ValueError):
            compute_bin_counts(torch.ones(2,4),[0.]*4,[1.]*4,chunk_size=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
