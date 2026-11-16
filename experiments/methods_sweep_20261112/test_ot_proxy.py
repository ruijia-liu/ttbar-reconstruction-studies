import copy
import io
import unittest
import numpy as np
import torch
from ot_proxy import (make_directions,assign_groups,fit_groups,conditional_ot_loss,
                      build_reference,ProxyConstraint)


def toy_reference():
    target=torch.tensor([[-.2]*4,[.2]*4,[.4]*4,[.6]*4,[1.2]*4,[1.4]*4,[1.6]*4,[2.]*4,[2.4]*4])
    pred=target+torch.linspace(-.3,.3,len(target))[:,None]
    edges=[[0.,1.,2.]]*4
    return pred,target,build_reference(pred,target,edges,[.1]*4,chunk_size=3)


class OTTests(unittest.TestCase):
    def test_fixed_directions_do_not_consume_global_rng(self):
        before=torch.random.get_rng_state().clone()
        d=make_directions()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        torch.testing.assert_close(d.norm(dim=1),torch.ones(32))
        torch.testing.assert_close(d,make_directions())

    def test_translation_has_expected_squared_projection_cost(self):
        t=torch.arange(24,dtype=torch.float32).reshape(6,4)
        shift=torch.tensor([1.,2.,3.,4.]);p=(t+shift).requires_grad_()
        value=conditional_ot_loss(p,t,torch.zeros(6,dtype=torch.long),[1.],torch.eye(4))
        torch.testing.assert_close(value,shift.square().mean())
        value.backward()
        torch.testing.assert_close(p.grad,shift.expand_as(p)*2/p.numel())

    def test_group_conditioning_detects_cross_group_exchange(self):
        t=torch.tensor([[0.]*4,[1.]*4,[10.]*4,[11.]*4]);p=t[[2,3,0,1]]
        groups=torch.tensor([0,0,1,1])
        global_loss=conditional_ot_loss(p,t,torch.zeros(4,dtype=torch.long),[1.],torch.eye(4))
        grouped=conditional_ot_loss(p,t,groups,[.5,.5],torch.eye(4))
        self.assertEqual(float(global_loss),0.)
        self.assertEqual(float(grouped),100.)
        self.assertGreater(float((p-t).abs().mean()),0.)

    def test_within_group_permutation_still_requires_paired_loss(self):
        t=torch.tensor([[0.]*4,[1.]*4,[10.]*4,[11.]*4]);p=t[[1,0,3,2]]
        v=conditional_ot_loss(p,t,torch.tensor([0,0,1,1]),[.5,.5],torch.eye(4))
        self.assertEqual(float(v),0.)
        self.assertGreater(float((p-t).square().mean()),0.)

    def test_missing_and_singleton_groups_are_not_renormalized(self):
        t=torch.zeros(3,4);p=torch.ones(3,4,requires_grad=True)
        v=conditional_ot_loss(p,t,torch.tensor([0,0,1]),[.2,.3,.5],torch.eye(4))
        self.assertAlmostEqual(float(v.detach()),.2,places=6)
        v.backward();self.assertEqual(float(p.grad[2].abs().sum()),0.)

    def test_training_group_edges_reused_and_last_edge_is_upper_group(self):
        x=np.arange(100,dtype=np.float32)
        stats=fit_groups(x)
        edges=torch.tensor(stats['internal_edges'])
        g=assign_groups(edges,edges)
        self.assertEqual(g.tolist(),[1,2])
        self.assertEqual(stats['counts'],[60,30,10])
        original=copy.deepcopy(stats)
        assign_groups(torch.tensor([-100.,1000.]),edges)
        self.assertEqual(stats,original)


class ProxyTests(unittest.TestCase):
    def test_full_reference_matches_histogram_and_actual_tail_rates(self):
        p,t,ref=toy_reference()
        for j in range(4):
            counts=np.histogram2d(t[:,j],p[:,j],bins=([0,1,2],[0,1,2]))[0]
            denom=np.histogram(t[:,j],bins=[0,1,2])[0]
            self.assertAlmostEqual(ref['baseline_trace'][j],np.diag(counts/denom[:,None]).mean())
            self.assertEqual(ref['truth_bin_counts'][j],denom.tolist())
            residual=(p[:,j]-t[:,j]).numpy()
            for side in range(2):
                q=ref['residual_quantiles'][side][j]
                expected=(residual<q).mean() if side==0 else (residual>q).mean()
                self.assertEqual(ref['baseline_tail_rates'][side][j],expected)
        loss,hard=ProxyConstraint(ref)(p,t)
        self.assertEqual(float(loss),0.)
        torch.testing.assert_close(hard,torch.zeros(12),atol=1e-7,rtol=0)

    def test_trace_is_row_balanced_and_counts_predicted_overflow_as_wrong(self):
        t=torch.tensor([[.2]*4,[.2]*4,[1.2]*4])
        p=torch.tensor([[.2]*4,[3.]*4,[1.2]*4])
        ref=build_reference(t,t,[[0,1,2]]*4,[.1]*4)
        _,hard=ProxyConstraint(ref)(p,t)
        torch.testing.assert_close(hard[:4],torch.full((4,),.25))
        self.assertEqual(ref['baseline_trace'],[1.]*4)

    def test_last_edge_included_and_truth_outside_not_renormalized(self):
        t=torch.tensor([[.2]*4,[2.]*4,[3.]*4])
        ref=build_reference(t,t,[[0,1,2]]*4,[.1]*4)
        self.assertEqual(ref['truth_bin_counts'][0],[1,1])
        self.assertAlmostEqual(sum(ref['bin_probabilities'][0]),2/3)
        _,hard=ProxyConstraint(ref)(t,t)
        torch.testing.assert_close(hard[:4],torch.zeros(4),atol=1e-7,rtol=0)

    def test_dual_uses_hard_violations_and_projection(self):
        _,_,ref=toy_reference();module=ProxyConstraint(ref,dual_lr=.5,dual_cap=.3)
        module.update(torch.ones(12))
        torch.testing.assert_close(module.multipliers,torch.full((12,),.3))
        module.update(-torch.ones(12))
        self.assertEqual(float(module.multipliers.sum()),0.)
        self.assertEqual(int(module.updates),2)

    def test_proxy_gradient_corrects_wrong_bin_with_hard_metrics_detached(self):
        t=torch.tensor([[.5]*4,[1.5]*4])
        ref=build_reference(t,t,[[0,1,2]]*4,[.2]*4)
        module=ProxyConstraint(ref);module.multipliers[:4]=1.
        p=torch.tensor([[1.1]*4,[1.5]*4],requires_grad=True)
        loss,hard=module(p,t)
        self.assertFalse(hard.requires_grad)
        loss.backward()
        self.assertTrue(torch.all(p.grad[0]>0))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_lower_and_upper_are_separate_constraints(self):
        p,t,ref=toy_reference();module=ProxyConstraint(ref)
        _,hi=module(p+1.,t);_,lo=module(p-1.,t)
        self.assertTrue((hi[8:]>0).all());self.assertTrue((hi[4:8]<0).all())
        self.assertTrue((lo[4:8]>0).all());self.assertTrue((lo[8:]<0).all())

    def test_bfloat16_and_checkpoint_roundtrip(self):
        p,t,ref=toy_reference();module=ProxyConstraint(ref);module.update(torch.ones(12))
        p=p.bfloat16().requires_grad_()
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,hard=module(p,t.bfloat16())
        self.assertEqual(loss.dtype,torch.float32);self.assertEqual(hard.dtype,torch.float32)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
        other=ProxyConstraint(ref);other.load_state_dict(module.state_dict())
        torch.testing.assert_close(other(p,t.bfloat16())[0],module(p,t.bfloat16())[0])

    def test_invalid_empty_truth_bins_fail(self):
        x=torch.zeros(6,4)
        with self.assertRaises(ValueError):build_reference(x,x,[[0,1,2]]*4,[.1]*4)


if __name__=='__main__':unittest.main()
