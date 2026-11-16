"""CPU scientific contracts; full Lightning/CUDA lifecycle is tested in Slurm."""
import copy
import io
import math
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from models import INPUTS,build_model,objective_components,summarize_details


class RecoBackbone(nn.Module):
    def forward(self,batch):
        if set(batch)!=set(INPUTS):raise AssertionError('Non-reco key reached backbone')
        return batch['global']


class Base(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=RecoBackbone();self.head=nn.Linear(8,4)


def construct(mode,base=None):
    base=Base() if base is None else base
    mean=[0,0,0,math.log(172.5)]*2
    std=[100,100,200,.05]*2
    return build_model(base,dict(mode=mode,d_model=8),.2,mean,std,[500,0,0,0],[150,1,.5,.5])


def batch(n=12):
    generator=torch.Generator().manual_seed(330)
    return {'global':torch.randn(n,8,generator=generator),
            'jet':torch.zeros(n,2,9),'jet_mask':torch.zeros(n,2,dtype=torch.bool),
            'lepton':torch.zeros(n,2,10),'lepton_mask':torch.zeros(n,2,dtype=torch.bool),
            'target':torch.randn(n,4,generator=generator),
            'tail_label':(torch.arange(n)%4==0).float(),
            'aux_target':torch.randn(n,8,generator=generator)}


class ModelTests(unittest.TestCase):
    def test_extra_truth_fields_cannot_change_inference(self):
        original=batch()
        altered={k:v.clone() for k,v in original.items()}
        altered['target'].fill_(999);altered['tail_label'].fill_(1);altered['aux_target'].fill_(-999)
        altered['true_mass']=torch.zeros(len(original['target']))
        for mode in ['control','moe_fixed','moe_learned','physics']:
            model=construct(mode).eval()
            torch.testing.assert_close(model(original),model(altered),rtol=0,atol=0)

    def test_initialization_preserves_rng_and_original_parameters(self):
        for mode in ['control','moe_fixed','moe_learned','physics']:
            base=Base();before=torch.random.get_rng_state().clone()
            weights={k:v.clone() for k,v in base.state_dict().items()}
            model=construct(mode,base)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            for k,v in weights.items():torch.testing.assert_close(v,base.state_dict()[k],rtol=0,atol=0)
            self.assertIs(model.core,base.head)

    def test_fixed_and_learned_arms_start_identically(self):
        base=Base();fixed=construct('moe_fixed',copy.deepcopy(base)).eval()
        learned=construct('moe_learned',copy.deepcopy(base)).eval()
        b=batch();f=fixed.forward_details(b);l=learned.forward_details(b)
        for k in f:torch.testing.assert_close(f[k],l[k],rtol=1e-6,atol=1e-7)
        for p,q in zip(fixed.parameters(),learned.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        for name,value in objective_components(fixed,f,b).items():
            torch.testing.assert_close(value,objective_components(learned,l,b)[name],rtol=1e-6,atol=1e-7)

    def test_learned_route_changes_prediction_and_fixed_route_does_not(self):
        models=[construct(mode).eval() for mode in ['moe_fixed','moe_learned']]
        b=batch();b['global'].zero_();b['global'][:,0]=torch.linspace(-2,2,len(b['global']))
        for model in models:
            with torch.no_grad():
                model.core.weight.zero_();model.core.bias.zero_()
                model.tail.weight.zero_();model.tail.bias.fill_(2.)
                for p in model.gate.parameters():p.zero_()
                model.gate[0].weight[0,0]=1.;model.gate[-1].weight[0,0]=2.
        fixed,learned=[m.forward_details(b) for m in models]
        torch.testing.assert_close(fixed['prediction'],torch.full_like(fixed['prediction'],.4))
        self.assertGreater(float((learned['prediction'][-1,0]-learned['prediction'][0,0]).detach()),1.)
        torch.testing.assert_close(learned['prediction'],2*learned['gate_prob'][:,None].expand(-1,4))

    def test_conditional_specialization_moves_experts_differently(self):
        model=construct('moe_learned');b=batch(20);b['global'].fill_(1)
        b['tail_label']=(torch.arange(20)%5==0).float()
        b['target']=torch.where(b['tail_label'][:,None].bool(),torch.ones(20,4)*3,-torch.ones(20,4))
        with torch.no_grad():
            for head in [model.core,model.tail]:head.weight.zero_();head.bias.zero_()
        details=model.forward_details(b);components=objective_components(model,details,b)
        components['specialization'].backward()
        self.assertTrue((model.core.bias.grad>0).all())
        self.assertTrue((model.tail.bias.grad<0).all())
        self.assertTrue(torch.isfinite(model.core.weight.grad).all())
        self.assertTrue(torch.isfinite(model.tail.weight.grad).all())

    def test_loss_weights_match_independent_formula(self):
        b=batch();model=construct('moe_learned');d=model.forward_details(b)
        c=objective_components(model,d,b);n=len(b['target'])
        spec=0.
        for i in range(n):
            y=b['target'][i]
            value=2*F.huber_loss(d['tail'][i] if b['tail_label'][i] else d['core'][i],y)
            spec=spec+value/(.2 if b['tail_label'][i] else .8)/(2*n)
        torch.testing.assert_close(c['specialization'],spec)
        expected=c['point']+.1*c['emd']+.2*spec+.05*c['gate_bce']
        torch.testing.assert_close(c['loss'],expected)

    def test_gate_classifier_trains_even_in_fixed_arm(self):
        for mode in ['moe_fixed','moe_learned']:
            model=construct(mode);b=batch()
            objective_components(model,model.forward_details(b),b)['loss'].backward()
            self.assertIsNotNone(model.gate[-1].bias.grad)
            self.assertGreater(float(model.gate[-1].bias.grad.abs().sum()),0.)

    def test_physics_supervision_and_consistency_have_gradients(self):
        b=batch();model=construct('physics');d=model.forward_details(b)
        c=objective_components(model,d,b)
        torch.testing.assert_close(c['loss'],c['point']+.1*c['emd']+.2*c['auxiliary']+.05*c['consistency'])
        c['consistency'].backward(retain_graph=True)
        self.assertGreater(float(model.aux_head[-1].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.core.weight.grad.abs().sum()),0.)
        model.zero_grad();c['auxiliary'].backward()
        self.assertGreater(float(model.aux_head[-1].weight.grad.abs().sum()),0.)

    def test_checkpoint_roundtrip_preserves_predictions(self):
        for mode in ['control','moe_fixed','moe_learned','physics']:
            model=construct(mode).eval();b=batch();expected=model(b)
            buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
            restored=construct(mode).eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
            torch.testing.assert_close(expected,restored(b),rtol=0,atol=0)

    def test_bfloat16_forward_and_loss_gradients_are_finite(self):
        for mode in ['control','moe_fixed','moe_learned','physics']:
            model=construct(mode);b=batch()
            with torch.autocast('cpu',dtype=torch.bfloat16):
                details=model.forward_details(b);c=objective_components(model,details,b)
            self.assertEqual(c['loss'].dtype,torch.float32)
            c['loss'].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_full_split_gate_diagnostics(self):
        model=construct('moe_learned').eval();b=batch();d=model.forward_details(b)
        s=summarize_details(d,b['tail_label'])
        self.assertEqual(s['truth_region_gate']['tail']['n'],int(b['tail_label'].sum()))
        self.assertAlmostEqual(s['gate_mean'],.2,places=6)
        self.assertEqual(s['expert_mean_abs_difference_per_observable'],[0.,0.,0.,0.])
        self.assertEqual(len(s['gate_quantiles']),7)


if __name__=='__main__':unittest.main()
