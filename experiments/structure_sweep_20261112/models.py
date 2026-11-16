"""Reco-only deterministic experts and physically supervised shared representation."""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

INPUTS = ('global', 'jet', 'jet_mask', 'lepton', 'lepton_mask')
MODES = ('control', 'moe_fixed', 'moe_learned', 'physics')


class StructureRegressor(nn.Module):
    def __init__(self, base_model, config, tail_prevalence, aux_mean, aux_std,
                 main_mean, main_effective_std):
        super().__init__()
        self.mode = config['mode']
        if self.mode not in MODES:
            raise ValueError('Unknown structural arm: '+self.mode)
        if not math.isfinite(float(tail_prevalence)) or not 0 < tail_prevalence < 1:
            raise ValueError('Need a nontrivial full-training tail prevalence')
        self.backbone = base_model.backbone
        self.core = base_model.head
        self.register_buffer('tail_prevalence',torch.tensor(float(tail_prevalence),dtype=torch.float32))
        for name,value,size in [('aux_mean',aux_mean,8),('aux_std',aux_std,8),
                                ('main_mean',main_mean,4),('main_effective_std',main_effective_std,4)]:
            tensor=torch.as_tensor(value,dtype=torch.float32).detach().clone()
            if tensor.shape!=(size,) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid statistics: '+name)
            if 'std' in name and not (tensor>0).all():raise ValueError('Nonpositive statistics: '+name)
            self.register_buffer(name,tensor)
        width=int(config['d_model'])
        # Constructors consume CPU RNG; restore it so the pre-existing backbone
        # and core dropout stream starts identically across configurations.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(4206)
            if self.mode.startswith('moe_'):
                self.tail=copy.deepcopy(self.core)
                self.gate=nn.Sequential(nn.Linear(width,32),nn.SiLU(),nn.Linear(32,1))
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.constant_(self.gate[-1].bias,math.log(tail_prevalence/(1-tail_prevalence)))
            elif self.mode=='physics':
                self.aux_head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,8))

    def forward_details(self,batch):
        # Explicit whitelist blocks targets, auxiliary labels, and truth regions.
        reco={key:batch[key] for key in INPUTS}
        embedding=self.backbone(reco)
        core=self.core(embedding)
        result={'prediction':core}
        if self.mode.startswith('moe_'):
            tail=self.tail(embedding)
            logit=self.gate(embedding).squeeze(-1)
            with torch.autocast(device_type=core.device.type,enabled=False):
                probability=logit.float().sigmoid()
                used=(probability if self.mode=='moe_learned' else self.tail_prevalence.expand_as(probability))
                prediction=(1-used[:,None])*core.float()+used[:,None]*tail.float()
            result.update(prediction=prediction,core=core,tail=tail,
                          gate_logit=logit,gate_prob=probability,used_gate=used)
        elif self.mode=='physics':
            from physics_aux import reconstruct_observables
            auxiliary=self.aux_head(embedding)
            physical_pair=reconstruct_observables(auxiliary,self.aux_mean,self.aux_std,
                                                  self.main_mean,self.main_effective_std)
            result.update(aux=auxiliary,physical_pair=physical_pair)
        return result

    def forward(self,batch):
        return self.forward_details(batch)['prediction']


def build_model(base_model,cfg,tail_prevalence,aux_mean,aux_std,main_mean,main_effective_std):
    return StructureRegressor(base_model,cfg,tail_prevalence,aux_mean,aux_std,main_mean,main_effective_std)


def objective_components(model,details,batch):
    """Training truths enter only here, after the reco-only forward pass."""
    prediction=details['prediction']
    with torch.autocast(device_type=prediction.device.type,enabled=False):
        pred=prediction.float();target=batch['target'].float()
        if pred.shape!=target.shape or pred.ndim!=2 or pred.shape[1]!=4:
            raise ValueError('Expected [B,4] target and prediction')
        paired=2*F.huber_loss(pred,target,delta=1.)
        emd=(pred.sort(dim=0).values-target.sort(dim=0).values).abs().mean()
        zero=pred.new_zeros(())
        specialization=gate_bce=auxiliary=consistency=zero
        if model.mode.startswith('moe_'):
            label=batch['tail_label'].detach().float().reshape(-1)
            if label.shape!=(len(pred),) or not ((label==0)|(label==1)).all():
                raise ValueError('tail_label must be the binary training mass region')
            prevalence=model.tail_prevalence.float()
            core_loss=2*F.huber_loss(details['core'].float(),target,delta=1.,reduction='none').mean(dim=1)
            tail_loss=2*F.huber_loss(details['tail'].float(),target,delta=1.,reduction='none').mean(dim=1)
            # Population-frequency normalization, no stochastic batch-count division.
            specialization=.5*((core_loss*(1-label)/(1-prevalence)).mean()+
                                (tail_loss*label/prevalence).mean())
            # Both expert arms receive the same classifier supervision; only the
            # output-routing rule changes in the fixed vs learned comparison.
            gate_bce=F.binary_cross_entropy_with_logits(details['gate_logit'].float(),label)
        elif model.mode=='physics':
            auxiliary_target=batch['aux_target'].detach().float()
            if auxiliary_target.shape!=(len(pred),8):raise ValueError('Expected [B,8] auxiliary target')
            auxiliary=2*F.huber_loss(details['aux'].float(),auxiliary_target,delta=1.)
            consistency=2*F.huber_loss(pred[:,:2],details['physical_pair'].float(),delta=1.)
        total=paired+.1*emd+.2*specialization+.05*gate_bce+.2*auxiliary+.05*consistency
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite structural objective')
        return dict(loss=total,point=paired,emd=emd,specialization=specialization,
                    gate_bce=gate_bce,auxiliary=auxiliary,consistency=consistency,
                    mse=(pred-target).square().mean())


@torch.no_grad()
def summarize_details(details,tail_labels=None):
    """Call on the combined full validation tensors for distribution diagnostics."""
    result={}
    if 'gate_prob' in details:
        probability=details['gate_prob'].float()
        result.update(gate_mean=float(probability.mean()),used_gate_mean=float(details['used_gate'].float().mean()),
                      gate_std=float(probability.std(unbiased=False)),
                      gate_quantiles=torch.quantile(probability,probability.new_tensor([0,.01,.1,.5,.9,.99,1])).cpu().tolist(),
                      gate_above_half_fraction=float((probability>.5).float().mean()),
                      expert_mean_abs_difference=float((details['tail'].float()-details['core'].float()).abs().mean()),
                      expert_mean_abs_difference_per_observable=(details['tail'].float()-details['core'].float()).abs().mean(0).cpu().tolist())
        if tail_labels is not None:
            mask=torch.as_tensor(tail_labels,device=probability.device,dtype=torch.bool).reshape(-1)
            if mask.shape!=probability.shape:raise ValueError('Wrong diagnostic truth-region shape')
            result['truth_region_gate']={}
            for name,region in [('tail',mask),('core',~mask)]:
                result['truth_region_gate'][name]=dict(n=int(region.sum()),mean=float(probability[region].mean()) if region.any() else None)
    if 'aux' in details:
        result['physical_pair_mean_abs_gap']=float((details['prediction'][:,:2].float()-details['physical_pair'].float()).abs().mean())
    return result
