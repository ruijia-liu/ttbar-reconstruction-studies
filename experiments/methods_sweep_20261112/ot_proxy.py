"""Conditional sliced-W2 discrepancy and proxy-based primal-dual constraints.

This is a finite-projection, fixed-reco-group OT LOSS, not a fitted optimal
Monge map. For each fixed projection, equal-size empirical W2 squared is the
mean squared difference between sorted projected samples. Group weights are
full-training population probabilities. A batch group with fewer than two
events contributes zero without renormalizing the other group weights; the
mini-batch estimate is not claimed to be an unbiased population OT distance.

The constraints are a practical proxy-based primal-dual adaptation inspired
by Cotter et al., not their full two-player game or guaranteed randomized
solution. Network gradients use smooth rate proxies; dual updates use actual
hard train rates. No validation/test data should enter the reference fit or
dual updates. Exact physical Res/Trace remain independent evaluation metrics.

Primary sources:
https://proceedings.mlr.press/v89/rowland19a.html
https://doi.org/10.1007/s10994-021-06060-0
https://jmlr.org/papers/v20/18-616.html
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_directions(count=32, dimensions=4, seed=4201):
    if count<1 or dimensions<1:raise ValueError('Positive direction counts required')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    return F.normalize(torch.randn(count,dimensions,generator=generator),dim=1)


def assign_groups(baseline_mass, internal_edges):
    """Call on immutable baseline predictions, never on corrected predictions."""
    x=torch.as_tensor(baseline_mass)
    e=torch.as_tensor(internal_edges,dtype=x.dtype,device=x.device)
    if x.ndim!=1 or e.ndim!=1 or not torch.isfinite(x).all() or not torch.isfinite(e).all():
        raise ValueError('Invalid baseline group values')
    if len(e) and not (e[1:]>e[:-1]).all():raise ValueError('Group edges must increase')
    return torch.bucketize(x.contiguous(),e,right=True)


def fit_groups(baseline_mass, quantiles=(.6,.9)):
    x=np.asarray(baseline_mass,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid training baseline masses')
    q=np.asarray(quantiles,dtype=float)
    if np.any(q<=0) or np.any(q>=1) or np.any(np.diff(q)<=0):raise ValueError('Invalid group quantiles')
    edges=np.quantile(x,q,method='linear').astype(np.float32)
    if np.any(np.diff(edges)<=0):raise ValueError('Training baseline quantiles are not distinct')
    groups=np.searchsorted(edges,x,side='right')
    counts=np.bincount(groups,minlength=len(edges)+1)
    if np.any(counts==0):raise ValueError('Empty training group')
    return dict(internal_edges=edges.tolist(),counts=counts.tolist(),
                probabilities=(counts/len(x)).tolist(),quantiles=q.tolist(),
                fit_rows=len(x),fit_split='train',group_source='frozen baseline prediction only')


def conditional_ot_loss(pred,target,groups,group_probabilities,directions):
    if pred.ndim!=2 or pred.shape!=target.shape or pred.shape[1]!=4 or not len(pred):
        raise ValueError('Expected equal nonempty [B,4] prediction and target')
    if pred.device!=target.device:raise ValueError('Prediction and target device mismatch')
    with torch.autocast(device_type=pred.device.type,enabled=False):
        p,t=pred.float(),target.detach().float()
        g=torch.as_tensor(groups,device=p.device)
        probabilities=torch.as_tensor(group_probabilities,device=p.device,dtype=torch.float32)
        d=torch.as_tensor(directions,device=p.device,dtype=torch.float32)
        if g.shape!=(len(p),) or g.dtype not in (torch.int32,torch.int64):raise ValueError('Expected integer group per event')
        if probabilities.ndim!=1 or not len(probabilities) or not (probabilities>0).all():raise ValueError('Invalid population weights')
        if not torch.isclose(probabilities.sum(),probabilities.new_tensor(1.),atol=1e-6,rtol=0):raise ValueError('Population weights must sum to one')
        if ((g<0)|(g>=len(probabilities))).any():raise ValueError('Group outside fitted range')
        if d.ndim!=2 or d.shape[1]!=4 or not len(d) or not torch.isfinite(d).all():raise ValueError('Invalid directions')
        norms=torch.linalg.vector_norm(d,dim=1)
        if not (norms>0).all():raise ValueError('Directions must be nonzero')
        d=d/norms[:,None]
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError('Nonfinite OT inputs')
        value=p.sum()*0.
        for group in range(len(probabilities)):
            mask=g==group
            if int(mask.sum())<2:continue
            a=(p[mask]@d.T).sort(dim=0).values
            b=(t[mask]@d.T).sort(dim=0).values
            value=value+probabilities[group]*(a-b).square().mean()
        return value


def _bins(x,edges):
    # Return 0..K-1 for finite bins, -1/K for underflow/overflow.
    index=torch.bucketize(x.contiguous(),edges,right=True)-1
    return torch.where(x==edges[-1],index-1,index)


def build_reference(pred0,target,edges_z,bin_temperatures,chunk_size=262144):
    """Fit full-training hard-rate references in the same float32 z coordinates.

    Quantile thresholds are rounded to float32 BEFORE computing reference tail
    rates. Thus the baseline itself has zero hard violation up to float32
    reduction rounding (about 1e-7), including ties and empirical quantile
    interpolation. Each column quantile is computed alone
    to avoid Torch's large-array quantile-size restriction.
    """
    p=torch.as_tensor(pred0);t=torch.as_tensor(target)
    if p.device.type!='cpu' or t.device.type!='cpu':raise ValueError('Reference fit must use CPU training arrays')
    if p.ndim!=2 or p.shape!=t.shape or p.shape[1]!=4 or not len(p):raise ValueError('Bad reference shape')
    if not isinstance(chunk_size,int) or chunk_size<=0:raise ValueError('Bad chunk size')
    if len(edges_z)!=4 or len(bin_temperatures)!=4:raise ValueError('Need four bin definitions')
    all_counts=[];traces=[];quantiles=[];tail_rates=[]
    for j in range(4):
        edges=torch.as_tensor(edges_z[j],dtype=torch.float32)
        if edges.ndim!=1 or len(edges)<2 or not torch.isfinite(edges).all() or not (edges[1:]>edges[:-1]).all():
            raise ValueError('Invalid finite bins')
        count=torch.zeros(len(edges)-1,dtype=torch.int64)
        correct=torch.zeros_like(count)
        residual=np.empty(len(p),dtype=np.float32)
        for start in range(0,len(p),chunk_size):
            stop=min(start+chunk_size,len(p))
            pc=p[start:stop,j].detach().float();tc=t[start:stop,j].detach().float()
            if not torch.isfinite(pc).all() or not torch.isfinite(tc).all():raise ValueError('Nonfinite reference row')
            residual[start:stop]=(pc-tc).numpy()
            ti=_bins(tc,edges);pi=_bins(pc,edges)
            valid=(ti>=0)&(ti<len(count))
            count+=torch.bincount(ti[valid],minlength=len(count))
            correct+=torch.bincount(ti[valid&(pi==ti)],minlength=len(count))
        if not (count>0).all():raise ValueError('Every finite training truth bin must be populated')
        q=np.quantile(residual,[.1587,.8413],method='linear').astype(np.float32)
        quantiles.append(q.tolist())
        tail_rates.append([float(np.mean(residual<q[0])),float(np.mean(residual>q[1]))])
        all_counts.append(count.tolist())
        traces.append(float((correct.double()/count).mean()))
    reference=dict(edges_z=[torch.as_tensor(e,dtype=torch.float32).tolist() for e in edges_z],
        bin_probabilities=[(np.asarray(c,dtype=float)/len(p)).tolist() for c in all_counts],
        baseline_trace=traces,residual_quantiles=np.asarray(quantiles).T.tolist(),
        baseline_tail_rates=np.asarray(tail_rates).T.tolist(),
        bin_temperatures=list(map(float,bin_temperatures)),truth_bin_counts=all_counts,
        fit_rows=len(p),fit_split='train',reference_coordinates='float32 original standardized target coordinates',
        trace_denominator='Full truth-bin count including out-of-range predictions',
        residual_comparison='strict lower/upper exceedance; empirical baseline rates retained exactly',
        constraint_order=['trace_'+str(j) for j in range(4)]+['lower_'+str(j) for j in range(4)]+['upper_'+str(j) for j in range(4)])
    # Validate complete schema here, before it is passed to a GPU run.
    ProxyConstraint(reference)
    return reference


class ProxyConstraint(nn.Module):
    def __init__(self,reference,residual_temperature=.05,dual_lr=.01,dual_cap=100.,tolerance=0.):
        super().__init__()
        for name,value in [('residual_temperature',residual_temperature),('dual_lr',dual_lr),('dual_cap',dual_cap)]:
            if not math.isfinite(float(value)) or value<=0:raise ValueError('Invalid '+name)
        if not math.isfinite(float(tolerance)) or tolerance<0:raise ValueError('Invalid tolerance')
        self.residual_temperature=float(residual_temperature);self.dual_lr=float(dual_lr)
        self.dual_cap=float(dual_cap);self.tolerance=float(tolerance)
        self.reference_rows=int(reference.get('fit_rows',0))
        for name,shape in [('baseline_trace',(4,)),('residual_quantiles',(2,4)),
                           ('baseline_tail_rates',(2,4)),('bin_temperatures',(4,))]:
            value=torch.as_tensor(reference[name],dtype=torch.float32)
            if value.shape!=shape or not torch.isfinite(value).all():raise ValueError('Invalid reference '+name)
            self.register_buffer(name,value)
        if not ((self.baseline_trace>=0)&(self.baseline_trace<=1)).all():raise ValueError('Invalid baseline trace')
        if not ((self.baseline_tail_rates>=0)&(self.baseline_tail_rates<=1)).all():raise ValueError('Invalid baseline tail rates')
        if not (self.residual_quantiles[1]>=self.residual_quantiles[0]).all():raise ValueError('Invalid residual interval')
        if not (self.bin_temperatures>0).all():raise ValueError('Invalid bin temperature')
        if len(reference['edges_z'])!=4 or len(reference['bin_probabilities'])!=4:raise ValueError('Need four references')
        for j in range(4):
            e=torch.as_tensor(reference['edges_z'][j],dtype=torch.float32)
            probability=torch.as_tensor(reference['bin_probabilities'][j],dtype=torch.float32)
            if e.ndim!=1 or len(e)<2 or not torch.isfinite(e).all() or not (e[1:]>e[:-1]).all():raise ValueError('Invalid edges')
            if probability.shape!=(len(e)-1,) or not torch.isfinite(probability).all() or not (probability>0).all() or probability.sum()>1+1e-6:
                raise ValueError('Invalid full-population truth-bin probabilities')
            self.register_buffer('edges_'+str(j),e)
            self.register_buffer('weights_'+str(j),1/(len(probability)*probability))
        self.register_buffer('multipliers',torch.zeros(12,dtype=torch.float32))
        self.register_buffer('updates',torch.zeros((),dtype=torch.int64))
        self.register_buffer('last_hard',torch.zeros(12,dtype=torch.float32))

    def forward(self,pred,target):
        if pred.ndim!=2 or pred.shape!=target.shape or pred.shape[1]!=4 or not len(pred):raise ValueError('Expected equal nonempty [B,4] tensors')
        if pred.device!=target.device or pred.device!=self.multipliers.device:raise ValueError('Constraint device mismatch')
        with torch.autocast(device_type=pred.device.type,enabled=False):
            p,t=pred.float(),target.detach().float()
            if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError('Nonfinite constraint inputs')
            hard_traces=[];soft_traces=[]
            for j in range(4):
                edges=getattr(self,'edges_'+str(j));weights=getattr(self,'weights_'+str(j))
                ti=_bins(t[:,j],edges);pi=_bins(p[:,j],edges)
                valid=(ti>=0)&(ti<len(weights))
                safe=ti.clamp(0,len(weights)-1)
                weight=weights[safe]*valid.float()
                hard_traces.append((weight*(pi==ti).float()).mean())
                lo,hi=edges[safe],edges[safe+1];tau=self.bin_temperatures[j]
                log_probability=(F.logsigmoid((hi-p[:,j])/tau)+F.logsigmoid((p[:,j]-lo)/tau)
                                 +torch.log(-torch.expm1(-(hi-lo)/tau)))
                soft_traces.append((weight*log_probability.exp()).mean())
            error=p-t
            hard_lower=(error<self.residual_quantiles[0]).float().mean(0)
            hard_upper=(error>self.residual_quantiles[1]).float().mean(0)
            soft_lower=torch.sigmoid((self.residual_quantiles[0]-error)/self.residual_temperature).mean(0)
            soft_upper=torch.sigmoid((error-self.residual_quantiles[1])/self.residual_temperature).mean(0)
            hard=torch.cat([self.baseline_trace-torch.stack(hard_traces),
                            hard_lower-self.baseline_tail_rates[0],hard_upper-self.baseline_tail_rates[1]])-self.tolerance
            proxy=torch.cat([self.baseline_trace-torch.stack(soft_traces),
                             soft_lower-self.baseline_tail_rates[0],soft_upper-self.baseline_tail_rates[1]])-self.tolerance
            # Sum corresponds to 12 separate constraint multipliers.
            penalty=(self.multipliers.detach().clone()*proxy).sum()
            return penalty,hard.detach()

    @torch.no_grad()
    def update(self,hard_constraints):
        violation=torch.as_tensor(hard_constraints,device=self.multipliers.device,dtype=torch.float32)
        if violation.shape!=(12,) or not torch.isfinite(violation).all():raise ValueError('Invalid dual violation vector')
        self.last_hard.copy_(violation)
        self.multipliers.add_(self.dual_lr*violation).clamp_(0,self.dual_cap)
        self.updates.add_(1)

    def diagnostics(self):
        return dict(multipliers=self.multipliers.detach().cpu().tolist(),updates=int(self.updates),
            last_hard_violation=self.last_hard.detach().cpu().tolist(),dual_lr=self.dual_lr,dual_cap=self.dual_cap,
            tolerance=self.tolerance,residual_temperature=self.residual_temperature,
            bin_temperatures=self.bin_temperatures.detach().cpu().tolist(),reference_rows=self.reference_rows,
            constraint_order=['trace_'+str(j) for j in range(4)]+['lower_'+str(j) for j in range(4)]+['upper_'+str(j) for j in range(4)],
            method='proxy-based primal-dual adaptation; not full proxy-Lagrangian game',
            guarantee='No finite-budget single-network feasibility guarantee; verify exact full-validation metrics')
