"""Differentiable, truth-row-balanced local migration surrogate.

Predictions and targets have the existing four standardized coordinates.
Each coordinate defines a logistic distribution centered on the SAME scalar
prediction used for ordinary regression. Its interval probabilities include
underflow and overflow and sum to one; no prediction/probability clipping or
finite-window renormalization is applied. This loss supplements paired
regression; alone it can favor bin centers and drive overflow predictions out.

Temperature is temperature_fraction * median(interior physical bin width) /
target_std. The supplied std must be the actual preprocessing denominator
(pass std+epsilon if that is what the dataset uses). Labels use float32
subtraction/division for standardized edges, matching target preprocessing
and the float32 loss
and full-training count helper. Exact last-edge values belong to the final
interior bin, as in numpy.histogram; internal edges belong to the upper bin.

Initial inverse-frequency weights are N/(number_of_categories * count).
Zero-count categories receive max_weight. Cap these INITIAL weights, then
normalize their training-frequency-weighted mean to one. Consequently the
FINAL maximum may exceed max_weight; diagnostics records both stages.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

KEYS = ['ttbar_mass', 'delta_abs_y', 'cos_K_plus', 'cos_phi']
PHYSICAL_BINS = {
    'ttbar_mass': [300,380,440,500,750,2000],
    'delta_abs_y': [-2.4,-1,-.5,0,.5,1,2.4],
    'cos_K_plus': [-1,-.5,0,.5,1],
    'cos_phi': [-1,-.5,0,.5,1],
}


def _statistics(mean, std):
    mean = torch.as_tensor(mean).detach().to(device='cpu',dtype=torch.float64)
    std = torch.as_tensor(std).detach().to(device='cpu',dtype=torch.float64)
    if mean.shape != (4,) or std.shape != (4,):
        raise ValueError('mean and std must each contain exactly four coordinates')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Target means must be finite and standard deviations finite/positive')
    return mean, std


def _standardized_edges(mean, std):
    edges = [(torch.tensor(PHYSICAL_BINS[key],dtype=torch.float32)-mean[j].float())/std[j].float()
             for j,key in enumerate(KEYS)]
    if any(not torch.isfinite(e).all() or not (e[1:] > e[:-1]).all() for e in edges):
        raise ValueError('Target statistics produce invalid float32 bin edges')
    return edges


def _bin_indices(column, edges):
    # right=True: left endpoint belongs to the interval starting there.
    index = torch.bucketize(column.contiguous(),edges,right=True)
    return torch.where(column == edges[-1],index-1,index)


def compute_bin_counts(target, mean, std, chunk_size=262144):
    """Return four full-training count lists, including both overflow categories.

    target must be a CPU tensor/NumPy array [N,4] of standardized training
    targets. Chunking avoids a full-dataset conversion or index allocation.
    No validation data, prediction or model input is used to fit these weights.
    """
    if not isinstance(chunk_size,int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    mean,std = _statistics(mean,std)
    edges = _standardized_edges(mean,std)
    values = torch.as_tensor(target)
    if values.device.type != 'cpu':
        raise ValueError('compute_bin_counts requires full-training targets on CPU')
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not values.is_floating_point():
        raise ValueError('Expected nonempty floating-point standardized training targets [N,4]')
    counts = [torch.zeros(len(e)+1,dtype=torch.int64) for e in edges]
    for start in range(0,len(values),chunk_size):
        chunk = values[start:start+chunk_size].detach().float()
        if not torch.isfinite(chunk).all():
            raise ValueError('All training targets must be finite; rows cannot be dropped')
        for j,e in enumerate(edges):
            counts[j] += torch.bincount(_bin_indices(chunk[:,j],e),minlength=len(e)+1)
    return [value.tolist() for value in counts]


def _log_probabilities(column, edges, tau):
    """Stable log logistic interval probabilities, including infinite tails."""
    low,high = edges[:-1],edges[1:]
    p = column[:,None]
    interior = (F.logsigmoid((high[None,:]-p)/tau)
                + F.logsigmoid((p-low[None,:])/tau)
                + torch.log(-torch.expm1(-(high-low)/tau))[None,:])
    underflow = F.logsigmoid((edges[0]-column)/tau)[:,None]
    overflow = F.logsigmoid((column-edges[-1])/tau)[:,None]
    return torch.cat((underflow,interior,overflow),dim=1)


class MigrationLoss(nn.Module):
    def __init__(self, mean, std, bin_counts, temperature_fraction=.1, max_weight=5):
        super().__init__()
        if not math.isfinite(float(temperature_fraction)) or temperature_fraction <= 0:
            raise ValueError('temperature_fraction must be finite and positive')
        if not math.isfinite(float(max_weight)) or max_weight <= 0:
            raise ValueError('max_weight must be finite and positive')
        mean,std = _statistics(mean,std)
        edges = _standardized_edges(mean,std)
        if isinstance(bin_counts,Mapping):
            if set(bin_counts) != set(KEYS):
                raise ValueError('Count mapping must contain the four observable keys')
            bin_counts = [bin_counts[key] for key in KEYS]
        if len(bin_counts) != 4:
            raise ValueError('bin_counts must contain four category-count vectors')
        self.temperature_fraction = float(temperature_fraction)
        self.max_weight = float(max_weight)
        self.register_buffer('target_mean',mean)
        self.register_buffer('target_std',std)
        temperatures = [];physical_temperatures = [];totals = []
        for j,key in enumerate(KEYS):
            physical = torch.tensor(PHYSICAL_BINS[key],dtype=torch.float64)
            # torch.median chooses the lower middle for even length; quantile
            # gives the conventional midpoint median needed for this definition.
            physical_tau = float(torch.quantile(physical[1:]-physical[:-1],.5))*self.temperature_fraction
            physical_temperatures.append(physical_tau)
            temperatures.append(physical_tau/float(std[j]))
            count = torch.as_tensor(bin_counts[j]).detach().to(device='cpu',dtype=torch.float64)
            if count.shape != (len(edges[j])+1,) or not torch.isfinite(count).all():
                raise ValueError('Count vector shape or values are invalid for '+key)
            if (count < 0).any() or not torch.equal(count,count.floor()) or count.sum() <= 0:
                raise ValueError('Counts must be nonnegative integers with positive total for '+key)
            total = float(count.sum());totals.append(total)
            raw = total/(len(count)*count.clamp_min(1))
            raw = torch.where(count > 0,raw,torch.full_like(raw,self.max_weight))
            capped = raw.clamp_max(self.max_weight)
            normalizer = (capped*count).sum()/total
            weights = (capped/normalizer).float()
            self.register_buffer('edges_'+str(j),edges[j])
            self.register_buffer('counts_'+str(j),count.to(torch.int64))
            self.register_buffer('capped_weights_'+str(j),capped.float())
            self.register_buffer('weights_'+str(j),weights)
            self.register_buffer('weight_normalizer_'+str(j),normalizer.float())
        if len(set(totals)) != 1:
            raise ValueError('Every coordinate must count the same full training sample')
        temperature = torch.tensor(temperatures,dtype=torch.float32)
        if not torch.isfinite(temperature).all() or not (temperature > 0).all():
            raise ValueError('Invalid standardized float32 temperatures')
        self.register_buffer('temperature',temperature)
        self.register_buffer('physical_temperature',torch.tensor(physical_temperatures,dtype=torch.float64))

    def forward(self, pred, target):
        if pred.ndim != 2 or pred.shape[1] != 4 or target.shape != pred.shape or not len(pred):
            raise ValueError('pred and target must have matching nonempty [B,4] shapes')
        if not pred.is_floating_point() or not target.is_floating_point():
            raise ValueError('pred and target must be floating-point standardized coordinates')
        if pred.device != target.device or pred.device != self.temperature.device:
            raise ValueError('pred, target and MigrationLoss must be on the same device')
        # Preserve pred gradients while doing every probability/loss operation
        # in float32 even if the caller supplies bfloat16 under autocast.
        with torch.autocast(device_type=pred.device.type,enabled=False):
            p,t = pred.float(),target.detach().float()
            if not torch.isfinite(p).all() or not torch.isfinite(t).all():
                raise ValueError('Nonfinite predictions/targets cannot enter migration loss')
            losses = []
            for j in range(4):
                edges = getattr(self,'edges_'+str(j)).float()
                index = _bin_indices(t[:,j],edges)
                log_probability = _log_probabilities(p[:,j],edges,self.temperature[j].float())
                selected = log_probability.gather(1,index[:,None]).squeeze(1)
                weights = getattr(self,'weights_'+str(j)).float()[index]
                losses.append((-selected*weights).mean())
            return torch.stack(losses).mean()

    def diagnostics(self):
        coordinates = {}
        for j,key in enumerate(KEYS):
            counts = getattr(self,'counts_'+str(j)).detach().cpu()
            weights = getattr(self,'weights_'+str(j)).detach().cpu()
            coordinates[key] = dict(physical_edges=PHYSICAL_BINS[key],
                standardized_edges=getattr(self,'edges_'+str(j)).detach().cpu().tolist(),
                temperature_physical=float(self.physical_temperature[j]),temperature_standardized=float(self.temperature[j]),
                counts=counts.tolist(),training_rows=int(counts.sum()),
                capped_weights_before_normalization=getattr(self,'capped_weights_'+str(j)).detach().cpu().tolist(),
                weight_normalizer=float(getattr(self,'weight_normalizer_'+str(j))),weights=weights.tolist(),
                final_max_weight=float(weights.max()),
                training_expected_weight=float((weights.double()*counts).sum()/counts.sum()))
        return dict(keys=KEYS,coordinates=coordinates,temperature_fraction=self.temperature_fraction,
            initial_weight_cap=self.max_weight,
            weighting='N/(K*count), empty categories set to cap; cap first then normalize training E[weight]=1; final cap can exceed initial cap',
            category_order='underflow, ascending finite intervals, overflow',
            edge_convention='float32 (physical_edge-mean)/effective_std; left-inclusive and right-exclusive except final finite edge is included',
            reduction='mean(weighted negative log interval probability) per coordinate, then mean of four coordinates; no batch weight renormalization',
            prediction_clipping=False,probability_clipping=False,finite_window_probability_renormalization=False,
            counts_fit='full training targets only; no inference routing or extra model inputs',
            warning='A coarse-bin surrogate, not exact Trace or Res optimization; retain paired continuous regression loss')
