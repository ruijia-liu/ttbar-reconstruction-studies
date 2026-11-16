"""RankSim auxiliary loss on a trainable event representation.

Adapted from BorealisAI/ranksim-imbalanced-regression, ranking.py and RankSim
README example. Copyright (c) 2021-present, Royal Bank of Canada.
This adapted file is licensed CC BY-NC-SA 4.0 (no warranties):
https://creativecommons.org/licenses/by-nc-sa/4.0/
Source: https://github.com/BorealisAI/ranksim-imbalanced-regression
Paper: https://proceedings.mlr.press/v162/gong22a.html
Changes: vectorized anchors, validation, float32 computation, deterministic
duplicate-label handling, differentiable zero for uninformative small batches.
The official SUM of per-anchor mean squared rank errors is preserved.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def rank_normalised(sequence: Tensor) -> Tensor:
    """Descending ranks in [1/N, 1], preserving the official tie convention."""
    order = torch.argsort(sequence, dim=-1).flip(-1)
    ranks = torch.argsort(order, dim=-1)
    return (ranks.to(torch.float32) + 1.0) / sequence.shape[-1]


class TrueRanker(torch.autograd.Function):
    """Black-box rank differentiation used by the official RankSim code."""

    @staticmethod
    def forward(ctx, sequence: Tensor, lambda_val: float) -> Tensor:
        ranks = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, ranks)
        return ranks

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        sequence, ranks = ctx.saved_tensors
        perturbed = rank_normalised(sequence + ctx.lambda_val * grad_output)
        return (perturbed - ranks) / (ctx.lambda_val + 1e-8), None


def rank_similarity_loss(
    features: Tensor, target_mass: Tensor, lambda_val: float = 2.0
) -> Tensor:
    """Match feature cosine-neighbor rankings to target-mass distance rankings.

    Pass a caller-selected subset (typically 256 rows) of TRAINABLE features.
    Keep the main supervised loss on every training row. This function does
    not sample, change RNG state, or use truth labels as inference inputs.
    The result sums one mean squared rank error per unique target anchor.
    Consequently its scale depends on the number of retained rows.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if target_mass.ndim not in (1, 2) or (
        target_mass.ndim == 2 and target_mass.shape[1] != 1
    ):
        raise ValueError("target_mass must have shape [N] or [N, 1]")
    if len(features) != len(target_mass):
        raise ValueError("feature and target row counts differ")
    if features.device != target_mass.device:
        raise ValueError("features and target_mass must share a device")
    if not math.isfinite(lambda_val) or lambda_val <= 0:
        raise ValueError("lambda_val must be finite and positive")
    with torch.autocast(device_type=features.device.type, enabled=False):
        x = features.float()
        y = target_mass.detach().reshape(-1).float()
        if len(x) < 2:
            return x.sum() * 0.0
        # Official RankSim samples one representative per distinct target.
        # The caller already randomly selects this subset; choose its first
        # representative deterministically without consuming a second RNG.
        unique, inverse = torch.unique(y, return_inverse=True)
        if unique.numel() != y.numel():
            first = torch.full_like(unique, len(y), dtype=torch.long)
            first.scatter_reduce_(
                0, inverse, torch.arange(len(y), device=y.device),
                reduce="amin", include_self=True,
            )
            x, y = x[first], y[first]
        if len(x) < 2:
            return x.sum() * 0.0
        normalized = F.normalize(x, p=2, dim=1)
        similarities = normalized @ normalized.T
        target_ranks = rank_normalised(-(y[:, None] - y[None, :]).abs())
        feature_ranks = TrueRanker.apply(similarities, float(lambda_val))
        return (feature_ranks - target_ranks).square().mean(dim=1).sum()
