"""Losses on standardised [batch, observable] targets, evaluated in float32.

The point losses use their standard definitions: Huber and log-cosh have
one-half the local quadratic curvature of MSE. Any factor of two belongs in
the experiment's point-loss weight, not in these functions.

For a fixed SW1 approximation, register ``make_directions()`` as a module
buffer once, then pass that buffer to ``sliced_wasserstein_loss`` each step.
Neither distribution loss preserves the pairing between individual events.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _pair_as_float32(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
        raise TypeError("pred and target must be torch tensors")
    if pred.ndim != 2 or pred.shape != target.shape:
        raise ValueError("pred and target must have identical [batch, dimensions] shapes")
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError("batch and dimensions must both be nonempty")
    if pred.device != target.device:
        raise ValueError("pred and target must be on the same device")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must have floating-point dtypes")
    return pred.float(), target.float()


def _positive_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def point_loss(
    pred: Tensor,
    target: Tensor,
    kind: str,
    delta: float = 1.0,
    scale: float = 1.0,
) -> Tensor:
    """Mean MSE, standard Huber, or ``scale**2 * log(cosh(residual/scale))``.

    The mean includes both events and observables. ``delta`` is the Huber
    transition in standardised target units; ``scale`` is the log-cosh scale.
    Explicitly disabling autocast keeps all loss arithmetic in float32.
    """
    p, t = _pair_as_float32(pred, target)
    with torch.autocast(device_type=p.device.type, enabled=False):
        if kind == "mse":
            return F.mse_loss(p, t, reduction="mean")
        if kind == "huber":
            return F.huber_loss(p, t, reduction="mean", delta=_positive_scalar(delta, "delta"))
        if kind == "logcosh":
            s = _positive_scalar(scale, "scale")
            absolute = ((p - t) / s).abs()
            # Equivalent to log(cosh), without exp(+|residual|) overflow.
            value = absolute + F.softplus(-2.0 * absolute) - math.log(2.0)
            return (s * s) * value.mean()
    raise ValueError(f"Unknown point loss {kind!r}; expected mse, huber, or logcosh")


def emd_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Average empirical W1 of the observable marginals (equal event weights).

    Each column is sorted along the batch axis. This is the mean absolute
    difference of sorted values, averaged over all events and columns.
    """
    p, t = _pair_as_float32(pred, target)
    with torch.autocast(device_type=p.device.type, enabled=False):
        return (p.sort(dim=0).values - t.sort(dim=0).values).abs().mean()


def make_directions(count: int = 32, dimensions: int = 4, seed: int = 4201) -> Tensor:
    """Return reproducible CPU float32 unit vectors of shape [count, dimensions].

    Gaussian vectors normalised to unit length sample sphere directions.
    A private CPU generator leaves the global CPU/CUDA random states alone.
    """
    for value, name in ((count, "count"), (dimensions, "dimensions")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a strictly positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn(count, dimensions, generator=generator, dtype=torch.float32)
    return F.normalize(directions, p=2, dim=1)


def sliced_wasserstein_loss(pred: Tensor, target: Tensor, directions: Tensor) -> Tensor:
    """Mean empirical W1 of projections onto fixed unit directions (SW1).

    ``directions`` must have shape [projections, dimensions]. Rows are
    normalised before projection; no additional sqrt(dimensions) factor is
    applied. Use a registered buffer on the prediction device to avoid
    repeatedly copying directions. Directions are constants, not parameters.
    """
    p, t = _pair_as_float32(pred, target)
    if not isinstance(directions, Tensor):
        raise TypeError("directions must be a torch tensor")
    if directions.ndim != 2 or directions.shape[0] == 0 or directions.shape[1] != p.shape[1]:
        raise ValueError("directions must have shape [nonempty projections, target dimensions]")
    if not directions.is_floating_point():
        raise TypeError("directions must have a floating-point dtype")
    with torch.autocast(device_type=p.device.type, enabled=False):
        d = directions.detach().to(device=p.device, dtype=torch.float32)
        norms = torch.linalg.vector_norm(d, dim=1, keepdim=True)
        if not bool(torch.isfinite(d).all()) or not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
            raise ValueError("directions must be finite, nonzero vectors with finite norms")
        d = d / norms
        projected_pred = p @ d.T
        projected_target = t @ d.T
        return (projected_pred.sort(dim=0).values - projected_target.sort(dim=0).values).abs().mean()
