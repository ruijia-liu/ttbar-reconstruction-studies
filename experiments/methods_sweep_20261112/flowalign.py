"""Two-stage FlowAlign-style correction in the four standardized observables.

Independent implementation of equations (2) and (3) from Yuan et al. (2026),
https://arxiv.org/html/2611.09863v1. See method_sources.txt for adaptations.
No inference-time flow integration or samples are required by this method.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VelocityField(nn.Module):
    """Source-conditioned velocity on z-space, with explicit scalar time."""

    def __init__(self, condition_dim: int = 132, hidden: int = 128):
        super().__init__()
        if condition_dim < 1 or hidden < 1:
            raise ValueError("condition_dim and hidden must be positive")
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim + 4 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4),
        )

    def forward(self, point_z: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if point_z.ndim != 2 or point_z.shape[1] != 4:
            raise ValueError("point_z must have shape [N, 4]")
        if condition.shape != (len(point_z), self.condition_dim):
            raise ValueError("condition has the wrong shape")
        if time.shape == (len(point_z),):
            time = time[:, None]
        if time.shape != (len(point_z), 1):
            raise ValueError("time must have shape [N] or [N, 1]")
        return self.net(torch.cat((point_z, time, condition), dim=1))


def _sample_path(
    base_z: Tensor, endpoint: Tensor, generator: torch.Generator, sigma: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if generator is None:
        raise ValueError("an explicit private torch.Generator is required")
    if base_z.ndim != 2 or base_z.shape[1] != 4 or endpoint.shape != base_z.shape:
        raise ValueError("base_z and endpoint must both have shape [N, 4]")
    if len(base_z) == 0:
        raise ValueError("an empty path batch is not supported")
    if base_z.device != endpoint.device:
        raise ValueError("base_z and endpoint must share a device")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    base = base_z.detach().float()
    noise = torch.randn(base.shape, dtype=base.dtype, device=base.device,
                        generator=generator)
    time = torch.rand((len(base), 1), dtype=base.dtype, device=base.device,
                      generator=generator)
    start = base + sigma * noise
    end = endpoint.float()
    point = (1.0 - time) * start + time * end
    velocity = end - start
    return point, time, velocity, start


def field_training_loss(
    field: nn.Module, condition: Tensor, base_z: Tensor, target_z: Tensor,
    generator: torch.Generator, *, sigma: float = 0.05,
) -> Tensor:
    """Stage 1: paired-target conditional flow matching, one path sample/row.

    Use training targets only for optimization. Validation may select a field
    checkpoint by held-out MSE. Targets and the frozen source are detached.
    The squared velocity error is averaged across events and four coordinates.
    """
    with torch.autocast(device_type=base_z.device.type, enabled=False):
        point, time, velocity, _ = _sample_path(
            base_z, target_z.detach(), generator, sigma
        )
        estimate = field(point, time, condition.detach().float())
        return (estimate - velocity).square().mean()


def path_consistency_loss(
    field: nn.Module, condition: Tensor, base_z: Tensor, pred_z: Tensor,
    generator: torch.Generator, *, sigma: float = 0.05,
) -> Tensor:
    """Stage 2: frozen-field agreement along a candidate's straight path.

    Freeze field parameters with field.requires_grad_(False) before use.
    Autograd MUST remain enabled through field(point, ...) to pred_z; setting
    the field to eval mode does not by itself freeze its parameters.
    Ground-truth targets are intentionally absent from this stage's API.
    """
    if any(parameter.requires_grad for parameter in field.parameters()):
        raise ValueError("freeze field parameters before path consistency training")
    with torch.autocast(device_type=base_z.device.type, enabled=False):
        point, time, velocity, _ = _sample_path(base_z, pred_z, generator, sigma)
        estimate = field(point, time, condition.detach().float())
        return (estimate - velocity).square().mean()
