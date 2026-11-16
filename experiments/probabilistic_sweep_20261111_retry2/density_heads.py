"""Small conditional four-dimensional densities with no external flow dependency.

Both heads model target_u in its caller-defined, transformed and standardized
coordinate system. By default every condition coordinate is only a covariate:
the last four coordinates may therefore contain the old point prediction z.
Optional anchor_last=True is valid ONLY when those four coordinates are an
anchor expressed in exactly the same u coordinates as target_u.

log_prob returns the JOINT log density of all four outputs, not an average.
sample returns reparameterized u samples [events, draws, 4] and retains
gradients. Its optional torch.Generator must match the condition device.
All density arithmetic is float32 even inside a surrounding autocast context.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

DIMENSIONS = 4
LOG_TWO_PI = math.log(2.0 * math.pi)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _condition(condition, condition_dim):
    if not isinstance(condition, torch.Tensor) or not condition.is_floating_point():
        raise TypeError("condition must be a floating-point torch tensor")
    if condition.ndim != 2 or condition.shape[1] != condition_dim or not condition.shape[0]:
        raise ValueError(f"Expected nonempty condition [events,{condition_dim}]")
    return condition.float()


def _value(value, condition):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("target/base values must be floating-point torch tensors")
    if value.shape != (condition.shape[0], DIMENSIONS):
        raise ValueError("Expected values [events,4] aligned with condition")
    if value.device != condition.device:
        raise ValueError("values and condition must be on the same device")
    return value.float()


def _network(input_dim, hidden_dim, output_dim):
    network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                            nn.Linear(hidden_dim, output_dim))
    # A common normal starting density, independent of conditioning, is useful
    # for numerical tests and keeps initial coupling transforms near identity.
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class ConditionalGaussian(nn.Module):
    """Learn a four-vector location and a full, positive-definite covariance."""

    def __init__(self, condition_dim=132, hidden_dim=128, anchor_last=False, min_scale=1e-3):
        super().__init__()
        self.condition_dim = _positive_int(condition_dim, "condition_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.anchor_last = bool(anchor_last)
        self.min_scale = float(min_scale)
        if self.anchor_last and condition_dim < DIMENSIONS:
            raise ValueError("An anchor requires at least four conditioning coordinates")
        if not math.isfinite(self.min_scale) or not 0 < self.min_scale < 1:
            raise ValueError("min_scale must lie strictly between zero and one")
        self.net = _network(condition_dim, hidden_dim, 14)
        rows, cols = torch.tril_indices(DIMENSIONS, DIMENSIONS)
        self.register_buffer("tril_rows", rows)
        self.register_buffer("tril_cols", cols)
        self.register_buffer("diagonal_positions", torch.nonzero(rows == cols).flatten())
        # softplus(raw_diagonal)+min_scale = 1 at initialization.
        initial_diagonal = math.log(math.expm1(1.0 - self.min_scale))
        with torch.no_grad():
            self.net[-1].bias[DIMENSIONS + self.diagonal_positions] = initial_diagonal

    def distribution_parameters(self, condition):
        """Return u-space location [N,4] and scale_tril [N,4,4]."""
        c = _condition(condition, self.condition_dim)
        with torch.autocast(device_type=c.device.type, enabled=False):
            raw = self.net(c)
            location = raw[:, :DIMENSIONS]
            if self.anchor_last:
                location = location + c[:, -DIMENSIONS:]
            packed = raw[:, DIMENSIONS:]
            scale_tril = packed.new_zeros((len(c), DIMENSIONS, DIMENSIONS))
            scale_tril[:, self.tril_rows, self.tril_cols] = packed
            diagonal = F.softplus(packed[:, self.diagonal_positions]) + self.min_scale
            diag_indices = torch.arange(DIMENSIONS, device=c.device)
            scale_tril[:, diag_indices, diag_indices] = diagonal
            return location, scale_tril

    def log_prob(self, target_u, condition):
        c = _condition(condition, self.condition_dim)
        target = _value(target_u, c)
        with torch.autocast(device_type=c.device.type, enabled=False):
            location, scale_tril = self.distribution_parameters(c)
            centered = (target - location).unsqueeze(-1)
            whitened = torch.linalg.solve_triangular(scale_tril, centered, upper=False).squeeze(-1)
            logdet = scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            return -0.5 * (DIMENSIONS * LOG_TWO_PI + whitened.square().sum(dim=-1)) - logdet

    def sample(self, condition, num_samples, generator=None):
        count = _positive_int(num_samples, "num_samples")
        c = _condition(condition, self.condition_dim)
        with torch.autocast(device_type=c.device.type, enabled=False):
            location, scale_tril = self.distribution_parameters(c)
            noise = torch.randn(len(c), count, DIMENSIONS, device=c.device,
                                dtype=torch.float32, generator=generator)
            return location[:, None, :] + torch.matmul(noise, scale_tril.transpose(-1, -2))


class ConditionalAffineCoupling(nn.Module):
    """One RealNVP map x -> z; entries marked one are left unchanged.

    The conditioner sees the unchanged coordinates and external condition.
    forward/inverse return the transformed value and their own log Jacobian.
    """

    def __init__(self, condition_dim, hidden_dim=128, mask=(1, 0, 1, 0), scale_limit=1.5):
        super().__init__()
        self.condition_dim = _positive_int(condition_dim, "condition_dim")
        _positive_int(hidden_dim, "hidden_dim")
        if len(mask) != DIMENSIONS or any(v not in (0, 1) for v in mask) or not 0 < sum(mask) < DIMENSIONS:
            raise ValueError("mask must contain four zero/one entries and change some but not all coordinates")
        self.scale_limit = float(scale_limit)
        if not math.isfinite(self.scale_limit) or self.scale_limit <= 0:
            raise ValueError("scale_limit must be finite and positive")
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32).reshape(1, DIMENSIONS))
        self.net = _network(DIMENSIONS + condition_dim, hidden_dim, 2 * DIMENSIONS)

    def _affine_parameters(self, fixed_value, condition):
        raw_scale, raw_shift = self.net(torch.cat((fixed_value * self.mask, condition), dim=-1)).chunk(2, dim=-1)
        changed = 1.0 - self.mask
        scale = self.scale_limit * torch.tanh(raw_scale / self.scale_limit) * changed
        shift = raw_shift * changed
        return scale, shift

    def forward(self, value, condition):
        c = _condition(condition, self.condition_dim)
        x = _value(value, c)
        with torch.autocast(device_type=c.device.type, enabled=False):
            scale, shift = self._affine_parameters(x, c)
            z = x * torch.exp(scale) + shift
            return z, scale.sum(dim=-1)

    def inverse(self, value, condition):
        c = _condition(condition, self.condition_dim)
        z = _value(value, c)
        with torch.autocast(device_type=c.device.type, enabled=False):
            scale, shift = self._affine_parameters(z, c)
            x = (z - shift) * torch.exp(-scale)
            return x, -scale.sum(dim=-1)


class ConditionalRealNVP(nn.Module):
    """Six alternating four-dimensional conditional affine-coupling layers."""

    def __init__(self, condition_dim=132, hidden_dim=128, num_layers=6, anchor_last=False, scale_limit=1.5):
        super().__init__()
        self.condition_dim = _positive_int(condition_dim, "condition_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.anchor_last = bool(anchor_last)
        if self.anchor_last and condition_dim < DIMENSIONS:
            raise ValueError("An anchor requires at least four conditioning coordinates")
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(condition_dim, hidden_dim,
                                      mask=(1, 0, 1, 0) if i % 2 == 0 else (0, 1, 0, 1),
                                      scale_limit=scale_limit)
            for i in range(num_layers)
        ])

    def to_base(self, target_u, condition):
        c = _condition(condition, self.condition_dim)
        value = _value(target_u, c)
        with torch.autocast(device_type=c.device.type, enabled=False):
            if self.anchor_last:
                value = value - c[:, -DIMENSIONS:]
            logdet = value.new_zeros(len(value))
            for layer in self.layers:
                value, change = layer(value, c)
                logdet = logdet + change
            return value, logdet

    def from_base(self, base_value, condition):
        c = _condition(condition, self.condition_dim)
        value = _value(base_value, c)
        with torch.autocast(device_type=c.device.type, enabled=False):
            logdet = value.new_zeros(len(value))
            for layer in reversed(self.layers):
                value, change = layer.inverse(value, c)
                logdet = logdet + change
            if self.anchor_last:
                value = value + c[:, -DIMENSIONS:]
            return value, logdet

    def log_prob(self, target_u, condition):
        c = _condition(condition, self.condition_dim)
        with torch.autocast(device_type=c.device.type, enabled=False):
            base_value, logdet = self.to_base(target_u, c)
            base_log_prob = -0.5 * (DIMENSIONS * LOG_TWO_PI + base_value.square().sum(dim=-1))
            return base_log_prob + logdet

    def sample(self, condition, num_samples, generator=None):
        count = _positive_int(num_samples, "num_samples")
        c = _condition(condition, self.condition_dim)
        with torch.autocast(device_type=c.device.type, enabled=False):
            repeated = c[:, None, :].expand(-1, count, -1).reshape(len(c) * count, self.condition_dim)
            base_value = torch.randn(len(c) * count, DIMENSIONS, device=c.device,
                                     dtype=torch.float32, generator=generator)
            values, _ = self.from_base(base_value, repeated)
            return values.reshape(len(c), count, DIMENSIONS)


def make_head(kind, condition_dim=132, hidden_dim=None, num_layers=6, anchor_last=False, hidden=None):
    """Build 'gaussian' or 'flow' ('realnvp' alias); hidden is a compatibility alias."""
    if hidden_dim is not None and hidden is not None and hidden_dim != hidden:
        raise ValueError("hidden and hidden_dim specify conflicting widths")
    width = hidden_dim if hidden_dim is not None else (hidden if hidden is not None else 128)
    if kind == "gaussian":
        return ConditionalGaussian(condition_dim=condition_dim, hidden_dim=width, anchor_last=anchor_last)
    if kind in ("flow", "realnvp"):
        return ConditionalRealNVP(condition_dim=condition_dim, hidden_dim=width,
                                 num_layers=num_layers, anchor_last=anchor_last)
    raise ValueError(f"Unknown density head {kind!r}; expected gaussian or flow")
