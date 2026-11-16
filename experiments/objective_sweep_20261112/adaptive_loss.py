"""Normalised Barron residual likelihood with four learned robustness values.

This experiment learns shape only: c=1 in standardised target units is fixed.
The scalar returned is 2 * mean(rho + log Z), matching the local residual
curvature of 2*Huber. At alpha=1, rho is pseudo-Huber, not piecewise Huber.
See adaptive_sources.txt and test_adaptive_loss.py for the independently
integrated partition table and numerical checks. No upstream code is vendored.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


ALPHA_MIN = 0.1
ALPHA_MAX = 1.9


class AdaptiveRobustLoss(nn.Module):
    """Learn one global alpha per observable; no event-dependent loss routing."""

    def __init__(self, dimensions: int = 4):
        super().__init__()
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions
        # Zero is the logit giving alpha=1; deterministic initialisation.
        self.raw_alpha = nn.Parameter(torch.zeros(dimensions, dtype=torch.float32))
        table_path = Path(__file__).with_name("barron_partition.npz")
        with np.load(table_path, allow_pickle=False) as table:
            grid = table["alpha"]
            values = table["log_z"]
            slopes = table["d_log_z"]
        if grid.shape != (361,) or values.shape != grid.shape or slopes.shape != grid.shape:
            raise ValueError("Unexpected partition table dimensions")
        if not all(np.isfinite(v).all() for v in (grid, values, slopes)):
            raise ValueError("Nonfinite partition table")
        if not np.allclose(grid, np.linspace(ALPHA_MIN, ALPHA_MAX, 361), rtol=0, atol=1e-12):
            raise ValueError("Unexpected partition table grid")
        step = float(grid[1] - grid[0])
        # Form cubic coefficients in float64 offline, then store float32.
        # Horner evaluation avoids cancellation of almost-equal knot values
        # when differentiating a conventional Hermite basis in float32.
        delta = np.diff(values)
        linear = step * slopes[:-1]
        quadratic = 3.0 * delta - step * (2.0 * slopes[:-1] + slopes[1:])
        cubic = -2.0 * delta + step * (slopes[:-1] + slopes[1:])
        coeff = np.stack((values[:-1], linear, quadratic, cubic), axis=1)
        self.register_buffer("partition_coeff", torch.from_numpy(coeff.astype(np.float32)))
        self.table_step = step

    def alpha(self) -> Tensor:
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * self.raw_alpha.sigmoid()

    def log_partition(self, alpha: Tensor) -> Tensor:
        """Cubic log Z on the declared bounded alpha interval."""
        position = (alpha - ALPHA_MIN) / self.table_step
        interval = position.floor().to(torch.int64).clamp(0, self.partition_coeff.shape[0] - 1)
        fraction = position - interval.to(position.dtype)
        coeff = self.partition_coeff.to(dtype=alpha.dtype)[interval]
        return coeff[..., 0] + fraction * (
            coeff[..., 1] + fraction * (coeff[..., 2] + fraction * coeff[..., 3])
        )

    @staticmethod
    def rho(residual: Tensor, alpha: Tensor) -> Tensor:
        """Barron rho for 0<alpha<2 and fixed c=1, without cancellation at zero."""
        beta = 2.0 - alpha
        exponent = 0.5 * alpha * torch.log1p(residual.square() / beta)
        return (beta / alpha) * torch.expm1(exponent)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim != 2 or pred.shape != target.shape or pred.shape[1] != self.dimensions:
            raise ValueError("pred and target must have identical [batch, dimensions] shapes")
        if pred.shape[0] == 0 or pred.device != target.device or pred.device != self.raw_alpha.device:
            raise ValueError("Need a nonempty batch and matching module/input devices")
        if not pred.is_floating_point() or not target.is_floating_point():
            raise TypeError("pred and target must be floating point")
        with torch.autocast(device_type=pred.device.type, enabled=False):
            residual = pred.float() - target.float()
            alpha = self.alpha().float()
            nll = self.rho(residual, alpha) + self.log_partition(alpha)
            return 2.0 * nll.mean()

    @torch.no_grad()
    def diagnostics(self) -> dict:
        return {
            "alpha": self.alpha().detach().cpu().tolist(),
            "alpha_bounds": [ALPHA_MIN, ALPHA_MAX],
            "scale": [1.0] * self.dimensions,
            "scale_learned": False,
            "scope": "one global robustness parameter per observable",
            "normalised_nll": True,
            "point_weight": 2.0,
            "initial_alpha": 1.0,
            "initial_shape": "pseudo-Huber; not piecewise Huber",
        }
