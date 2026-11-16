"""Frozen event gates for a four-output regression loss.

Column order is mass, delta_abs_y, cos_K_plus, cos_phi. All residuals are
already standardized with training-only statistics. ``mass_gate`` is loss
metadata, never an input feature: remove it before calling the model.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


def gated_loss(pred: Tensor, target: Tensor, gate, delta: float = 0.5) -> Tensor:
    """Mean over events and four outputs, with a detached mass-only gate.

    The mass term is ``(1-g)*2*Huber_delta + g*MSE``. The other three
    outputs always use ``2*Huber_0.5``. ``gate`` may be a scalar, [B], or
    [B, 1], with values in [0, 1]. It is detached, never fitted by this
    objective. Loss arithmetic is float32 even under autocast.
    """
    if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
        raise TypeError("pred and target must be torch tensors")
    if pred.ndim != 2 or pred.shape != target.shape or pred.shape[1] != 4:
        raise ValueError("pred and target must have identical [batch, 4] shapes")
    if pred.shape[0] == 0:
        raise ValueError("batch must be nonempty")
    if pred.device != target.device:
        raise ValueError("pred and target must be on the same device")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must have floating-point dtypes")
    if not isinstance(delta, Real) or not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("delta must be finite and strictly positive")

    g = torch.as_tensor(gate, device=pred.device)
    if g.is_complex():
        raise TypeError("gate must be real-valued")
    g = g.detach().to(torch.float32)
    if g.ndim == 0:
        g = g.expand(pred.shape[0])
    elif g.shape == (pred.shape[0], 1):
        g = g[:, 0]
    elif g.shape != (pred.shape[0],):
        raise ValueError("gate must be a scalar or have shape [batch] or [batch, 1]")
    if not bool(torch.isfinite(g).all()) or bool(((g < 0) | (g > 1)).any()):
        raise ValueError("gate must be finite and lie in [0, 1]")

    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        robust_mass = 2 * F.huber_loss(p[:, 0], t[:, 0], delta=float(delta), reduction="none")
        mass = (1 - g) * robust_mass + g * (p[:, 0] - t[:, 0]).square()
        other = 2 * F.huber_loss(p[:, 1:], t[:, 1:], delta=0.5, reduction="none")
        return (mass.sum() + other.sum()) / p.numel()


class WeightedDataset(Dataset):
    """Attach immutable, row-aligned gate metadata to an existing dataset.

    The supplied weights are detached and copied once to CPU float32.
    Cached dataset attributes such as ``stats`` and ``max_jets`` are
    forwarded to ``base``. Finalize the base cache before wrapping it.
    Both HepSetReg's already-collated dict batches and ordinary datasets
    returning lists of individual items are supported.
    """

    def __init__(self, base, weights):
        self.base = base
        w = torch.as_tensor(weights)
        if w.ndim != 1 or len(w) != len(base):
            raise ValueError("weights must have shape [len(base)]")
        if w.is_complex():
            raise TypeError("weights must be real-valued")
        w = w.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
        if not bool(torch.isfinite(w).all()) or bool(((w < 0) | (w > 1)).any()):
            raise ValueError("weights must be finite and lie in [0, 1]")
        self._weights = w

    @property
    def weights(self) -> Tensor:
        """Return a copy so external mutation cannot change fixed weights."""
        return self._weights.clone()

    def __getattr__(self, name):
        # Safe while pickle reconstructs an instance whose base is not set yet.
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _attach(item, weight):
        if not isinstance(item, Mapping):
            raise TypeError("base items/batches must be dictionaries")
        if "mass_gate" in item:
            raise ValueError("base already contains mass_gate; refusing to overwrite it")
        result = dict(item)
        target = result.get("target")
        if isinstance(target, Tensor):
            weight = weight.to(target.device)
        result["mass_gate"] = weight
        return result

    def __getitem__(self, index):
        item = self.base[index]
        # A scalar tensor index otherwise returns a view into fixed weights.
        weight = self._weights[index].clone()
        return self._attach(item, weight)

    def __getitems__(self, indices):
        index = torch.as_tensor(indices, device="cpu")
        if index.ndim != 1 or index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("batch indices must be a one-dimensional integer sequence")
        index = index.to(torch.long)
        fetch = getattr(self.base, "__getitems__", None)
        if fetch is None:
            return [self[int(i)] for i in index]
        # Normalize unsigned tensors too: raw uint8 indexing means a mask in
        # torch, whereas DataLoader's index contract is a sequence of rows.
        batch = fetch(index.tolist())
        weights = self._weights[index]
        if isinstance(batch, Mapping):
            return self._attach(batch, weights)
        if isinstance(batch, (list, tuple)) and len(batch) == len(index):
            return [self._attach(item, weight.clone()) for item, weight in zip(batch, weights)]
        raise TypeError("base batched fetch must return a dict or one item per index")
