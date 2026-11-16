"""A small, independent gate using reconstructed inputs only."""
from __future__ import annotations

import torch
from torch import nn

INPUT_DIM = 45
GLOBAL_FIELDS = ["global_njet", "global_nelectron", "global_nmuon", "global_nbtagged", "global_met_met", "global_met_phi"]
P4_FIELDS = ["pt", "eta", "phi", "e", "m", "px", "py", "pz"]
JET_FIELDS = ["jet_" + name for name in P4_FIELDS] + ["jet_bTag"]
LEPTON_FIELDS = P4_FIELDS + ["charge", "is_electron"]
TARGET_FIELDS = ["ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi"]
FEATURE_NAMES = (GLOBAL_FIELDS + ["jet_mean_" + x for x in JET_FIELDS]
                 + ["jet_max_" + x for x in JET_FIELDS] + ["real_jet_count"]
                 + [f"lepton_{i}_{x}" for i in range(2) for x in LEPTON_FIELDS])


def dataset_features():
    return {"global": GLOBAL_FIELDS.copy(), "jet": JET_FIELDS.copy(),
            "lepton": LEPTON_FIELDS.copy(), "target": TARGET_FIELDS.copy()}


def extract_gate_features(batch):
    """Return [B,45]: global6, masked jet mean9/max9/count1, two leptons20.

    EventDatasetBase standardises the input coordinates on training data.
    The jet count here is the raw mask count; padded jets never enter pooling.
    Lepton order is the existing ELECTRON-then-MUON order. Neither ``target``
    nor any other unlisted batch field is accessed.
    """
    global_values = batch["global"]
    jets, jet_mask = batch["jet"], batch["jet_mask"]
    leptons = batch["lepton"]
    if global_values.ndim != 2 or global_values.shape[1] != 6:
        raise ValueError("Expected global features [batch,6]")
    n = global_values.shape[0]
    if jets.ndim != 3 or jets.shape[0] != n or jets.shape[2] != 9 or jets.shape[1] == 0:
        raise ValueError("Expected jets [batch,nonempty padded slots,9]")
    if jet_mask.shape != jets.shape[:2] or jet_mask.dtype != torch.bool:
        raise ValueError("Expected boolean real-jet mask [batch,padded slots]")
    if leptons.shape != (n, 2, 10):
        raise ValueError("Expected exactly two reconstructed leptons with ten features")
    if "lepton_mask" in batch:
        lm = batch["lepton_mask"]
        if lm.shape != (n, 2) or not bool(lm.all()):
            raise ValueError("Gate requires exactly two real leptons in every event")
    jets = jets.float()
    mask = jet_mask.unsqueeze(-1)
    count = jet_mask.sum(dim=1, keepdim=True).float()
    mean = torch.where(mask, jets, torch.zeros_like(jets)).sum(dim=1) / count.clamp_min(1)
    maximum = jets.masked_fill(~mask, -torch.inf).amax(dim=1)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((global_values.float(), mean, maximum, count, leptons.float().reshape(n, 20)), dim=1)


class GateMLP(nn.Module):
    """45 -> 64 -> 32 -> 1 logits; no dropout, class weights, or shared backbone."""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected gate input [batch,{self.input_dim}]")
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.net(features.float()).squeeze(-1)
