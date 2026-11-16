"""Synthetic demonstration of fixed grouping, checkpoint metadata and backprop."""

import json

import numpy as np
import torch
from torch import nn

from ttbar_reco import GroupDefinition, assign_groups, conditional_ot_loss, fit_groups, make_directions


def main() -> None:
    torch.manual_seed(42)
    # In a real run these are predictions for the full training split.
    training_baseline_mass = np.linspace(300.0, 1500.0, 1000, dtype=np.float32)
    reference = fit_groups(training_baseline_mass)
    checkpoint_metadata = json.dumps(reference.to_dict())
    restored_reference = GroupDefinition.from_dict(json.loads(checkpoint_metadata))

    # Both prediction and target must use the same train-fitted standardisation.
    batch_mass = torch.tensor(training_baseline_mass[::4])
    fixed_groups = assign_groups(batch_mass, restored_reference)
    baseline_z = torch.randn(len(batch_mass), 4)
    target_z = baseline_z + 0.1 * torch.randn_like(baseline_z)
    correction = nn.Linear(4, 4)
    nn.init.zeros_(correction.weight)
    nn.init.zeros_(correction.bias)
    prediction_z = baseline_z + 0.5 * torch.tanh(correction(baseline_z))

    point_loss = nn.functional.huber_loss(prediction_z, target_z, delta=1.0)
    cot_loss = conditional_ot_loss(
        prediction_z, target_z, fixed_groups,
        restored_reference.probabilities, make_directions(),
    )
    # The original correction objective retains the marginal W1/EMD term.
    marginal_loss = (prediction_z.sort(dim=0).values - target_z.detach().sort(dim=0).values).abs().mean()
    loss = 2.0 * point_loss + 0.1 * marginal_loss + 0.1 * cot_loss
    loss.backward()
    print(f"Training groups: {restored_reference.counts}")
    print(f"Huber={point_loss.item():.6f}; marginal W1={marginal_loss.item():.6f}; COT={cot_loss.item():.6f}")
    print(f"Correction gradient norm={correction.weight.grad.norm().item():.6f}")


if __name__ == '__main__':
    main()
