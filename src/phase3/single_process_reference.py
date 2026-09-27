"""Deterministic tiny regression problem used as the mathematical reference."""
from __future__ import annotations

import torch
from torch import nn


def make_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed, small regression dataset suitable for exact partitioning."""
    # TODO: IMPLEMENT: define deterministic CPU features [N, 4] and targets
    # [N, 1], with no random state dependence, and return them.
    raise NotImplementedError


def make_model(seed: int = 123) -> nn.Module:
    """Create the capstone MLP with reproducible initial parameters.

    TODO: IMPLEMENT: seed PyTorch locally, construct Linear(4,16), ReLU,
    Linear(16,1), and ensure parameters are CPU float tensors.
    """
    raise NotImplementedError


def reference_step(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                   learning_rate: float = 0.05) -> dict[str, object]:
    """Run one full-batch mean-MSE SGD step and return loss/grads/weights.

    TODO: IMPLEMENT: snapshot weights, zero gradients, calculate mean MSE,
    backpropagate, snapshot gradients, step plain SGD, snapshot resulting
    weights, and return those snapshots with the scalar loss for comparison.
    """
    raise NotImplementedError
