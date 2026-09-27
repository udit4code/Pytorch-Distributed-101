from __future__ import annotations

import torch
from torch import nn


def make_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed, small regression dataset suitable for exact partitioning."""

    # 16 samples, 4 features each, so four ranks can each take a shard of 4.
    # Explicit constants mean there is zero dependence on RNG state.
    inputs = torch.tensor(
        [
            [0.0,  1.0,  2.0,  3.0],
            [1.0,  2.0,  3.0,  4.0],
            [2.0,  0.0,  1.0,  3.0],
            [3.0,  1.0,  0.0,  2.0],
            [4.0,  2.0,  1.0,  0.0],
            [1.0,  3.0,  2.0,  0.0],
            [2.0,  4.0,  0.0,  1.0],
            [3.0,  0.0,  4.0,  1.0],
            [4.0,  1.0,  3.0,  2.0],
            [0.0,  4.0,  1.0,  2.0],
            [1.0,  0.0,  4.0,  3.0],
            [2.0,  3.0,  1.0,  4.0],
            [3.0,  2.0,  4.0,  0.0],
            [4.0,  3.0,  0.0,  1.0],
            [0.0,  2.0,  3.0,  4.0],
            [1.0,  4.0,  0.0,  2.0],
        ],
        dtype=torch.float32,
    )

    # Fixed regression rule:
    #
    # y = 2*x0 - x1 + 0.5*x2 + 3*x3 + 1
    weights = torch.tensor(
        [[2.0], [-1.0], [0.5], [3.0]],
        dtype=torch.float32,
    )
    bias = torch.tensor([1.0], dtype=torch.float32)

    targets = inputs @ weights + bias

    return inputs, targets


def make_model(seed: int = 123) -> nn.Module:
    """Create the capstone MLP with reproducible initial parameters."""

    # fork_rng prevents model construction from permanently modifying the
    # caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    # nn.Linear defaults to CPU float32, but make that requirement explicit.
    return model.to(device="cpu", dtype=torch.float32)


def reference_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    learning_rate: float = 0.05,
) -> dict[str, object]:
    """Run one full-batch mean-MSE SGD step and return loss/grads/weights."""

    # Snapshot initial parameters before changing anything.
    weights_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    # Remove any gradients left over from previous computation.
    model.zero_grad(set_to_none=True)

    # Forward pass.
    predictions = model(inputs)

    # Mean MSE over the complete dataset.
    loss = nn.functional.mse_loss(
        predictions,
        targets,
        reduction="mean",
    )

    # Compute dL/dθ.
    loss.backward()

    # Snapshot gradients before optimizer.step() changes parameters.
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }

    # Plain SGD: θ <- θ - learning_rate * gradient
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )
    optimizer.step()

    # Snapshot updated parameters.
    weights_after = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    return {
        "loss": loss.detach().item(),
        "weights_before": weights_before,
        "gradients": gradients,
        "weights_after": weights_after,
    }
