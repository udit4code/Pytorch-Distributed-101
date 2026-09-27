"""Manual synchronous data-parallel SGD capstone; deliberately no DDP."""
from __future__ import annotations

from torch import nn


def synchronize_gradients(model: nn.Module) -> None:
    """AllReduce each present parameter gradient and average it in place.

    TODO: IMPLEMENT: walk parameters in stable order; skip absent gradients;
    SUM each gradient across the default group and divide by world size. This
    equal averaging assumes equal local sample counts and mean-reduced local
    losses. Do not use DDP or autograd hooks.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError


def run(steps: int = 10, sync_gradients: bool = True) -> None:
    """Train deterministic rank shards, verify replicas, and emit rank-0 JSON.

    TODO: IMPLEMENT: initialize Gloo, create identical MLP replicas (prefer
    rank-0 parameter broadcast), shard the same synthetic dataset without
    overlap, and run local forward/backward followed by optional gradient
    synchronization and SGD. Log one JSON record per step on rank 0. When
    synchronization is disabled, expose divergence. Always clean up the group.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run()
