"""Sharded gradient aggregation and toy SGD capstone scaffold."""
from __future__ import annotations
import torch
import torch.distributed as dist
from .reduce_scatter_demo import reduce_scatter_sum


def aggregate_gradient_shard(local_gradient: torch.Tensor) -> torch.Tensor:
    """Sum rank-local gradient contributions and retain this rank's shard."""
    return reduce_scatter_sum(local_gradient)


def toy_sharded_sgd_step(local_parameter_shard: torch.Tensor, learning_rate: float = 0.1) -> torch.Tensor:
    """Perform one deterministic sharded SGD update; complete the exercise."""
    # TODO: IMPLEMENT forward materialization, local gradients, ReduceScatter, update
    raise NotImplementedError("implement toy sharded SGD step")


def main() -> None:
    if not dist.is_initialized():
        raise RuntimeError("launch with torchrun after initializing a process group")
    raise NotImplementedError("TODO: IMPLEMENT capstone entry point")
