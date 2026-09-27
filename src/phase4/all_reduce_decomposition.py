"""Demonstrate AllReduce SUM semantics as ReduceScatter followed by AllGather."""
import torch
import torch.distributed as dist


def all_reduce_via_reduce_scatter_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Return the full sum on each rank without calling dist.all_reduce."""
    if tensor.ndim != 1 or tensor.numel() % dist.get_world_size():
        raise ValueError("tensor must be 1-D with elements divisible by world size")
    # TODO: IMPLEMENT
    raise NotImplementedError("implement ReduceScatter + AllGather")
