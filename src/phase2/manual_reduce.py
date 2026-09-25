"""Reduction algorithms assembled from Phase 1 point-to-point operations."""
import torch
import torch.distributed as dist


def manual_reduce_sum(tensor: torch.Tensor, dst: int) -> torch.Tensor:
    """Sum all world tensors at dst; non-destination results are unspecified."""
    # TODO: IMPLEMENT using only dist.send() and dist.recv().
    raise NotImplementedError


def tree_reduce_sum(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    """Binary tree sum; initially require dst=0 and power-of-two world size."""
    # TODO: IMPLEMENT using only dist.send() and dist.recv().
    raise NotImplementedError
