"""Broadcast algorithms assembled from Phase 1 point-to-point operations."""
import torch
import torch.distributed as dist


def manual_broadcast(tensor: torch.Tensor, src: int) -> torch.Tensor:
    """Naive source-to-every-rank broadcast. All world ranks call this."""
    # TODO: IMPLEMENT using only dist.send() and dist.recv().
    raise NotImplementedError


def tree_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Binary tree broadcast; initially require src=0 and power-of-two world size."""
    # TODO: IMPLEMENT using only dist.send() and dist.recv().
    raise NotImplementedError
