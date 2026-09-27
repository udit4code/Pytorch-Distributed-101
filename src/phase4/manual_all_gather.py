"""Construct AllGather semantics from Phase 1 point-to-point primitives."""
import torch
import torch.distributed as dist


def manual_all_gather(local_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Exchange one same-shaped tensor with every rank using send/recv only."""
    # TODO: IMPLEMENT (do not call gather, broadcast, or an all-gather API)
    raise NotImplementedError("implement point-to-point AllGather")
