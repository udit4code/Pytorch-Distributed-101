"""Teaching implementations of AllReduce assembled from simpler primitives."""
import torch
import torch.distributed as dist


def naive_all_reduce_sum(tensor: torch.Tensor, root: int = 0) -> torch.Tensor:
    """Implement SUM AllReduce as Reduce followed by Broadcast.

    TODO: IMPLEMENT: validate root, reduce SUM to root, then broadcast the
    root's result so every rank's input tensor contains the same sum. Assume
    all ranks call with matching shape, dtype, and root.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError


def point_to_point_all_reduce_sum(tensor: torch.Tensor, root: int = 0) -> torch.Tensor:
    """Implement SUM AllReduce using only blocking send and recv.

    TODO: IMPLEMENT: use a deadlock-safe protocol. A simple gather-to-root
    phase can receive and accumulate one contribution per peer; then a
    distribution phase must deliver the sum to every non-root rank. Do not
    call reduce, broadcast, or all_reduce. Preserve in-place semantics.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError
