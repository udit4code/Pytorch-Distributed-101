"""Message-order and optional tag experiments."""

import torch
import torch.distributed as dist


def disagreeing_order(*, receive_b_first: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    a, b = torch.tensor([1]), torch.tensor([2])
    first, second = torch.empty_like(a), torch.empty_like(b)
    if rank == 0:
        # TODO: IMPLEMENT sends A then B, with matching peer.
        pass
    elif rank == 1:
        # TODO: IMPLEMENT choose receive order and observe payload placement.
        pass
    return first, second


def tagged_messages() -> None:
    """Optional: probe Gloo tag behavior for the installed PyTorch version."""
    # TODO: IMPLEMENT tag=10/20 experiment if supported by the local backend.
    raise NotImplementedError("Optional tag experiment")
