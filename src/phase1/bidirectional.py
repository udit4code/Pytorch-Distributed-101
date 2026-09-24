"""Two-rank matching-order exercises."""

import torch
import torch.distributed as dist


def naive_exchange(rank: int) -> torch.Tensor:
    incoming = torch.empty(1, dtype=torch.int64)
    outgoing = torch.tensor([10 if rank == 0 else 20], dtype=torch.int64)
    if rank == 0:
        # TODO: IMPLEMENT naive pattern: send, then recv. Predict deadlock first.
        pass
    elif rank == 1:
        # TODO: IMPLEMENT naive pattern: send, then recv. Predict deadlock first.
        pass
    return incoming


def ordered_exchange(rank: int) -> torch.Tensor:
    incoming = torch.empty(1, dtype=torch.int64)
    outgoing = torch.tensor([10 if rank == 0 else 20], dtype=torch.int64)
    if rank == 0:
        # TODO: IMPLEMENT: send then recv.
        pass
    elif rank == 1:
        # TODO: IMPLEMENT: recv then send.
        pass
    return incoming
