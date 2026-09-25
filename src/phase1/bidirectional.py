"""Two-rank exercises for matching point-to-point communication order."""

import argparse

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group


def naive_exchange(rank: int) -> torch.Tensor:
    """Exercise the risky pattern where both ranks send before receiving.

    Run with exactly two ranks. Rank 0 sends ``[10]`` to rank 1, and rank 1
    sends ``[20]`` to rank 0; each then tries to receive the other's value.
    Predict whether this can deadlock before running this scenario. Whether it
    hangs can depend on transport buffering and message size.
    """
    incoming = torch.empty(1, dtype=torch.int64)
    outgoing = _outgoing_for_rank(rank)
    if rank == 0:
        # Consider what rank 1 is doing while rank 0 is blocked in send().
        dist.send(tensor=outgoing, dst=1)
        dist.recv(tensor=incoming, src=1)
    elif rank == 1:
        # Both ranks now attempt the send before either posts its receive.
        dist.send(tensor=outgoing, dst=0)
        dist.recv(tensor=incoming, src=0)
    else:
        raise ValueError(f"naive_exchange supports ranks 0 and 1; got rank {rank}")
    return incoming


def ordered_exchange(rank: int) -> torch.Tensor:
    """Exercise a matching order that lets one receive make progress first.

    Rank 0 sends ``[10]`` then receives ``[20]``. Rank 1 receives ``[10]``
    first, then sends ``[20]``. Run this scenario and compare it with
    :func:`naive_exchange`.
    """
    incoming = torch.empty(1, dtype=torch.int64)
    outgoing = _outgoing_for_rank(rank)
    if rank == 0:
        dist.send(tensor=outgoing, dst=1)
        dist.recv(tensor=incoming, src=1)
    elif rank == 1:
        dist.recv(tensor=incoming, src=0)
        dist.send(tensor=outgoing, dst=0)
    else:
        raise ValueError(f"ordered_exchange supports ranks 0 and 1; got rank {rank}")
    return incoming


def _outgoing_for_rank(rank: int) -> torch.Tensor:
    """Build this rank's payload, rejecting ranks outside the two-rank exercise."""
    if rank not in (0, 1):
        raise ValueError(f"bidirectional exchange requires rank 0 or 1; got rank {rank}")
    return torch.tensor([10 if rank == 0 else 20], dtype=torch.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("ordered", "naive"), default="ordered")
    args = parser.parse_args()

    setup_process_group(timeout_seconds=10)
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != 2:
            raise ValueError(
                f"bidirectional exercises require exactly 2 ranks; got {world_size}"
            )
        exchange = naive_exchange if args.scenario == "naive" else ordered_exchange
        incoming = exchange(rank)
        print(f"rank {rank} received {incoming.tolist()}", flush=True)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
