"""Broadcast algorithms assembled from Phase 1 point-to-point operations."""

import argparse

import torch
import torch.distributed as dist

from .distributed import setup_process_group, cleanup_process_group, rank_record


def manual_broadcast(tensor: torch.Tensor, src: int) -> torch.Tensor:
    """Broadcast ``tensor`` using only point-to-point send/receive operations.

    Every world rank must call this function with the same source rank. The
    source owns the input value; every receiver must provide a tensor with the
    same shape and dtype because ``recv`` writes into that buffer in place.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not (0 <= src < world_size):
        raise ValueError(f"src={src} must be in [0, {world_size})")

    if rank == src:
        # The source sends one full copy to every peer. Each send is blocking,
        # so this simple fan-out puts all sending work on the source.
        for destination in range(world_size):
            # The source already owns the data and has no matching recv, so it
            # must not send a copy to itself.
            if destination != src:
                dist.send(tensor=tensor, dst=destination)
    else:
        # recv does not allocate or return a new tensor. It copies the source
        # payload into this rank's existing local tensor.
        dist.recv(tensor=tensor, src=src)

    return tensor


# Direct fan-out makes the source send world_size - 1 copies sequentially. A
# tree spreads that work across ranks that have already received the value, so
# the number of communication rounds grows logarithmically for power-of-two
# world sizes.  Thus, Tree broadcast is especially attractive because it reduces the latency term from \(O(P)\) to \(O(\log P)\)
def tree_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast from rank zero through a binary tree of point-to-point calls."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # This learning implementation deliberately keeps the rank arithmetic
    # simple: the source is rank zero and each round doubles the number of
    # ranks that hold the value.
    if not (0 <= src < world_size):
        raise ValueError(f"src={src} must be in [0, {world_size})")
    if src != 0:
        raise NotImplementedError("tree_broadcast currently only supports src=0")
    if world_size & (world_size - 1) != 0:
        raise NotImplementedError(
            "tree_broadcast currently only supports power-of-two world sizes"
        )

    # At the beginning of a round, ranks [0, step) already have the value.
    # Each informed rank r sends to r + step, informing the next equally sized
    # block. Other ranks stay idle until their receiving round.
    step = 1
    while step < world_size:
        if rank < step:
            dist.send(tensor=tensor, dst=rank + step)
        elif rank < 2 * step:
            dist.recv(tensor=tensor, src=rank - step)
        step *= 2

    return tensor


def main() -> None:
    """Create one tensor per rank and run either broadcast implementation."""
    parser = argparse.ArgumentParser(
        description="Broadcast one integer using only dist.send and dist.recv."
    )
    parser.add_argument(
        "--algorithm",
        choices=("manual", "tree"),
        default="manual",
        help="point-to-point broadcast algorithm (default: manual)",
    )
    parser.add_argument("--src", type=int, default=0, help="source rank (default: 0)")
    parser.add_argument(
        "--value",
        type=int,
        default=100,
        help="integer initially owned by the source rank (default: 100)",
    )
    args = parser.parse_args()

    # Initialize once outside the algorithm. This keeps the algorithm reusable
    # for repeated calls, including warmups and benchmark trials.
    setup_process_group()
    try:
        rank = dist.get_rank()
        initial_value = args.value if rank == args.src else -1
        tensor = torch.tensor([initial_value], dtype=torch.int64)

        operation = f"{args.algorithm}_broadcast"
        rank_record(rank, operation, tensor, phase="before", src=args.src)
        if args.algorithm == "tree":
            tree_broadcast(tensor, src=args.src)
        else:
            manual_broadcast(tensor, src=args.src)
        rank_record(rank, operation, tensor, phase="after", src=args.src)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
