"""Blocking point-to-point exercises. Run with ``torchrun``."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group, trace_event


def sender(dst: int) -> None:
    value = torch.tensor([10, 20, 30], dtype=torch.int64)
    trace_event(rank=dist.get_rank(), operation="send", peer=dst, tensor=value)
    # TODO: IMPLEMENT: send value to dst with dist.send().


def receiver(src: int) -> None:
    # recv() writes into storage that already exists in this process. The receiver
    # therefore allocates a buffer with a compatible shape and dtype first.
    value = torch.empty(3, dtype=torch.int64)
    trace_event(rank=dist.get_rank(), operation="recv", peer=src, tensor=value)
    # TODO: IMPLEMENT: receive from src with dist.recv().
    print(f"received from rank {src}: {value.tolist()}", flush=True)


def scalar_message(rank: int, world_size: int) -> None:
    """Exercise tensor-encoded scalar and rank-ID messages; no object APIs."""
    if rank == 0:
        message = torch.tensor([42], dtype=torch.int64)
        # TODO: IMPLEMENT: send message to rank 1.
    elif rank == 1:
        message = torch.empty(1, dtype=torch.int64)
        # TODO: IMPLEMENT: receive from rank 0, then report the scalar.
    if world_size > 2 and rank == 0:
        for dst in range(1, world_size):
            message = torch.tensor([rank], dtype=torch.int64)
            # TODO: IMPLEMENT: send rank ID to each destination.


def many_to_one(rank: int, world_size: int) -> None:
    if rank == 0:
        for src in range(1, world_size):
            value = torch.empty(1, dtype=torch.int64)
            trace_event(rank=rank, operation="recv", peer=src, tensor=value)
            # TODO: IMPLEMENT: receive from this explicit source.
            print(f"received from rank {src}: {value.item()}", flush=True)
    else:
        value = torch.tensor([rank], dtype=torch.int64)
        trace_event(rank=rank, operation="send", peer=0, tensor=value)
        # TODO: IMPLEMENT: send to rank 0.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("basic", "many_to_one", "metadata", "scalar"), default="basic")
    args = parser.parse_args()
    setup_process_group()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    try:
        if args.scenario == "many_to_one":
            many_to_one(rank, world_size)
        elif args.scenario == "scalar":
            scalar_message(rank, world_size)
        elif args.scenario == "metadata":
            # Deliberately incompatible receiver buffer: run in a subprocess with timeout.
            if rank == 0:
                value = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
                # TODO: IMPLEMENT: send to rank 1.
            elif rank == 1:
                value = torch.zeros(3, dtype=torch.float32)
                # TODO: IMPLEMENT: receive into the intentionally wrong buffer.
        elif world_size >= 2:
            if rank == 0:
                sender(1)
            elif rank == 1:
                receiver(0)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
