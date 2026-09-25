"""Blocking point-to-point exercises. Run with ``torchrun``."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group, trace_event


def sender(dst: int, value: torch.Tensor | None = None) -> None:
    """Send a tensor to dst; default to the basic exercise payload."""
    if value is None:
        value = torch.tensor([10, 20, 30], dtype=torch.int64)
    trace_event(rank=dist.get_rank(), operation="send", peer=dst, tensor=value)
    # send() blocks until the matching communication makes progress. The peer
    # need not already be inside recv(), but it must eventually receive a
    # compatible tensor.
    dist.send(tensor=value, dst=dst)


def receiver(src: int, value: torch.Tensor | None = None) -> torch.Tensor:
    """Receive into value, or allocate the basic exercise's receive buffer."""
    # recv() writes into storage already allocated by this process. When no
    # buffer is supplied, allocate one matching the basic sender's tensor.
    if value is None:
        value = torch.empty(3, dtype=torch.int64)
    trace_event(rank=dist.get_rank(), operation="recv", peer=src, tensor=value)
    dist.recv(tensor=value, src=src)
    print(f"received from rank {src}: {value.tolist()}", flush=True)
    return value


def scalar_message(rank: int, world_size: int) -> None:
    """Exercise tensor-encoded scalar and rank-ID messages; no object APIs."""
    if rank == 0:
        message = torch.tensor([42], dtype=torch.int64)
        dist.send(tensor=message, dst=1)
    elif rank == 1:
        message = torch.empty(1, dtype=torch.int64)
        dist.recv(tensor=message, src=0)
        print(f"received from rank 0: {message.item()}", flush=True)
    if world_size > 2 and rank == 0:
        for dst in range(1, world_size):
            message = torch.tensor([rank], dtype=torch.int64)
            dist.send(tensor=message, dst=dst)


def many_to_one(rank: int, world_size: int) -> None:
    if rank == 0:
        for src in range(1, world_size):
            value = torch.empty(1, dtype=torch.int64)
            receiver(src, value)
            print(f"received from rank {src}: {value.item()}", flush=True)
    else:
        value = torch.tensor([rank], dtype=torch.int64)
        sender(0, value)


def validate_world_size(scenario: str, world_size: int) -> None:
    """Validate rank counts before entering a scenario's communication protocol."""
    if scenario == "metadata" and world_size != 2:
        raise ValueError("the metadata scenario requires exactly 2 ranks")
    if scenario in ("basic", "many_to_one", "scalar") and world_size < 2:
        raise ValueError(f"the {scenario} scenario requires at least 2 ranks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("basic", "many_to_one", "metadata", "scalar"), default="basic")
    args = parser.parse_args()
    setup_process_group()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    try:
        validate_world_size(args.scenario, world_size)
        if args.scenario == "many_to_one":
            many_to_one(rank, world_size)
        elif args.scenario == "scalar":
            scalar_message(rank, world_size)
        elif args.scenario == "metadata":
            # Deliberately incompatible receiver buffer: run in a subprocess with timeout.
            if rank == 0:
                value = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
                sender(1, value)
            elif rank == 1:
                value = torch.zeros(3, dtype=torch.float32)
                receiver(0, value)
        elif world_size >= 2:
            if rank == 0:
                sender(1)
            elif rank == 1:
                receiver(0)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
