"""Blocking point-to-point exercises. Run with ``torchrun``."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group, trace_event


def sender(dst: int) -> None:
    value = torch.tensor([10, 20, 30], dtype=torch.int64)
    trace_event(rank=dist.get_rank(), operation="send", peer=dst, tensor=value)
    # send() is blocking: it may wait for the matching receive to make progress.
    # The receiver need not have entered recv() before this call begins, but both
    # ranks must eventually perform matching operations.
    dist.send(tensor=value, dst=dst)


def receiver(src: int) -> None:
    # recv() writes into storage that already exists in this process. The receiver
    # therefore allocates a buffer with a compatible shape and dtype first.
    value = torch.empty(3, dtype=torch.int64)
    trace_event(rank=dist.get_rank(), operation="recv", peer=src, tensor=value)
    # recv() blocks until a matching message from src is available. The sender
    # does not need to have entered send() yet, but it must eventually send a
    # compatible tensor or this receive cannot complete.
    dist.recv(tensor=value, src=src)
    print(f"received from rank {src}: {value.tolist()}", flush=True)


def scalar_message(rank: int, world_size: int) -> None:
    """Exercise tensor-encoded scalar and rank-ID messages; no object APIs."""
    if rank == 0:
        message = torch.tensor([42], dtype=torch.int64)
        dist.send(tensor=message, dst=1)
        # In the extension, rank 0 sends its rank ID to every other rank.
        # Rank 1 receives this as a second message after the scalar 42.
        if world_size > 2:
            for dst in range(1, world_size):
                rank_message = torch.tensor([rank], dtype=torch.int64)
                dist.send(tensor=rank_message, dst=dst)
    elif rank == 1:
        message = torch.empty(1, dtype=torch.int64)
        dist.recv(tensor=message, src=0)
        print(f"received scalar from rank 0: {message.item()}", flush=True)
        if world_size > 2:
            rank_message = torch.empty(1, dtype=torch.int64)
            dist.recv(tensor=rank_message, src=0)
            print(f"received rank ID from rank 0: {rank_message.item()}", flush=True)
    elif rank > 1:
        rank_message = torch.empty(1, dtype=torch.int64)
        dist.recv(tensor=rank_message, src=0)
        print(f"received rank ID from rank 0: {rank_message.item()}", flush=True)


def many_to_one(rank: int, world_size: int) -> None:
    if rank == 0:
        for src in range(1, world_size):
            value = torch.empty(1, dtype=torch.int64)
            trace_event(rank=rank, operation="recv", peer=src, tensor=value)
            dist.recv(tensor=value, src=src)
            print(f"received from rank {src}: {value.item()}", flush=True)
    else:
        value = torch.tensor([rank], dtype=torch.int64)
        trace_event(rank=rank, operation="send", peer=0, tensor=value)
        dist.send(tensor=value, dst=0)


def validate_world_size(scenario: str, world_size: int) -> None:
    """Fail consistently on every rank when a scenario has invalid rank count."""
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
                dist.send(tensor=value, dst=1)
            elif rank == 1:
                value = torch.zeros(3, dtype=torch.float32)
                # This should report a receive-size mismatch: send does not
                # resize this process's destination tensor or transmit its shape.
                dist.recv(tensor=value, src=0)
        elif world_size >= 2:
            if rank == 0:
                sender(1)
            elif rank == 1:
                receiver(0)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
