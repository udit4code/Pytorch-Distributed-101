"""Nonblocking communication exercises and per-rank event timing."""

import argparse
import json
import time

import torch
import torch.distributed as dist


def async_ring_exchange(value: torch.Tensor) -> torch.Tensor:
    """Send value clockwise and receive from the previous rank asynchronously.

    The returned tensor is local storage owned by this rank. The work handles
    represent communication that has been launched but may still be in flight.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size < 2:
        raise ValueError("async_ring_exchange requires at least 2 ranks")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    # Ranks do not share tensor objects or memory. irecv writes the incoming
    # bytes into a buffer allocated here, with shape and dtype compatible with
    # the tensor sent by prev_rank.
    received = torch.empty_like(value)

    # Start both sides of the protocol before waiting. isend/irecv return work
    # handles immediately, allowing this process to do independent work while
    # the transfers progress. Posting the receive first also makes the expected
    # incoming message explicit before the neighbor sends.
    recv_work = dist.irecv(tensor=received, src=prev_rank)
    send_work = dist.isend(tensor=value, dst=next_rank)

    # This tiny calculation is only a demonstration of work between launch and
    # completion; it makes no meaningful performance claim on a CPU backend.
    _ = sum(range(100))

    # Do not read the receive buffer until its receive has completed, and keep
    # the send tensor unchanged/alive until the send has completed.
    recv_work.wait()
    send_work.wait()
    return received


def compare_timestamps() -> dict[str, float]:
    """Compare local event timing for blocking and nonblocking ring exchange.

    Requires an initialized process group with at least two ranks. Values are
    from this process's monotonic clock and must not be compared across ranks.
    They illustrate execution order, not general communication performance.
    """
    from phase1.ring import ring_exchange

    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size < 2:
        raise ValueError("compare_timestamps requires at least 2 ranks")

    value = torch.tensor([rank], dtype=torch.int64)
    stamps: dict[str, float] = {}

    # A blocking call occupies this process until its send/receive protocol
    # completes, so the independent CPU work can only begin afterward.
    stamps["blocking_communication_start"] = time.perf_counter()
    ring_exchange(value)
    stamps["blocking_communication_complete"] = time.perf_counter()
    stamps["blocking_computation_start"] = time.perf_counter()
    _ = sum(range(100))
    stamps["blocking_computation_end"] = time.perf_counter()

    # Nonblocking calls return work handles before communication is complete.
    # The local receive buffer belongs to this process and is filled by irecv.
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    received = torch.empty_like(value)
    stamps["nonblocking_communication_start"] = time.perf_counter()
    recv_work = dist.irecv(tensor=received, src=prev_rank)
    send_work = dist.isend(tensor=value, dst=next_rank)

    stamps["nonblocking_computation_start"] = time.perf_counter()
    _ = sum(range(100))
    stamps["nonblocking_computation_end"] = time.perf_counter()
    stamps["nonblocking_wait_start"] = time.perf_counter()
    recv_work.wait()
    send_work.wait()
    stamps["nonblocking_communication_complete"] = time.perf_counter()
    return stamps


def main() -> None:
    """Launch a real multi-process ring exchange for the async exercise."""
    from phase1.distributed import cleanup_process_group, setup_process_group

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("exchange", "timings"), default="exchange")
    args = parser.parse_args()
    setup_process_group(timeout_seconds=15)
    rank = dist.get_rank()
    try:
        if args.scenario == "timings":
            stamps = compare_timestamps()
            print(json.dumps({"rank": rank, **stamps}), flush=True)
        else:
            value = torch.tensor([rank], dtype=torch.int64)
            received = async_ring_exchange(value)
            print(f"rank {rank} receives {received.item()}", flush=True)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
