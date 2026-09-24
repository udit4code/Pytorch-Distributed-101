"""Nonblocking communication scaffolding."""

import time

import torch
import torch.distributed as dist


def async_ring_exchange(value: torch.Tensor) -> torch.Tensor:
    rank, size = dist.get_rank(), dist.get_world_size()
    next_rank, prev_rank = (rank + 1) % size, (rank - 1 + size) % size
    received = torch.empty_like(value)
    # TODO: IMPLEMENT: launch isend/irecv, do trivial independent CPU work, wait.
    send_work = None
    recv_work = None
    _ = sum(range(100))
    if send_work is not None:
        send_work.wait()
    if recv_work is not None:
        recv_work.wait()
    return received


def compare_timestamps() -> dict[str, float]:
    """Timestamp exercise; timings describe this run, not general performance."""
    stamps: dict[str, float] = {"communication_start": time.perf_counter()}
    # TODO: IMPLEMENT blocking and nonblocking variants and record all requested events.
    stamps["computation_start"] = time.perf_counter()
    _ = sum(range(100))
    stamps["computation_end"] = time.perf_counter()
    stamps["wait_start"] = time.perf_counter()
    stamps["communication_complete"] = time.perf_counter()
    return stamps
