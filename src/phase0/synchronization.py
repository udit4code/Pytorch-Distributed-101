"""Barrier exercise showing when all ranks have reached a point."""

import time

import torch.distributed as dist


def barrier_demo(delay_seconds: float = 0.0) -> None:
    """Sleep by a rank-dependent amount, log timestamps, and meet at a barrier."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before the barrier demo")

    rank = dist.get_rank()
    print(f"rank={rank} before sleep: {time.time():.6f}", flush=True)
    time.sleep(rank * delay_seconds)
    print(f"rank={rank} before barrier: {time.time():.6f}", flush=True)
    dist.barrier()
    print(f"rank={rank} after barrier: {time.time():.6f}", flush=True)
