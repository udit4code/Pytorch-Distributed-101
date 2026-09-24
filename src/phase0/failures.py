"""Exercises for incompatible collectives and process-group cleanup."""

from datetime import timedelta

import torch
import torch.distributed as dist


def mismatched_collective_demo() -> None:
    """Explore what happens when ranks call incompatible collectives.

    Run only as a multi-process experiment: incompatible collective sequences
    can hang or time out. Keep the timeout short when you implement this.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before this demo")
    if dist.get_world_size() < 2:
        raise RuntimeError("Run this experiment with at least two workers")

    rank = dist.get_rank()
    if rank == 0:
        work = dist.barrier(async_op=True)
    else:
        work = dist.all_reduce(torch.ones(1), async_op=True)

    try:
        work.wait(timeout=timedelta(seconds=5))
    except RuntimeError as exc:
        print(f"rank={rank} observed incompatible collectives: {exc}", flush=True)
    else:
        print(f"rank={rank} collective completed despite the mismatch", flush=True)
    print(
        "Every rank must execute compatible collectives in the same order.",
        flush=True,
    )


def cleanup_demo() -> None:
    """Exercise cleanup after successful work and while handling an exception."""
    from phase0.distributed import cleanup_process_group, setup_process_group

    try:
        setup_process_group()
        dist.barrier()
    finally:
        cleanup_process_group()

    if dist.is_initialized():
        raise RuntimeError("Process-group cleanup did not complete")
    print("Process group cleaned up successfully.", flush=True)
