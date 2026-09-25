"""Barrier timing and synchronization-versus-data experiment."""
import time
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def run() -> None:
    setup_process_group()
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank])
        before_sleep = time.time()
        time.sleep({0: 0, 1: 1, 2: 2, 3: 4}.get(rank, rank))
        before = time.time()
        rank_record(rank, "barrier", tensor, phase="before", timestamp_before_sleep=before_sleep)
        # TODO: IMPLEMENT dist.barrier().
        raise NotImplementedError
        after = time.time()
        rank_record(rank, "barrier", tensor, phase="after", wait_seconds=after-before)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    run()
