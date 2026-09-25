"""Native broadcast demo; launch with torchrun -m phase2.broadcast_demo."""
import argparse
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def run(src: int = 0, delay_rank: int = -1, delay_seconds: float = 0.0) -> None:
    import time
    setup_process_group()
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([(999 if src == 2 else 100) if rank == src else -1], dtype=torch.int64)
        if rank == delay_rank:
            time.sleep(delay_seconds)
        rank_record(rank, "broadcast", tensor, phase="before", src=src)
        # TODO: IMPLEMENT: every rank in the group calls dist.broadcast(tensor, src=src).
        raise NotImplementedError
        rank_record(rank, "broadcast", tensor, phase="after", src=src)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    run(args.src, args.delay_rank, args.delay_seconds)
