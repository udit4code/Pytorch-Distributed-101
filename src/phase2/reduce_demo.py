"""Native reduce demo, with deterministic operator examples."""
import argparse
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def run(dst: int = 0, operator: str = "SUM") -> None:
    setup_process_group()
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank + 1, (rank + 1) * 10], dtype=torch.int64)
        op = getattr(dist.ReduceOp, operator)
        rank_record(rank, "reduce", tensor, phase="before", dst=dst, reduce_op=operator)
        # TODO: IMPLEMENT dist.reduce(tensor, dst=dst, op=op).
        raise NotImplementedError
        if rank == dst:
            rank_record(rank, "reduce", tensor, phase="after", dst=dst, reduce_op=operator)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=int, default=0)
    parser.add_argument("--operator", choices=("SUM", "MAX", "MIN", "PRODUCT"), default="SUM")
    args = parser.parse_args()
    run(args.dst, args.operator)
