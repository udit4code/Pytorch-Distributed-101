"""Exercise native AllReduce with deterministic rank-local tensors."""
import torch
import torch.distributed as dist

from .distributed import cleanup_process_group, rank_record, setup_process_group


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across the default group; each rank must receive the sum.

    TODO: IMPLEMENT: invoke the SUM AllReduce in place, preserving dtype and
    shape. All ranks must call this in the same collective order.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError


def run() -> None:
    """Run the four-rank [rank+1, 10*(rank+1)] demonstration."""
    setup_process_group()
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank + 1, (rank + 1) * 10], dtype=torch.int64)
        rank_record(rank, "all_reduce", tensor, phase="before")
        all_reduce_sum_(tensor)
        rank_record(rank, "all_reduce", tensor, phase="after")
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    run()
