"""ReduceScatter SUM exercise with equal contiguous shards."""
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def reduce_scatter_sum(local_input: torch.Tensor) -> torch.Tensor:
    """Sum corresponding input elements and retain this rank's shard."""
    if local_input.ndim != 1:
        raise ValueError("exercise expects a one-dimensional tensor")
    world = dist.get_world_size()
    if local_input.numel() % world:
        raise ValueError("input element count must be divisible by world size")
    output = torch.empty(local_input.numel() // world, dtype=local_input.dtype)
    # TODO: IMPLEMENT
    return output


def main() -> None:
    setup_process_group()
    try:
        rank = dist.get_rank()
        x = torch.tensor([1, 2, 3, 4], dtype=torch.int64) * (10 ** rank)
        rank_record(rank, "reduce_scatter_sum", reduce_scatter_sum(x))
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
