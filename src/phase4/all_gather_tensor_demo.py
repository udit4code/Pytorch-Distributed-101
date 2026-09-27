"""Contiguous and multidimensional AllGather output-shape exercise."""
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def gather_into_tensor(local_tensor: torch.Tensor) -> torch.Tensor:
    """Gather along dimension zero into a correctly sized contiguous buffer."""
    if local_tensor.ndim < 1:
        raise ValueError("local_tensor must have at least one dimension")
    output = torch.empty((dist.get_world_size() * local_tensor.shape[0], *local_tensor.shape[1:]), dtype=local_tensor.dtype)
    # TODO: IMPLEMENT
    return output


def main() -> None:
    setup_process_group()
    try:
        rank = dist.get_rank()
        local = torch.arange(rank * 2, rank * 2 + 2, dtype=torch.int64)
        rank_record(rank, "all_gather_into_tensor", gather_into_tensor(local))
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
