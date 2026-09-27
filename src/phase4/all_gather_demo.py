"""Basic rank-ordered AllGather exercise."""
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def gather_rank_values(local_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather equal-shaped tensors in rank order; implement the collective."""
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    # TODO: IMPLEMENT
    return gathered


def main() -> None:
    setup_process_group()
    try:
        rank = dist.get_rank()
        local = torch.tensor([10 * (rank + 1)], dtype=torch.int64)
        result = gather_rank_values(local)
        rank_record(rank, "all_gather", torch.cat(result))
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
