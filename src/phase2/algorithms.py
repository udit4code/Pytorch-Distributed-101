"""Mini coordinator using only broadcast, barrier, and reduce."""
from __future__ import annotations
import json
import torch
import torch.distributed as dist


def run_capstone() -> dict[str, int | float] | None:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    config = torch.tensor([123, 5] if rank == 0 else [0, 0], dtype=torch.int64)
    # TODO: IMPLEMENT configuration broadcast.
    seed, steps = (int(v) for v in config.tolist())
    # Deterministic rank-varying toy workload; count is always positive.
    local_examples = rank + 1
    local_loss_sum = float((rank + 1) * steps)
    # TODO: IMPLEMENT barrier before aggregation.
    aggregate = torch.tensor([local_loss_sum, float(local_examples)], dtype=torch.float64)
    # TODO: IMPLEMENT reduction to rank 0 using SUM.
    if rank == 0:
        result: dict[str, int | float] = {
            "world_size": world_size,
            "total_examples": int(aggregate[1].item()),
            "global_mean_loss": aggregate[0].item() / aggregate[1].item(),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    return None


if __name__ == "__main__":
    from .distributed import setup_process_group, cleanup_process_group
    setup_process_group(timeout_seconds=30)
    try:
        run_capstone()
    finally:
        cleanup_process_group()
