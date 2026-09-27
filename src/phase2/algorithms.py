"""Mini coordinator using only broadcast, barrier, and reduce.

Run this module with ``torchrun``. Each worker executes this same file, while
``dist.get_rank()`` gives each process its role in the distributed algorithm.
"""

from __future__ import annotations

import json

import torch
import torch.distributed as dist


def run_capstone() -> dict[str, int | float] | None:
    """Coordinate toy work and return the global metric on rank zero.

    The caller must initialize a process group first. Every rank must call this
    function because broadcast, barrier, and reduce are collective operations
    over the default WORLD process group.
    """

    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Only rank 0 owns the configuration initially. All other ranks allocate a
    # buffer with the same shape and dtype because broadcast writes into each
    # participant's existing tensor; it does not allocate the receive buffer.
    config = torch.tensor([123, 5] if rank == 0 else [0, 0], dtype=torch.int64)

    # One-to-many phase: every WORLD rank calls this with the same source.
    # After it returns, every local `config` contains [123, 5].
    dist.broadcast(tensor=config, src=0)

    # Convert the two scalar tensor values into ordinary Python integers for
    # local control flow. Rank 0 no longer has a special value here: all ranks
    # received the same seed and step count.
    seed, steps = (int(v) for v in config.tolist())

    # A real worker would use the shared seed and its rank to create a distinct,
    # reproducible random stream. The toy arithmetic below does not use random
    # numbers, but setting the seed demonstrates how the broadcast configuration
    # controls every worker without giving all ranks an identical random stream.
    torch.manual_seed(seed + rank)

    # Deterministic rank-varying toy workload. For four ranks, example counts
    # are 1, 2, 3, and 4. Each rank contributes `steps` loss per example, so its
    # loss sum is examples * 5. Counts are positive, making division safe.
    local_examples = rank + 1
    local_loss_sum = float((rank + 1) * steps)

    # Everyone-waits phase: no rank begins metric aggregation until every rank
    # has finished its local work. Reduce would already coordinate its own data
    # transfer, so this barrier is an explicit phase boundary for the exercise.
    # It also demonstrates that the fastest rank waits for the slowest rank.
    dist.barrier()

    # Pack both sufficient statistics into one tensor so one SUM reduction can
    # aggregate them. Reducing loss sums and counts produces the correct global
    # weighted mean even when ranks process different numbers of examples.
    aggregate = torch.tensor([local_loss_sum, float(local_examples)], dtype=torch.float64)

    # Many-to-one phase: every WORLD rank contributes its tensor, while only
    # rank 0 is guaranteed to hold the final SUM. Non-destination buffers must
    # not be treated as global results after a reduce.
    dist.reduce(tensor=aggregate, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        # For four ranks:
        #   total loss     = 5 + 10 + 15 + 20 = 50
        #   total examples = 1 +  2 +  3 +  4 = 10
        #   global mean    = 50 / 10 = 5.0
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

    # torchrun supplies RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT. This
    # initializes one Gloo WORLD process group from those environment values.
    setup_process_group(timeout_seconds=30)
    try:
        run_capstone()
    finally:
        # Every worker releases its local process-group resources, including
        # nonzero ranks whose run_capstone() return value is None.
        cleanup_process_group()
