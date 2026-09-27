"""Gradient aggregation helpers for equal and unequal local batch sizes."""
import argparse

import torch
import torch.distributed as dist

from .distributed import cleanup_process_group, emit, setup_process_group


def average_gradient_(grad: torch.Tensor) -> None:
    """Replace a local gradient with the equal-weight mean across ranks.

    This is sample-correct when every rank's local gradient is a mean over the
    same number of examples. The operation is in place; all ranks must call it
    with matching tensor shapes and in the same collective order.
    """
    world_size = dist.get_world_size()
    # After this SUM, every rank has the sum of the rank-local gradients.
    dist.all_reduce(tensor=grad, op=dist.ReduceOp.SUM)
    # Equal-sized local batches contribute equally to the global sample mean.
    grad.div_(world_size)


def weighted_average_gradient_(
    grad: torch.Tensor,
    local_sample_count: int,
) -> None:
    """Aggregate local mean gradients with each example weighted equally.

    A rank with no examples must provide a zero gradient and a count of zero.
    The total sample count across the process group must be positive.
    """

    if local_sample_count < 0:
        raise ValueError(
            f"local_sample_count must be non-negative, got {local_sample_count}"
        )

    # A local mean gradient gives each local example weight
    # 1 / local_sample_count. Multiply by the count to recover the local sum.
    # An empty rank has no local mean, so explicitly contribute a zero tensor.
    if local_sample_count == 0:
        grad.zero_()
    else:
        grad.mul_(local_sample_count)

    # Every rank contributes its local gradient sum. All ranks must execute
    # this collective in the same order and with matching gradient shapes.
    dist.all_reduce(
        grad,
        op=dist.ReduceOp.SUM,
    )

    # Aggregate the scalar denominator separately so unequal shard sizes are
    # weighted by examples rather than by rank.
    count = torch.tensor(
        [local_sample_count],
        dtype=torch.int64,
        device=grad.device,
    )

    dist.all_reduce(
        count,
        op=dist.ReduceOp.SUM,
    )

    global_sample_count = count.item()
    if global_sample_count == 0:
        raise ValueError("the global sample count must be positive")

    # Divide the summed per-example gradients by the total number of examples.
    grad.div_(global_sample_count)


def run_self_check() -> None:
    """Verify equal and sample-weighted averages across four real Gloo ranks."""
    setup_process_group()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Local means 1, 2, 3, 4 should average to 2.5 across four ranks.
        equal_grad = torch.tensor([float(rank + 1)], dtype=torch.float64)
        average_gradient_(equal_grad)
        expected_equal = 2.5
        if not torch.allclose(equal_grad, torch.tensor([expected_equal], dtype=equal_grad.dtype)):
            raise AssertionError(
                f"rank {rank}: equal average was {equal_grad.tolist()}, "
                f"expected [{expected_equal}]"
            )

        # Counts are 0, 1, 2, 3 and local means are 0, 2, 4, 6. Rank 0's
        # NaN placeholder verifies that an empty rank is explicitly zeroed.
        sample_count = rank
        weighted_grad = torch.tensor(
            [float("nan") if sample_count == 0 else float(2 * rank)],
            dtype=torch.float64,
        )
        weighted_average_gradient_(weighted_grad, sample_count)
        expected_weighted = 14.0 / 3.0
        if not torch.allclose(
            weighted_grad,
            torch.tensor([expected_weighted], dtype=weighted_grad.dtype),
            atol=1e-12,
            rtol=0.0,
        ):
            raise AssertionError(
                f"rank {rank}: weighted average was {weighted_grad.tolist()}, "
                f"expected [{expected_weighted}]"
            )

        emit({
            "rank": rank,
            "operation": "gradient_sync_self_check",
            "world_size": world_size,
            "equal_average": equal_grad.tolist(),
            "weighted_average": weighted_grad.tolist(),
            "phase": "passed",
        })
    finally:
        cleanup_process_group()


def main() -> None:
    """Expose the bounded multi-rank correctness check through torchrun."""
    parser = argparse.ArgumentParser(description="Check distributed gradient averaging.")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify equal and weighted gradient averaging across ranks",
    )
    args = parser.parse_args()
    if not args.self_check:
        parser.error("pass --self-check to run the multi-rank verification")
    run_self_check()


if __name__ == "__main__":
    main()
