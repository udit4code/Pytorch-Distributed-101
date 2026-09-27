"""Gradient aggregation helpers for equal and unequal local batch sizes."""
import torch
import torch.distributed as dist


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
