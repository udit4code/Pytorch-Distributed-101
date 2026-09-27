"""Gradient aggregation helpers for equal and unequal local batch sizes."""
import torch
import torch.distributed as dist


def average_gradient_(grad: torch.Tensor) -> None:
    """Replace a local gradient with the equal-weight mean across ranks.

    TODO: IMPLEMENT: SUM the tensor across the group and divide in-place by
    world size. Explain in your reasoning that this is sample-correct only
    when local mean gradients represent equally sized batches (or equivalent
    weighting).
    """
    # TODO: IMPLEMENT
    raise NotImplementedError


def weighted_average_gradient_(grad: torch.Tensor, local_sample_count: int) -> None:
    """Aggregate local mean gradients weighted by each rank's sample count.

    TODO: IMPLEMENT: validate a positive count; multiply the local mean
    gradient by its count, SUM gradients across ranks, SUM counts in an
    integer tensor on a compatible device, and divide by the global count.
    All ranks must participate even when their local counts differ.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError
