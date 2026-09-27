"""Asynchronous collective semantics, without performance claims."""
import torch
import torch.distributed as dist


def async_sum_(tensor: torch.Tensor) -> dist.Work:
    """Launch an asynchronous SUM AllReduce and return its Work handle.

    TODO: IMPLEMENT: call all_reduce with SUM and async_op=True. The caller
    must wait before reading the result; do not claim localhost overlap gains.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError


def run() -> None:
    """Demonstrate launch, independent CPU work, wait, and safe consumption.

    TODO: IMPLEMENT: initialize/cleanup the process group; launch async_sum_,
    perform deterministic independent CPU work, wait, then log the result.
    Keep tensor access after wait so the ordering lesson is explicit.
    """
    raise NotImplementedError
