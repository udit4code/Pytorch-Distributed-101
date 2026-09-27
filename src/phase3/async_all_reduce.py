"""Asynchronous collective semantics, without performance claims."""

import torch
import torch.distributed as dist


def async_sum_(tensor: torch.Tensor) -> dist.Work:
    """Launch an asynchronous SUM AllReduce and return its Work handle.

    The call returns a handle without waiting for the collective to finish.
    Keep the tensor alive and do not read from or modify it until ``wait()``
    completes. All ranks must still launch matching collectives in the same
    order.
    """
    return dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
        async_op=True,
    )


def run() -> None:
    """Demonstrate launch, independent CPU work, wait, and safe consumption."""
    dist.init_process_group(backend="gloo")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        tensor = torch.tensor(
            [rank + 1],
            dtype=torch.float32,
        )

        # Start the AllReduce. Unlike the default blocking call, this returns
        # a handle to Python without waiting for the result to be ready.
        work = async_sum_(tensor)

        # This CPU work is independent: it neither reads nor changes the tensor
        # participating in the collective, so it is safe to do before wait().
        cpu_result = sum(i * i for i in range(1000))

        # wait() blocks this process until this collective has completed for
        # this rank. It is not a replacement for matching collective order.
        work.wait()

        # Safe to consume the reduced tensor now; check the expected rank sum.
        expected_sum = sum(range(1, world_size + 1))
        actual_sum = tensor.item()
        if actual_sum != expected_sum:
            raise AssertionError(
                f"rank {rank}: AllReduce returned {actual_sum}, expected {expected_sum}"
            )

        print(
            f"rank={rank}, "
            f"cpu_result={cpu_result}, "
            f"sum={tensor.item()}",
            flush=True,
        )

    finally:
        dist.destroy_process_group()
