"""Construct ReduceScatter SUM from earlier reduction and P2P exercises."""
import torch


def manual_reduce_scatter_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Return this rank's shard of the elementwise sum using send/recv."""
    # TODO: IMPLEMENT (native ReduceScatter is not permitted here)
    raise NotImplementedError("implement point-to-point ReduceScatter SUM")
