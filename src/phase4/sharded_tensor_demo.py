"""Local tensor-sharding and temporary parameter materialization exercises."""
import torch
import torch.distributed as dist
from .all_gather_tensor_demo import gather_into_tensor


def create_local_parameter_shard(global_parameter: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return this rank's equal contiguous shard of a 1-D logical parameter."""
    if global_parameter.ndim != 1 or world_size <= 0 or global_parameter.numel() % world_size:
        raise ValueError("parameter must be 1-D and evenly divisible by world_size")
    shard_size = global_parameter.numel() // world_size
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return global_parameter[rank * shard_size:(rank + 1) * shard_size].clone()


def materialize_parameter(local_shard: torch.Tensor) -> torch.Tensor:
    """Temporarily reconstruct the full parameter on every rank."""
    return gather_into_tensor(local_shard)


def main() -> None:
    if not dist.is_initialized():
        raise RuntimeError("launch with torchrun after initializing a process group")
    raise NotImplementedError("TODO: IMPLEMENT sharded parameter demo workflow")
