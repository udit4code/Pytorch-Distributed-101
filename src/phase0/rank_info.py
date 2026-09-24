"""A small snapshot of this worker's process and distributed identity."""

from dataclasses import dataclass
import os

import torch.distributed as dist


@dataclass(frozen=True)
class RankInfo:
    rank: int
    local_rank: int
    world_size: int
    pid: int


def get_rank_info() -> RankInfo:
    """Read distributed identity and the torchrun local-rank environment."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before reading rank information")

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as exc:
        raise RuntimeError("LOCAL_RANK is required in the environment") from exc
    except ValueError as exc:
        raise RuntimeError("LOCAL_RANK must be an integer") from exc

    return RankInfo(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        pid=os.getpid(),
    )
