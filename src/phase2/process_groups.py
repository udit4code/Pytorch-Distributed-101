"""Subgroup collective exercises for world ranks {0,1} and {2,3}."""
import torch
import torch.distributed as dist


def create_pair_groups() -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    # TODO: IMPLEMENT. Every world rank must create groups in the same order.
    raise NotImplementedError


def run_pair_broadcasts() -> None:
    rank = dist.get_rank()
    group_a, group_b = create_pair_groups()
    group = group_a if rank < 2 else group_b
    root = 0 if rank < 2 else 2
    tensor = torch.tensor([100 if rank == root else -1])
    # TODO: IMPLEMENT subgroup broadcast using global src and the matching group.
    raise NotImplementedError
