"""Subgroup collective exercises for world ranks {0,1} and {2,3}."""

import torch
import torch.distributed as dist

from .distributed import cleanup_process_group, rank_record, setup_process_group


PAIR_A = [0, 1]
PAIR_B = [2, 3]


def _log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def create_pair_groups() -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create groups {0,1} and {2,3} on every world rank.

    This exercise deliberately requires exactly four world ranks. Group
    creation is itself a distributed protocol: every world rank makes these
    calls in the same order, even though each rank later communicates through
    only one of the returned groups.
    """
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(
            f"pair-group demo requires world_size=4, received {world_size}"
        )

    # Keeping this order identical on all ranks ensures that each process maps
    # group_a and group_b to the same memberships and communication contexts.
    group_a = dist.new_group(ranks=PAIR_A)
    group_b = dist.new_group(ranks=PAIR_B)

    return group_a, group_b


def run_pair_broadcasts() -> torch.Tensor:
    """Broadcast a different value inside each disjoint pair."""
    rank = dist.get_rank()
    group_a, group_b = create_pair_groups()

    # Every rank belongs to exactly one pair. Collective membership, the source
    # rank, and the expected value are therefore local decisions derived from
    # the global rank after both groups have been created.
    if rank in PAIR_A:
        group_name = "A"
        members = PAIR_A
        group = group_a
        root = 0
        source_value = 100
    else:
        group_name = "B"
        members = PAIR_B
        group = group_b
        root = 2
        source_value = 200

    # The subgroup source owns meaningful input. Its partner allocates a tensor
    # with the same shape and dtype to serve as the receive buffer.
    tensor = torch.tensor(
        [source_value if rank == root else -1],
        dtype=torch.int64,
    )

    _log(
        rank,
        f"group {group_name} members={members}: before broadcast "
        f"tensor={tensor.tolist()}, global src={root}",
    )
    rank_record(
        rank,
        "subgroup_broadcast",
        tensor,
        phase="before",
        group=group_name,
        members=members,
        src=root,
    )

    # `src` is a GLOBAL rank. Passing `group` restricts participation to that
    # group's members: ranks 0/1 never enter group B's broadcast, and ranks 2/3
    # never enter group A's broadcast.
    dist.broadcast(tensor=tensor, src=root, group=group)

    _log(
        rank,
        f"group {group_name}: after broadcast tensor={tensor.tolist()}",
    )
    rank_record(
        rank,
        "subgroup_broadcast",
        tensor,
        phase="after",
        group=group_name,
        members=members,
        src=root,
    )
    return tensor


def main() -> None:
    """Join the default world, run pair broadcasts, and release resources."""
    # torchrun creates four processes. This call connects them into WORLD, the
    # default process group from which the two smaller groups are constructed.
    setup_process_group()
    try:
        rank = dist.get_rank()
        _log(
            rank,
            f"WORLD ready: world_size={dist.get_world_size()}, "
            f"backend={dist.get_backend()}",
        )
        run_pair_broadcasts()
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
