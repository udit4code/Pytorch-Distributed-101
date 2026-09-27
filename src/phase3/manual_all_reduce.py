"""Teaching implementations of AllReduce assembled from simpler primitives."""
import argparse

import torch
import torch.distributed as dist

from .distributed import cleanup_process_group, rank_record, setup_process_group


def naive_all_reduce_sum(tensor: torch.Tensor, root: int = 0) -> torch.Tensor:
    """Implement SUM AllReduce as Reduce followed by Broadcast.

    Naive implementation : Reduce SUM to ``root``, 
    then broadcast the result so every rank ends with
    the same summed tensor. Operation is in place.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Step 1 : Sum all the tensors at root 
    dist.reduce(tensor=tensor, dst=root, op=dist.ReduceOp.SUM)
    # Step 2 : Send root's reduced result back to every rank 
    dist.broadcast(tensor=tensor, src=root)
    
    return tensor


def point_to_point_all_reduce_sum(tensor: torch.Tensor, root: int = 0) -> torch.Tensor:
    """Implement SUM AllReduce using only blocking send and recv.

    Implementation : Uses a deadlock-safe two-phase protocol:
      1. gather + sum at root
      2. distribute final sum from root to all peers

    Preserves in-place semantics.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not 0 <= root < world_size:
        raise ValueError(
            f"root={root} must be in [0, {world_size})"
        )
    # No barrier is needed: the root receives every contribution before it
    # sends the result, and each peer waits for that result after sending.
    # These matching send/recv operations provide the required ordering.
    if rank == root: 
        # CASE 1 : When the current rank is the root itself
        # Start with the root's own contribution
        result = tensor.clone()
        recv_buffer = torch.empty_like(tensor)
        # Phase 1 : Gather and accumulate all peer contributions
        for src in range(world_size):
            if src != root:
                dist.recv(tensor=recv_buffer,src=src)
                result += recv_buffer
        # Write back result to tensor
        # Root's input tensor should contain the final sum too, so that it can be distributed to all other ranks 
        tensor.copy_(result)
        
        # Phase 2 : distribute the final sum to every peer/rank in the world 
        for dst in range(world_size):
            if dst != root:
                dist.send(tensor=tensor, dst=dst)
    else:
        # CASE 2 : When the current rank is NOT the root 
        # PHASE 1 : Send local contribution to root
        dist.send(tensor=tensor, dst=root)
        # PHASE 2 : Receive the final reduced resu;t back from the root.
        dist.recv(tensor=tensor, src=root)
    return tensor


def run_self_check() -> None:
    """Run both manual SUM algorithms and compare them with native AllReduce."""
    setup_process_group()
    try:
        rank = dist.get_rank()
        # Use multiple elements so the check covers elementwise tensor sums.
        initial = torch.tensor([rank + 1, (rank + 1) * 10], dtype=torch.int64)

        expected = initial.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        reduced_then_broadcast = naive_all_reduce_sum(initial.clone())
        point_to_point = point_to_point_all_reduce_sum(initial.clone())

        if not torch.equal(reduced_then_broadcast, expected):
            raise AssertionError(
                f"rank {rank}: Reduce+Broadcast gave {reduced_then_broadcast.tolist()}, "
                f"expected {expected.tolist()}"
            )
        if not torch.equal(point_to_point, expected):
            raise AssertionError(
                f"rank {rank}: send/recv gave {point_to_point.tolist()}, "
                f"expected {expected.tolist()}"
            )
        rank_record(rank, "manual_all_reduce_self_check", point_to_point, phase="passed")
    finally:
        cleanup_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the manual AllReduce exercises.")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="compare both manual algorithms with native SUM AllReduce",
    )
    args = parser.parse_args()
    if not args.self_check:
        parser.error("pass --self-check to run the four-rank verification")
    run_self_check()


if __name__ == "__main__":
    main()
