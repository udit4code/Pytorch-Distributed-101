"""Reduction algorithms assembled from Phase 1 point-to-point operations."""

import argparse

import torch
import torch.distributed as dist

from .distributed import setup_process_group, cleanup_process_group, rank_record


def _log(enabled: bool, rank: int, message: str) -> None:
    """Print one flushed, rank-prefixed event during an interactive run."""
    if enabled:
        print(f"[rank {rank}] {message}", flush=True)


SUPPORTED_OPERATORS = ("SUM", "MAX", "MIN", "PRODUCT")


def _normalize_operator(operator: str) -> str:
    """Return a validated uppercase operator name."""
    normalized = operator.upper()
    if normalized not in SUPPORTED_OPERATORS:
        supported = ", ".join(SUPPORTED_OPERATORS)
        raise ValueError(f"operator={operator!r} must be one of: {supported}")
    return normalized


def _combine_(
    accumulator: torch.Tensor, incoming: torch.Tensor, operator: str
) -> None:
    """Apply one elementwise reduction into ``accumulator`` in place."""
    if operator == "SUM":
        accumulator.add_(incoming)
    elif operator == "PRODUCT":
        accumulator.mul_(incoming)
    elif operator == "MAX":
        accumulator.copy_(torch.maximum(accumulator, incoming))
    else:  # MIN; operator has already been validated.
        accumulator.copy_(torch.minimum(accumulator, incoming))

# The direct protocol under the hood is:
# 1. For every non-destination rank, we send its tensor to the destination.
# 2. For the destination-rank, we start with its own tensor,
# receive one tensor from every rank, and combine it with the accumulator using
# the selected elementwise operator.
def manual_reduce(
    tensor: torch.Tensor,
    dst: int,
    operator: str = "SUM",
    verbose: bool = False,
) -> torch.Tensor:
    """Reduce all world tensors at ``dst``; default to elementwise SUM.

    All ranks must pass the same operator, tensor shape, and tensor dtype.
    Supported operators are SUM, MAX, MIN, and PRODUCT. Only the destination's
    tensor is guaranteed to contain the final result.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    operator = _normalize_operator(operator)
    if not (0 <= dst < world_size):
        raise ValueError(f"dst={dst} must be in [0, {world_size})")

    _log(
        verbose,
        rank,
        f"started {operator}: local tensor={tensor.tolist()}, dst={dst}",
    )

    if rank == dst:
        # CASE 1: If the current rank is the destination
        # Start with a copy of the local tensor, then combine every other
        # rank's tensor using the selected operator. This is the accumulator.
        result = tensor.clone()
        _log(verbose, rank, f"initialized accumulator to {result.tolist()}")
        # Receive each peer's tensor into a separate buffer and accumulate it.
        recv_buffer = torch.empty_like(tensor)
        for source in range(world_size):
            if source != dst:
                _log(verbose, rank, f"waiting for rank {source}")
                dist.recv(tensor=recv_buffer, src=source)
                _log(verbose, rank, f"received {recv_buffer.tolist()} from rank {source}")
                _combine_(result, recv_buffer, operator)
                _log(
                    verbose,
                    rank,
                    f"after {operator}, accumulator is {result.tolist()}",
                )

        # Match reduce-like in-place semantics at the destination
        tensor.copy_(result)
        _log(verbose, rank, f"finished {operator} with result {tensor.tolist()}")
    else:
        # CASE 2: The current rank is not the destination.
        # So, the current rank sends its local tensor to the destination.
        # send does not allocate or return a new tensor. It copies this rank's
        # local payload into the destination's buffer.
        _log(verbose, rank, f"sending {tensor.tolist()} to rank {dst}")
        dist.send(tensor=tensor, dst=dst)
        _log(verbose, rank, f"send to rank {dst} completed")
    return tensor


# SUM dry run for eight ranks. Other associative operators use the same tree;
# only the elementwise combine operation changes.
#       0:36
#      /    \
#    0:10    4:26
#    /  \    /  \
#  0:3  2:7 4:11 6:15
#  / \  / \ / \  / \
# 0  1 2  3 4  5 6  7
def tree_reduce(
    tensor: torch.Tensor,
    dst: int = 0,
    operator: str = "SUM",
    verbose: bool = False,
) -> torch.Tensor:
    """Reduce tensors through a binary tree; default to elementwise SUM.

    This teaching implementation requires destination zero and a power-of-two
    world size. SUM, MAX, MIN, and PRODUCT are associative, so partial results
    may be combined in tree order without changing the mathematical result.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    operator = _normalize_operator(operator)

    if dst != 0:
        raise ValueError("This implementation currently mandates dst=0")
    if world_size <= 0 or (world_size & (world_size - 1)) != 0:
        raise ValueError(f"world_size={world_size} must be a power of two.")

    _log(
        verbose,
        rank,
        f"started tree {operator}: local tensor={tensor.tolist()}, dst={dst}",
    )

    step = 1
    round_number = 1
    while step < world_size:
        _log(
            verbose,
            rank,
            f"round {round_number}: step={step}, partial result={tensor.tolist()}",
        )
        # Ranks whose bit is set for the current step is set, become senders.
        # Example :
        # step 1 -> 1, 3, 5, 7 ranks send.
        # step 2 : 2, 6 ranks send.
        # step 4 : Rank 4 sends.
        if rank % (2 * step) == step:
            destination = rank - step
            _log(
                verbose,
                rank,
                f"round {round_number}: sending {tensor.tolist()} to rank {destination}",
            )
            dist.send(tensor=tensor, dst=destination)
            _log(verbose, rank, f"round {round_number}: send completed; exiting")
            # This rank's partial result now lives at `destination`, so sending
            # again in a later round would count those values twice.
            break
        else:
            # Surviving ranks receive from their tree partner.
            source = rank + step
            if source < world_size:
                result = tensor.clone()
                recv_buffer = torch.empty_like(tensor)
                _log(verbose, rank, f"round {round_number}: waiting for rank {source}")
                dist.recv(tensor=recv_buffer, src=source)
                _log(
                    verbose,
                    rank,
                    f"round {round_number}: received {recv_buffer.tolist()} from rank {source}",
                )
                _combine_(result, recv_buffer, operator)
                tensor.copy_(result)
                _log(
                    verbose,
                    rank,
                    f"round {round_number}: partial {operator} is {tensor.tolist()}",
                )

        step *= 2
        round_number += 1

    if rank == dst:
        _log(
            verbose,
            rank,
            f"tree root finished {operator} with result {tensor.tolist()}",
        )
    return tensor


def manual_reduce_sum(
    tensor: torch.Tensor, dst: int, verbose: bool = False
) -> torch.Tensor:
    """Compatibility wrapper for direct SUM reduction."""
    return manual_reduce(tensor, dst=dst, operator="SUM", verbose=verbose)


def tree_reduce_sum(
    tensor: torch.Tensor, dst: int = 0, verbose: bool = False
) -> torch.Tensor:
    """Compatibility wrapper for tree SUM reduction."""
    return tree_reduce(tensor, dst=dst, operator="SUM", verbose=verbose)


def run(
    algorithm: str = "manual",
    dst: int = 0,
    operator: str = "SUM",
    verbose: bool = True,
) -> None:
    """Connect torchrun workers, execute one reduction, and disconnect them.

    torchrun creates the Python processes and exports RANK, WORLD_SIZE,
    MASTER_ADDR, and MASTER_PORT. setup_process_group reads those variables and
    joins the existing processes into one Gloo group.
    """
    # Every worker must join the group before calling get_rank, send, or recv.
    setup_process_group()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _log(
            verbose,
            rank,
            f"process group ready: backend={dist.get_backend()}, "
            f"world_size={world_size}",
        )

        # Each process owns a separate tensor. Rank r contributes r + 1.
        tensor = torch.tensor([rank + 1], dtype=torch.int64)
        operator = _normalize_operator(operator)
        operation = f"{algorithm}_reduce"
        rank_record(
            rank,
            operation,
            tensor,
            phase="before",
            dst=dst,
            reduce_op=operator,
        )

        if algorithm == "tree":
            tree_reduce(tensor, dst=dst, operator=operator, verbose=verbose)
        else:
            manual_reduce(tensor, dst=dst, operator=operator, verbose=verbose)

        if rank == dst:
            rank_record(
                rank,
                operation,
                tensor,
                phase="after",
                dst=dst,
                reduce_op=operator,
            )
    finally:
        # Cleanup belongs to the runner, not the reusable reduction functions.
        cleanup_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce rank-local integers using point-to-point messages."
    )
    parser.add_argument(
        "--algorithm", choices=("manual", "tree"), default="manual"
    )
    parser.add_argument("--dst", type=int, default=0)
    parser.add_argument(
        "--operator",
        choices=SUPPORTED_OPERATORS,
        default="SUM",
        help="elementwise reduction operator (default: SUM)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide the detailed per-rank send/receive trace",
    )
    args = parser.parse_args()
    run(
        algorithm=args.algorithm,
        dst=args.dst,
        operator=args.operator,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
