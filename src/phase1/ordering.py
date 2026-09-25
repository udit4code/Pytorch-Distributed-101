"""Message-order and optional tag experiments."""

import argparse

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group


def disagreeing_order(*, receive_b_first: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Send A then B, optionally receiving into B's buffer first.

    The return value is always ``(A_buffer, B_buffer)``. Rank 0 sends two
    same-shaped tensors in order. Rank 1's choice of Python variable names does
    not identify a message: with no tags, the first matching receive gets the
    first message from rank 0, and the second receive gets the second message.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError(f"disagreeing_order requires exactly 2 ranks; got {world_size}")

    # Equal shape and dtype keep this experiment focused on message order.
    a, b = torch.tensor([1]), torch.tensor([2])
    a_buffer, b_buffer = torch.empty_like(a), torch.empty_like(b)
    if rank == 0:
        # Rank 0 sends tensor bytes to rank 1 in this order. The names A and B
        # exist only in this Python process; they are not sent as labels.
        dist.send(tensor=a, dst=1)
        dist.send(tensor=b, dst=1)
    elif rank == 1:
        if receive_b_first:
            # Despite receiving into b_buffer first, this receives A's payload,
            # because A is the first message sent by rank 0.
            dist.recv(tensor=b_buffer, src=0)
            dist.recv(tensor=a_buffer, src=0)
        else:
            dist.recv(tensor=a_buffer, src=0)
            dist.recv(tensor=b_buffer, src=0)
    return a_buffer, b_buffer


def tagged_messages() -> tuple[torch.Tensor, torch.Tensor]:
    """Send two tagged messages and receive them in the opposite tag order.

    The return value is always ``(A_buffer, B_buffer)``. The tag is a small
    matching label in the communication protocol; it does not carry or infer
    tensor names, shapes, or dtypes. This experiment requires a backend that
    supports point-to-point tags (Gloo does in supported PyTorch builds).
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise ValueError(f"tagged_messages requires exactly 2 ranks; got {world_size}")

    a = torch.tensor([10], dtype=torch.int64)
    b = torch.tensor([20], dtype=torch.int64)
    a_buffer, b_buffer = torch.empty_like(a), torch.empty_like(b)

    if rank == 0:
        # Launch both sends so rank 0 does not block on tag 10 while rank 1 is
        # waiting for tag 20. Keep each source tensor alive until its work ends.
        send_a = dist.isend(tensor=a, dst=1, tag=10)
        send_b = dist.isend(tensor=b, dst=1, tag=20)
        send_a.wait()
        send_b.wait()
    else:
        # Although A was launched first, the tag lets this receive match B.
        # The second receive then matches A. Each destination buffer remains
        # independently allocated and must have a compatible tensor contract.
        dist.recv(tensor=b_buffer, src=0, tag=20)
        dist.recv(tensor=a_buffer, src=0, tag=10)

    return a_buffer, b_buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receive-order",
        choices=("A-first", "B-first"),
        default="B-first",
        help="which named buffer rank 1 fills first",
    )
    parser.add_argument("--scenario", choices=("ordering", "tags"), default="ordering")
    args = parser.parse_args()

    setup_process_group(timeout_seconds=10)
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != 2:
            raise ValueError(f"ordering experiment requires exactly 2 ranks; got {world_size}")
        if args.scenario == "tags":
            a_buffer, b_buffer = tagged_messages()
        else:
            a_buffer, b_buffer = disagreeing_order(
                receive_b_first=(args.receive_order == "B-first")
            )
        if rank == 1:
            print(
                f"scenario={args.scenario} receive order={args.receive_order}: "
                f"A_buffer={a_buffer.tolist()}, B_buffer={b_buffer.tolist()}",
                flush=True,
            )
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
