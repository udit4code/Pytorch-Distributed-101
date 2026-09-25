"""Ring exchange and manual ring-gather exercises."""

import torch
import torch.distributed as dist


def ring_exchange(value: torch.Tensor) -> torch.Tensor:
    """Send value clockwise and receive the previous rank's value.

    This uses one matched send/recv pair per rank. Rank 0 starts the blocking
    protocol; the other ranks receive before forwarding their own tensor. That
    ordering lets the message move through the ring without a collective or a
    circular wait, including when the ring has an odd number of ranks.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size < 2:
        raise ValueError("ring_exchange requires at least 2 ranks")

    # Ranks are separate processes with separate memory. This buffer belongs to
    # this rank; recv copies the previous rank's tensor data into it.
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    received = torch.empty_like(value)

    # A blocking send can wait until the destination posts a matching recv.
    # Rank 0 opens the chain by sending to rank 1. Every other rank first waits
    # for its predecessor, then sends its own original value to its successor.
    # The last rank's send matches rank 0's recv and closes the ring.
    if rank == 0:
        dist.send(tensor=value, dst=next_rank)
        dist.recv(tensor=received, src=prev_rank)
    else:
        dist.recv(tensor=received, src=prev_rank)
        dist.send(tensor=value, dst=next_rank)

    return received


def broken_ring_exchange(value: torch.Tensor) -> torch.Tensor:
    """Send first on every rank, exposing a possible circular wait.

    Whether this small-message example visibly hangs depends on the backend's
    buffering behavior. Run it only with an external timeout.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size < 2:
        raise ValueError("broken_ring_exchange requires at least 2 ranks")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    received = torch.empty_like(value)

    # Each process owns its own `value`; send copies its data toward next_rank.
    # With this ordering, every rank may block in send while its matching recv
    # is still waiting to run: the protocol has a circular wait around the ring.
    dist.send(tensor=value, dst=next_rank)
    dist.recv(tensor=received, src=prev_rank)
    return received


def safe_ring_exchange(value: torch.Tensor) -> torch.Tensor:
    """Exchange with a deterministic order that breaks the ring wait cycle.

    Rank 0 starts the send chain. Other ranks first receive from their
    predecessor, then send their own value onward. This works for odd and even
    world sizes and uses only blocking point-to-point operations.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size < 2:
        raise ValueError("safe_ring_exchange requires at least 2 ranks")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    received = torch.empty_like(value)

    # The buffer is local storage for data arriving from prev_rank. Rank 0
    # sends first to rank 1; each later rank is already waiting in recv when
    # its predecessor sends. The last rank's send matches rank 0's recv.
    if rank == 0:
        dist.send(tensor=value, dst=next_rank)
        dist.recv(tensor=received, src=prev_rank)
    else:
        dist.recv(tensor=received, src=prev_rank)
        dist.send(tensor=value, dst=next_rank)
    return received


def circulate(value: torch.Tensor, steps: int | None = None) -> list[int]:
    """Forward the latest received scalar around the ring and record each step.

    Every process starts with its own value. On each step, the value moves one
    rank clockwise; the process then forwards that newly received value on the
    next step. With four ranks and three steps, rank 0 observes 0, 3, 2, 1.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("circulate requires at least 2 ranks")
    if value.numel() != 1:
        raise ValueError("circulate expects a tensor with exactly one element")
    if steps is None:
        steps = world_size - 1
    if steps < 0:
        raise ValueError("steps must be non-negative")

    seen_values = [int(value.item())]
    current = value
    for _ in range(steps):
        # Each rank sends its current value to its next neighbor and receives
        # into separate local storage from its previous neighbor. Replacing
        # `current` is what makes the next iteration forward the new value.
        current = ring_exchange(current)
        seen_values.append(int(current.item()))
    return seen_values


def ring_gather(local_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Collect one same-shaped tensor from every rank by circulating shards.

    The result is ordered by source rank. This is built from matched point-to-
    point sends and receives; no collective operation is used.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size < 2:
        raise ValueError("ring_gather requires at least 2 ranks")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    # Every process has separate memory. Keep one local slot per source rank;
    # local_tensor is already the shard whose source index is this rank.
    # All ranks must agree on shape and dtype because P2P sends tensor payloads
    # into buffers that the receiver allocates; metadata is not negotiated.
    local_shard = local_tensor.contiguous()
    gathered = [torch.empty_like(local_shard) for _ in range(world_size)]
    gathered[rank] = local_shard.clone()

    # On round 0, rank r sends its own shard and receives rank r-1's shard.
    # On each later round it forwards the shard received in the previous round.
    # After world_size - 1 rounds, each rank has every source's shard.
    for round_index in range(world_size - 1):
        send_index = (rank - round_index) % world_size
        receive_index = (rank - round_index - 1) % world_size
        incoming = torch.empty_like(local_shard)

        # Post both operations before waiting. If every rank used blocking
        # send-first, all could wait for a receive no rank has posted yet.
        # These work handles represent operations in flight; wait before using
        # the received buffer or reusing the sent tensor in another round.
        receive_work = dist.irecv(tensor=incoming, src=prev_rank)
        send_work = dist.isend(tensor=gathered[send_index], dst=next_rank)
        receive_work.wait()
        send_work.wait()

        # Store by the original source rank, not by the order this rank saw it.
        gathered[receive_index] = incoming

    return gathered


def main() -> None:
    import argparse
    from phase1.distributed import cleanup_process_group, setup_process_group, trace_event

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("exchange", "gather", "broken", "safe", "circulate"),
        default="exchange",
    )
    args = parser.parse_args()
    setup_process_group()
    rank = dist.get_rank()
    try:
        value = torch.tensor([rank], dtype=torch.int64)
        if args.scenario == "gather":
            gathered = ring_gather(value)
            print(f"rank {rank} gathered {[int(item.item()) for item in gathered]}", flush=True)
        elif args.scenario == "circulate":
            seen = circulate(value)
            print(f"rank {rank} saw values {seen}", flush=True)
        else:
            exchange = {
                "exchange": ring_exchange,
                "broken": broken_ring_exchange,
                "safe": safe_ring_exchange,
            }[args.scenario]
            received = exchange(value)
            trace_event(rank=rank, operation="recv", peer=(rank - 1 + dist.get_world_size()) % dist.get_world_size(), tensor=received)
            print(f"rank {rank} receives {received.item()}", flush=True)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
