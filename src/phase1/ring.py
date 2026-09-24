"""Ring exchange and manual ring-gather exercises."""

import torch
import torch.distributed as dist


def ring_exchange(value: torch.Tensor) -> torch.Tensor:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    received = torch.empty_like(value)
    # TODO: IMPLEMENT: exchange with prev_rank/next_rank without a collective.
    return received


def broken_ring_exchange(value: torch.Tensor) -> torch.Tensor:
    """Intentionally unsafe send-first experiment; run only with a timeout."""
    # TODO: IMPLEMENT: every rank sends first, then receives. Observe behavior.
    return torch.empty_like(value)


def safe_ring_exchange(value: torch.Tensor) -> torch.Tensor:
    """Choose and document a deterministic order that breaks the wait cycle."""
    # TODO: IMPLEMENT: reason about even/odd ranks and matching peers.
    return torch.empty_like(value)


def circulate(value: torch.Tensor, steps: int | None = None) -> list[int]:
    """Forward the most recently received value and return local observations."""
    if steps is None:
        steps = dist.get_world_size() - 1
    seen_values = [int(value.item())]
    current = value
    # TODO: IMPLEMENT: repeat a safe neighbor exchange and append each result.
    return seen_values


def ring_gather(local_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Capstone: circulate tensor payloads until every rank has every shard."""
    # TODO: IMPLEMENT using only send/recv/isend/irecv.
    return [local_tensor]


def main() -> None:
    import argparse
    from phase1.distributed import cleanup_process_group, setup_process_group, trace_event

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("exchange", "gather"), default="exchange")
    args = parser.parse_args()
    setup_process_group()
    rank = dist.get_rank()
    try:
        value = torch.tensor([rank], dtype=torch.int64)
        if args.scenario == "gather":
            gathered = ring_gather(value)
            print(f"rank {rank} gathered {[int(item.item()) for item in gathered]}", flush=True)
        else:
            received = ring_exchange(value)
            trace_event(rank=rank, operation="recv", peer=(rank - 1 + dist.get_world_size()) % dist.get_world_size(), tensor=received)
            print(f"rank {rank} receives {received.item()}", flush=True)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
