"""Bounded failure experiments. Use torchrun and a finite process-group timeout."""

import argparse

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group, trace_event

SCENARIOS = ("receiver_never_receives", "sender_never_sends", "wrong_source", "shape_mismatch", "dtype_mismatch", "rank_exits_early", "circular_wait")


def run_scenario(scenario: str) -> None:
    rank, size = dist.get_rank(), dist.get_world_size()
    peer = (rank + 1) % size
    tensor = torch.ones(2, dtype=torch.float32)
    trace_event(rank=rank, operation=scenario, peer=peer, tensor=tensor)
    # TODO: IMPLEMENT each deliberately invalid protocol. Keep it bounded by
    # setup_process_group's timeout and run subprocess experiments with a timeout.
    raise NotImplementedError(f"TODO: IMPLEMENT scenario {scenario}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    args = parser.parse_args()
    setup_process_group(timeout_seconds=10)
    try:
        run_scenario(args.scenario)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
