"""Bounded, intentionally invalid protocol exercises; run only as subprocesses."""
from __future__ import annotations
import argparse
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group


SCENARIOS = ("inconsistent-size", "inconsistent-dtype", "wrong-order", "missing-rank", "wrong-output-shape", "not-divisible")


def run_scenario(name: str) -> None:
    """Run one deliberately invalid collective scenario under a group timeout."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
    # TODO: IMPLEMENT intentionally invalid per-rank inputs/order, safely isolated
    raise NotImplementedError(f"implement failure scenario: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=SCENARIOS)
    args = parser.parse_args()
    setup_process_group(timeout_seconds=8)
    try:
        run_scenario(args.scenario)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
