"""Opt-in failure demonstrations; use process-group timeouts in subprocesses."""

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable

import torch
import torch.distributed as dist


def _log(**record: object) -> None:
    # One write keeps small JSON records intact when local worker processes
    # share stdout under torchrun (notably on macOS, where stream redirection
    # behavior can differ from Linux).
    payload = (json.dumps(record) + "\n").encode()
    os.write(sys.stdout.fileno(), payload)


def mismatched_collective_order() -> None:
    """Intentionally issue rank-dependent AllReduce order to cause a failure.

    Rank 0 reduces A then B while rank 1 reduces B then A.
    Gloo may report a shape mismatch or time out. Run only in a subprocess
    with a short process-group timeout.
    """
    rank = dist.get_rank()

    if dist.get_world_size() != 2:
        raise ValueError("This demonstration requires world_size=2")

    # Distinct shapes make the mismatch easier to diagnose.
    tensor_a = torch.tensor(
        [10.0 + rank],
        dtype=torch.float32,
    )
    tensor_b = torch.tensor(
        [20.0 + rank, 30.0 + rank],
        dtype=torch.float32,
    )

    if rank == 0:
        order = [
            ("A", tensor_a),
            ("B", tensor_b),
        ]
    else:
        order = [
            ("B", tensor_b),
            ("A", tensor_a),
        ]

    for step, (name, tensor) in enumerate(order):
        _log(
            rank=rank,
            pid=os.getpid(),
            # Use the CLI scenario name consistently in every record so
            # subprocess tests and operators can filter the complete trace.
            demo="mismatched_order",
            step=step,
            tensor=name,
            shape=list(tensor.shape),
            phase="before_all_reduce",
        )

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        # In a failing run we may never reach this record.
        _log(
            rank=rank,
            pid=os.getpid(),
            demo="mismatched_order",
            step=step,
            tensor=name,
            shape=list(tensor.shape),
            phase="after_all_reduce",
        )


def missing_rank() -> None:
    """Demonstrate one rank leaving before peers enter AllReduce.

    Run only under an externally bounded subprocess timeout.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        raise ValueError("This demonstration requires at least 2 ranks")

    tensor = torch.tensor(
        [float(rank + 1)],
        dtype=torch.float32,
    )

    # Let the final rank leave the function without participating.
    missing = world_size - 1

    if rank == missing:
        _log(
            rank=rank,
            pid=os.getpid(),
            demo="missing_rank",
            step=0,
            shape=list(tensor.shape),
            phase="exit_before_all_reduce",
        )
        return

    _log(
        rank=rank,
        pid=os.getpid(),
        demo="missing_rank",
        step=0,
        shape=list(tensor.shape),
        phase="before_all_reduce",
        missing_rank=missing,
    )

    # Peers wait here until Gloo detects the failed/missing participant
    # or the configured process-group timeout fires.
    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    _log(
        rank=rank,
        pid=os.getpid(),
        demo="missing_rank",
        step=0,
        shape=list(tensor.shape),
        phase="after_all_reduce",
        value=tensor.tolist(),
    )


def straggler_delay(seconds: float = 3.0) -> None:
    """Delay the final rank before AllReduce and report synchronization time."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"seconds must be finite and non-negative, got {seconds}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    straggler_rank = world_size - 1

    tensor = torch.tensor(
        [float(rank + 1)],
        dtype=torch.float32,
    )

    _log(
        rank=rank,
        pid=os.getpid(),
        demo="straggler_delay",
        step=0,
        shape=list(tensor.shape),
        phase="start",
        timestamp=time.monotonic(),
    )

    if rank == straggler_rank:
        _log(
            rank=rank,
            pid=os.getpid(),
            demo="straggler_delay",
            step=0,
            phase="sleep_start",
            delay_seconds=seconds,
            timestamp=time.monotonic(),
        )

        time.sleep(seconds)

        _log(
            rank=rank,
            pid=os.getpid(),
            demo="straggler_delay",
            step=0,
            phase="sleep_end",
            timestamp=time.monotonic(),
        )

    collective_start = time.monotonic()

    _log(
        rank=rank,
        pid=os.getpid(),
        demo="straggler_delay",
        step=0,
        shape=list(tensor.shape),
        phase="before_all_reduce",
        timestamp=collective_start,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    collective_end = time.monotonic()

    _log(
        rank=rank,
        pid=os.getpid(),
        demo="straggler_delay",
        step=0,
        shape=list(tensor.shape),
        phase="after_all_reduce",
        timestamp=collective_end,
        collective_seconds=collective_end - collective_start,
        value=tensor.tolist(),
    )


def run_scenario(
    scenario: str,
    *,
    seconds: float = 3.0,
    timeout_seconds: int = 10,
) -> None:
    """Run one scenario in an initialized Gloo group and always clean up.

    Mismatch and missing-rank scenarios are expected to fail. Run them only
    through this CLI or another externally bounded subprocess.
    """
    from .distributed import cleanup_process_group, setup_process_group

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    scenarios: dict[str, Callable[[], None]] = {
        "mismatched_order": mismatched_collective_order,
        "missing_rank": missing_rank,
        "straggler": lambda: straggler_delay(seconds),
    }
    if scenario not in scenarios:
        raise ValueError(f"unknown scenario: {scenario}")

    setup_process_group(timeout_seconds=timeout_seconds)
    try:
        _log(
            rank=dist.get_rank(),
            pid=os.getpid(),
            demo=scenario,
            phase="process_group_ready",
            world_size=dist.get_world_size(),
            timeout_seconds=timeout_seconds,
        )
        scenarios[scenario]()
    finally:
        cleanup_process_group()


def main() -> None:
    """Run one bounded failure or straggler demonstration under torchrun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("mismatched_order", "missing_rank", "straggler"),
        required=True,
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    args = parser.parse_args()
    run_scenario(
        args.scenario,
        seconds=args.seconds,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
