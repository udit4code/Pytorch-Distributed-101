"""Compare direct and tree point-to-point broadcast algorithms.

This is an educational local benchmark. It measures communication after the
process group has been initialized, but localhost results do not predict the
performance of a multi-node GPU cluster or optimized native collectives.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
import torch.distributed as dist

from .distributed import cleanup_process_group, setup_process_group
from .manual_broadcast import manual_broadcast, tree_broadcast

BroadcastAlgorithm = Callable[[torch.Tensor, int], torch.Tensor]


def _reset_tensor(tensor: torch.Tensor, rank: int, src: int, value: float) -> None:
    """Restore source data and receiver sentinels before each independent run."""
    tensor.fill_(value if rank == src else -1.0)


def _run_trial(
    algorithm: BroadcastAlgorithm,
    tensor: torch.Tensor,
    rank: int,
    src: int,
    value: float,
) -> float:
    """Return the slowest rank's elapsed time for one broadcast in seconds."""
    _reset_tensor(tensor, rank, src, value)

    # Align the start of the trial. This barrier is outside the timed region.
    dist.barrier()
    started = time.perf_counter()
    algorithm(tensor, src)
    local_elapsed = time.perf_counter() - started

    if not torch.all(tensor == value):
        raise RuntimeError(f"rank {rank} received an incorrect broadcast value")

    # Different ranks can finish at different times. The job completes only as
    # fast as its slowest participant, so rank zero records the maximum. This
    # reduction happens after each rank captures its local operation time.
    elapsed = torch.tensor([local_elapsed], dtype=torch.float64)
    dist.reduce(elapsed, dst=0, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def benchmark(
    name: str,
    algorithm: BroadcastAlgorithm,
    tensor: torch.Tensor,
    src: int,
    value: float,
    warmups: int,
    trials: int,
) -> dict[str, object] | None:
    rank = dist.get_rank()

    # Warmups absorb one-time backend and memory effects. Their timings are not
    # included in the reported samples.
    for _ in range(warmups):
        _run_trial(algorithm, tensor, rank, src, value)

    samples_ms = [
        _run_trial(algorithm, tensor, rank, src, value) * 1_000
        for _ in range(trials)
    ]
    if rank != 0:
        return None

    return {
        "algorithm": name,
        "world_size": dist.get_world_size(),
        "tensor_elements": tensor.numel(),
        "tensor_bytes": tensor.numel() * tensor.element_size(),
        "warmups": warmups,
        "trials": trials,
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark direct fan-out against binary-tree broadcast."
    )
    parser.add_argument(
        "--algorithm",
        choices=("manual", "tree", "both"),
        default="both",
    )
    parser.add_argument(
        "--order",
        choices=("manual-first", "tree-first"),
        default="manual-first",
        help="execution order when --algorithm=both",
    )
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--value", type=float, default=42.0)
    args = parser.parse_args()

    if args.elements <= 0:
        parser.error("--elements must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.trials <= 0:
        parser.error("--trials must be positive")

    setup_process_group()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.empty(args.elements, dtype=torch.float32)

        algorithms = {
            "manual": manual_broadcast,
            "tree": tree_broadcast,
        }
        if args.algorithm in ("tree", "both"):
            if world_size & (world_size - 1) != 0:
                raise ValueError("tree benchmark requires a power-of-two world size")

        if args.algorithm == "both":
            names = (
                ("manual", "tree")
                if args.order == "manual-first"
                else ("tree", "manual")
            )
        else:
            names = (args.algorithm,)
        selected = [(name, algorithms[name]) for name in names]

        results = []
        for name, algorithm in selected:
            result = benchmark(
                name,
                algorithm,
                tensor,
                src=0,
                value=args.value,
                warmups=args.warmups,
                trials=args.trials,
            )
            if result is not None:
                results.append(result)

        if rank == 0:
            print(json.dumps({"results": results}, indent=2), flush=True)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
