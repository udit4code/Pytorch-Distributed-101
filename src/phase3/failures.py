"""Opt-in failure demonstrations; use process-group timeouts in subprocesses."""
import torch
import torch.distributed as dist


def mismatched_collective_order() -> None:
    """Intentionally issue rank-dependent AllReduce order to trigger timeout.

    TODO: IMPLEMENT: rank 0 reduces A then B while rank 1 reduces B then A;
    attach distinct values/shapes for diagnosis and rely on a short configured
    Gloo timeout. Never call this from the regular unit-test process.
    """
    raise NotImplementedError


def missing_rank() -> None:
    """Demonstrate one rank exiting before peers enter AllReduce.

    TODO: IMPLEMENT: make one rank exit and others enter a collective, with a
    short process-group timeout and logs containing rank, PID, shape, and step.
    Run only under an externally bounded subprocess timeout.
    """
    raise NotImplementedError


def straggler_delay(seconds: float = 3.0) -> None:
    """Delay a selected rank before AllReduce and report synchronization time.

    TODO: IMPLEMENT: sleep only on the final rank, record timestamps, then
    collectively reduce a deterministic tensor and emit structured records.
    """
    raise NotImplementedError
