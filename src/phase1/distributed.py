"""Shared Gloo process-group setup and structured tracing."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def setup_process_group(timeout_seconds: int = 30) -> None:
    """Initialize a local torchrun Gloo group with a finite timeout."""
    from datetime import timedelta

    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=timeout_seconds))


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def trace_event(*, rank: int, operation: str, peer: int | None, tensor: torch.Tensor | None) -> None:
    """Emit one machine-readable event per line for debugging protocols."""
    print(json.dumps({
        "rank": rank,
        "pid": os.getpid(),
        "operation": operation,
        "peer": peer,
        "shape": list(tensor.shape) if tensor is not None else None,
        "dtype": str(tensor.dtype) if tensor is not None else None,
        "timestamp": time.time(),
    }), flush=True)
