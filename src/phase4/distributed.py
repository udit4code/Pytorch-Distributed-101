"""Shared CPU/Gloo process-group helpers for Phase 4 exercises."""
from __future__ import annotations

import json
from datetime import timedelta
import torch
import torch.distributed as dist


def setup_process_group(timeout_seconds: int = 20) -> None:
    """Initialize a bounded Gloo group using torchrun environment variables."""
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=timeout_seconds))


def cleanup_process_group() -> None:
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def emit(record: dict[str, object]) -> None:
    """Print one JSON record for integration-test parsing."""
    print(json.dumps(record, sort_keys=True), flush=True)


def rank_record(rank: int, operation: str, tensor: torch.Tensor | None = None, **extra: object) -> None:
    """Emit rank, shape, and values as a structured record."""
    emit({"rank": rank, "operation": operation,
          "shape": list(tensor.shape) if tensor is not None else None,
          "values": tensor.tolist() if tensor is not None else None, **extra})
