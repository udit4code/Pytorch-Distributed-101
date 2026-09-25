"""CPU/Gloo process-group lifecycle and structured worker output."""
from __future__ import annotations
import json
import os
import time
from datetime import timedelta
import torch
import torch.distributed as dist


def setup_process_group(timeout_seconds: int = 30) -> None:
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=timeout_seconds))


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def emit(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def rank_record(rank: int, operation: str, tensor: torch.Tensor | None = None, **extra: object) -> None:
    emit({"rank": rank, "operation": operation, "shape": list(tensor.shape) if tensor is not None else None,
          "timestamp": time.time(), "values": tensor.tolist() if tensor is not None else None, **extra})
