"""Small structured trace records for collective entry and completion."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import time
import torch


@dataclass
class CollectiveEvent:
    rank: int
    operation: str
    group_name: str
    tensor_shape: tuple[int, ...] | None
    timestamp: float
    phase: str


def trace_collective(rank: int, operation: str, group_name: str, tensor: torch.Tensor | None, phase: str) -> None:
    event = CollectiveEvent(rank, operation, group_name, tuple(tensor.shape) if tensor is not None else None,
                            time.time(), phase)
    print(json.dumps(asdict(event)), flush=True)
