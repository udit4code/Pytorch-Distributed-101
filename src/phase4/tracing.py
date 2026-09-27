"""Structured tracing of logical tensor shape and physical local ownership."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from typing import Literal

TensorStateName = Literal["replicated", "sharded", "partial"]


@dataclass(frozen=True)
class TensorState:
    """Describe one rank's physical tensor relative to its logical tensor."""
    rank: int
    name: str
    logical_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    state: TensorStateName

    def __post_init__(self) -> None:
        if self.rank < 0 or not self.name:
            raise ValueError("rank must be nonnegative and name must be nonempty")
        if self.state not in ("replicated", "sharded", "partial"):
            raise ValueError("state must be replicated, sharded, or partial")
        if any(size < 0 for size in (*self.logical_shape, *self.local_shape)):
            raise ValueError("shape dimensions must be nonnegative")

    def to_json(self) -> str:
        """Serialize shapes as JSON arrays for machine-readable traces."""
        return json.dumps(asdict(self), sort_keys=True)
