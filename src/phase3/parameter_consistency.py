"""Model initialization and replica consistency exercises."""
import torch.distributed as dist
from torch import nn


def broadcast_model_parameters(model: nn.Module, src: int = 0) -> None:
    """Broadcast every parameter from ``src`` so replicas start identically.

    TODO: IMPLEMENT: validate the source rank and broadcast parameters in a
    deterministic named-parameter order. Keep buffers out of scope and call
    this collectively on every rank.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError


def assert_parameters_in_sync(model: nn.Module, atol: float = 0.0) -> None:
    """Raise AssertionError if any parameter differs across group members.

    TODO: IMPLEMENT: compare each parameter against a canonical rank's copy
    using distributed primitives already studied (for example broadcast a
    clone and compare locally). Every rank must execute the same sequence.
    """
    # TODO: IMPLEMENT
    raise NotImplementedError
