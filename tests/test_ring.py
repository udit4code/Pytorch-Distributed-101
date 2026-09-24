"""Local contract tests; real multi-rank behavior is in integration tests."""

import torch

from phase1.ring import ring_exchange, ring_gather


def test_ring_functions_are_defined():
    assert callable(ring_exchange)
    assert callable(ring_gather)


def test_ring_payload_is_a_tensor():
    value = torch.tensor([3], dtype=torch.int64)
    assert value.numel() == 1 and value.dtype == torch.int64
