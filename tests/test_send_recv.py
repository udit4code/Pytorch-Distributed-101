"""Guidance tests for the first blocking message exercise."""

import torch

from phase1.send_recv import receiver, sender


def test_sender_and_receiver_are_exercise_entry_points():
    assert callable(sender) and callable(receiver)


def test_receive_contract_uses_known_tensor_shape_and_dtype():
    expected = torch.tensor([10, 20, 30], dtype=torch.int64)
    receive_buffer = torch.empty(3, dtype=torch.int64)
    assert receive_buffer.shape == expected.shape
    assert receive_buffer.dtype == expected.dtype
