"""Guidance tests for the first blocking message exercise."""

import torch
import pytest

from phase1.send_recv import receiver, sender, validate_world_size


def test_sender_and_receiver_are_exercise_entry_points():
    assert callable(sender) and callable(receiver)


def test_receive_contract_uses_known_tensor_shape_and_dtype():
    expected = torch.tensor([10, 20, 30], dtype=torch.int64)
    receive_buffer = torch.empty(3, dtype=torch.int64)
    assert receive_buffer.shape == expected.shape
    assert receive_buffer.dtype == expected.dtype


@pytest.mark.parametrize("scenario", ["basic", "many_to_one", "scalar"])
def test_scenarios_requiring_two_ranks_reject_single_rank(scenario):
    with pytest.raises(ValueError, match=f"{scenario}.*at least 2 ranks"):
        validate_world_size(scenario, 1)


@pytest.mark.parametrize("scenario", ["basic", "many_to_one", "scalar"])
def test_scenarios_requiring_two_ranks_accept_two_or_more(scenario):
    validate_world_size(scenario, 2)
    validate_world_size(scenario, 4)


def test_metadata_scenario_requires_exactly_two_ranks():
    validate_world_size("metadata", 2)
    with pytest.raises(ValueError, match="metadata.*exactly 2 ranks"):
        validate_world_size("metadata", 1)
    with pytest.raises(ValueError, match="metadata.*exactly 2 ranks"):
        validate_world_size("metadata", 4)
