"""Local semantic checks for Phase 4 utilities."""
import json

import pytest
import torch

from phase4.sharded_tensor_demo import create_local_parameter_shard
from phase4.tracing import TensorState


def test_local_parameter_shard() -> None:
    parameter = torch.arange(8)
    assert torch.equal(create_local_parameter_shard(parameter, 2, 4), torch.tensor([4, 5]))


@pytest.mark.parametrize("rank,world", [(-1, 4), (4, 4), (0, 0)])
def test_invalid_shard_metadata(rank: int, world: int) -> None:
    with pytest.raises(ValueError):
        create_local_parameter_shard(torch.arange(8), rank, world)


def test_tensor_state_json_preserves_logical_and_local_shapes() -> None:
    state = TensorState(2, "weight", (8,), (2,), "sharded")
    assert json.loads(state.to_json()) == {
        "rank": 2, "name": "weight", "logical_shape": [8],
        "local_shape": [2], "state": "sharded",
    }
