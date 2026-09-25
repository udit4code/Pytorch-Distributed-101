"""Real distributed manual-broadcast checks are opt-in; implementation is an exercise."""
import pytest
from phase2.manual_broadcast import manual_broadcast, tree_broadcast


def test_manual_broadcast_is_a_distributed_exercise():
    with pytest.raises(NotImplementedError):
        manual_broadcast(None, 0)  # type: ignore[arg-type]


def test_tree_broadcast_is_a_distributed_exercise():
    with pytest.raises(NotImplementedError):
        tree_broadcast(None)  # type: ignore[arg-type]
