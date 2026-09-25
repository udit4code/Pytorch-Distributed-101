import pytest
from phase2.manual_reduce import manual_reduce_sum, tree_reduce_sum


def test_manual_reduce_is_a_distributed_exercise():
    with pytest.raises(NotImplementedError):
        manual_reduce_sum(None, 0)  # type: ignore[arg-type]


def test_tree_reduce_is_a_distributed_exercise():
    with pytest.raises(NotImplementedError):
        tree_reduce_sum(None)  # type: ignore[arg-type]
