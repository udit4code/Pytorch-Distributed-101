"""Native and manually assembled AllReduce equivalence checks."""
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_manual_reduce_broadcast_and_point_to_point_match_native():
    result = run_torchrun("phase3.manual_all_reduce", 4, "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_manual_functions_exist():
    from phase3.manual_all_reduce import naive_all_reduce_sum, point_to_point_all_reduce_sum
    assert callable(naive_all_reduce_sum) and callable(point_to_point_all_reduce_sum)
