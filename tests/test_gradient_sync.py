"""Gradient aggregation contract tests exercised by distributed workers."""
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_equal_and_unequal_gradient_aggregation():
    result = run_torchrun("phase3.gradient_sync", 4, "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_aggregation_functions_exist():
    from phase3.gradient_sync import average_gradient_, weighted_average_gradient_
    assert callable(average_gradient_) and callable(weighted_average_gradient_)
