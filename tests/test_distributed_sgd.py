"""Numerical equivalence contract for manual synchronous SGD."""
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_one_and_multiple_steps_match_reference_and_stay_synchronized():
    result = run_torchrun("phase3.distributed_sgd", 4, "--steps", "3", "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_gradient_synchronizer_declared():
    from phase3.distributed_sgd import synchronize_gradients
    assert callable(synchronize_gradients)
