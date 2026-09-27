"""Parameter initialization and replica checks using real Gloo workers."""
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_broadcast_parameters_makes_replicas_equal():
    result = run_torchrun("phase3.parameter_consistency", 4, "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_parameter_helpers_exist():
    from phase3.parameter_consistency import assert_parameters_in_sync, broadcast_model_parameters
    assert callable(assert_parameters_in_sync) and callable(broadcast_model_parameters)
