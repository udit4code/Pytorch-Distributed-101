"""Parameter initialization and replica checks using real Gloo workers."""
import json
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_broadcast_parameters_makes_replicas_equal():
    result = run_torchrun("phase3.parameter_consistency", 4, "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{")
    ]
    checks = [
        record for record in records
        if record.get("operation") == "parameter_consistency_self_check"
    ]
    assert {record["rank"] for record in checks} == {0, 1, 2, 3}, result.stdout
    assert all(record["same_seed_equal"] for record in checks)
    assert all(record["different_seed_diverged"] for record in checks)
    assert all(record["broadcast_equal"] for record in checks)
    assert all(record["phase"] == "passed" for record in checks)


def test_parameter_helpers_exist():
    from phase3.parameter_consistency import assert_parameters_in_sync, broadcast_model_parameters
    assert callable(assert_parameters_in_sync) and callable(broadcast_model_parameters)
