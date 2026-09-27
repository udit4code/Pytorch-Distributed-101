"""Gradient aggregation contract tests exercised by distributed workers."""
import os
import json

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_equal_and_unequal_gradient_aggregation():
    result = run_torchrun("phase3.gradient_sync", 4, "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{")
    ]
    checks = [
        record for record in records
        if record.get("operation") == "gradient_sync_self_check"
    ]
    assert {record["rank"] for record in checks} == {0, 1, 2, 3}, result.stdout
    assert all(record["phase"] == "passed" for record in checks)
    assert all(record["equal_average"] == [2.5] for record in checks)
    assert all(abs(record["weighted_average"][0] - 14 / 3) < 1e-12 for record in checks)


def test_aggregation_functions_exist():
    from phase3.gradient_sync import average_gradient_, weighted_average_gradient_
    assert callable(average_gradient_) and callable(weighted_average_gradient_)
