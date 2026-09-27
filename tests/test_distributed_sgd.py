"""Numerical equivalence contract for manual synchronous SGD."""
import json
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_one_and_multiple_steps_match_reference_and_stay_synchronized():
    result = run_torchrun("phase3.distributed_sgd", 4, "--steps", "3", "--self-check")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{")
    ]
    assert len(records) == 3, result.stdout
    assert [record["step"] for record in records] == [0, 1, 2]
    assert all(record["world_size"] == 4 for record in records)
    assert all(record["local_batch_size"] == 4 for record in records)
    assert all(record["global_batch_size"] == 16 for record in records)
    assert all(record["max_replica_diff"] <= 1e-6 for record in records)
    assert all(record["reference_max_diff"] <= 1e-5 for record in records)


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_replicas_diverge_without_gradient_synchronization():
    result = run_torchrun(
        "phase3.distributed_sgd", 4, "--steps", "1", "--sync-gradients=false"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{")
    ]
    assert len(records) == 1, result.stdout
    assert records[0]["sync_gradients"] is False
    assert records[0]["max_replica_diff"] > 0.0


def test_gradient_synchronizer_declared():
    from phase3.distributed_sgd import synchronize_gradients
    assert callable(synchronize_gradients)
