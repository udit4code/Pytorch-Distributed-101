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


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_global_mean_loss_weights_uneven_shards_and_reaches_every_rank():
    result = run_torchrun("phase3.distributed_sgd", 4, "--metrics-self-check")
    assert result.returncode == 0, result.stdout + result.stderr

    # torchrun may concatenate adjacent worker writes on one output line, so
    # decode JSON values from the complete stream instead of parsing by lines.
    decoder = json.JSONDecoder()
    records = []
    position = 0
    while (start := result.stdout.find("{", position)) >= 0:
        try:
            record, end = decoder.raw_decode(result.stdout, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(record, dict):
            records.append(record)
        position = end

    checks = [
        record for record in records
        if record.get("operation") == "global_loss_self_check"
    ]
    assert {record["rank"] for record in checks} == {0, 1, 2, 3}, result.stdout
    assert all(record["phase"] == "passed" for record in checks)
    assert all(record["global_mean_loss"] == 3.0 for record in checks)
    assert all(record["global_example_count"] == 10 for record in checks)


def test_gradient_synchronizer_declared():
    from phase3.distributed_sgd import synchronize_gradients
    assert callable(synchronize_gradients)
