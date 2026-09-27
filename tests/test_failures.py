"""Bounded subprocess tests for the Phase 3 failure demonstrations."""
import json
import os

import pytest

from test_integration import run_torchrun


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED") != "1",
    reason="set RUN_DISTRIBUTED=1 to run real torchrun tests",
)


def _json_records(output: str) -> list[dict[str, object]]:
    """Parse JSON objects even if torchrun joins adjacent worker writes."""
    decoder = json.JSONDecoder()
    records = []
    position = 0
    while position < len(output):
        start = output.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(output, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            records.append(value)
        position = end
    return records


def test_straggler_collective_completes_and_sums_all_ranks():
    result = run_torchrun(
        "phase3.failures",
        4,
        "--scenario",
        "straggler",
        "--seconds",
        "0.2",
        "--timeout-seconds",
        "5",
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = _json_records(result.stdout)
    completed = [record for record in records if record.get("phase") == "after_all_reduce"]
    assert {record["rank"] for record in completed} == {0, 1, 2, 3}
    assert all(record["value"] == [10.0] for record in completed)
    assert all(record["collective_seconds"] >= 0 for record in completed)
    assert any(
        record.get("rank") == 3 and record.get("phase") == "sleep_start"
        for record in records
    )


@pytest.mark.parametrize(
    ("scenario", "nproc"),
    [("mismatched_order", 2), ("missing_rank", 4)],
)
def test_protocol_failures_exit_without_hanging(scenario: str, nproc: int):
    result = run_torchrun(
        "phase3.failures",
        nproc,
        "--scenario",
        scenario,
        "--timeout-seconds",
        "3",
        timeout=20,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    records = _json_records(result.stdout)
    if scenario == "mismatched_order":
        assert any(
            record.get("demo") == scenario
            and record.get("phase") == "before_all_reduce"
            for record in records
        ), result.stdout
    else:
        phases = {record.get("phase") for record in records if record.get("demo") == scenario}
        assert "exit_before_all_reduce" in phases, result.stdout
        assert "before_all_reduce" in phases, result.stdout
