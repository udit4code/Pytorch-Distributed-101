"""Real distributed AllReduce checks; subprocess tests opt in via RUN_DISTRIBUTED=1."""
import json
import os

import pytest

from test_integration import run_torchrun


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED") != "1", reason="opt-in torchrun integration")
def test_all_reduce_demo_returns_expected_sum_on_every_rank():
    result = run_torchrun("phase3.all_reduce_demo", 4)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    after = [row for row in rows if row.get("phase") == "after"]
    assert {row["rank"] for row in after} == {0, 1, 2, 3}
    assert all(row["values"] == [10, 100] for row in after)


def test_collective_helper_is_declared():
    from phase3.all_reduce_demo import all_reduce_sum_
    assert callable(all_reduce_sum_)
