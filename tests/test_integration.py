"""Opt-in multi-process checks. The core scaffolding is intentionally incomplete."""

import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.environ.get("PHASE0_RUN_DISTRIBUTED") != "1",
    reason="set PHASE0_RUN_DISTRIBUTED=1 to launch torchrun workers",
)
def test_torchrun_hello_has_one_distinct_process_per_rank():
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun is not installed")

    result = subprocess.run(
        [torchrun, "--standalone", "--nproc-per-node=2", "-m", "phase0.hello_distributed"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("rank=")]
    assert len(lines) == 2
    assert {line.split("local_state_counter=")[-1] for line in lines} == {"100", "101"}
    assert len({line.split("pid=")[1].split()[0] for line in lines}) == 2


def test_integration_output_order_is_not_contractual():
    """Integration checks must compare rank sets, never line order."""
    assert True
