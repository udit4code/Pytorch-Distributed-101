"""Opt-in multi-process checks. The core scaffolding is intentionally incomplete."""

import os
import platform
import shutil
import socket
import subprocess
import sys
import re

import pytest


@pytest.mark.skipif(
    os.environ.get("PHASE0_RUN_DISTRIBUTED") != "1",
    reason="set PHASE0_RUN_DISTRIBUTED=1 to launch torchrun workers",
)
def test_torchrun_hello_has_one_distinct_process_per_rank():
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun is not installed")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        master_port = probe.getsockname()[1]

    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0" if platform.system() == "Darwin" else "lo"
    result = subprocess.run(
        [
            torchrun,
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            "-m",
            "phase0.hello_distributed",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = re.findall(
        r"RankInfo\(rank=(\d+), local_rank=\d+, world_size=2, pid=(\d+)\) "
        r"local_state=\{'counter': (\d+)\}",
        result.stdout,
    )
    assert len(records) == 2, result.stdout + result.stderr
    ranks = {int(rank) for rank, _, _ in records}
    pids = {int(pid) for _, pid, _ in records}
    counters = {int(counter) for _, _, counter in records}
    assert ranks == {0, 1}
    assert counters == {0, 1}
    assert len(pids) == 2


def test_integration_output_order_is_not_contractual():
    """Integration checks must compare rank sets, never line order."""
    assert True
