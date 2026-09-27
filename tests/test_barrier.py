"""Opt-in integration coverage for the Phase 2 barrier demonstration."""

import json
import os
import platform
import re
import shutil
import socket
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("PHASE2_RUN_DISTRIBUTED") != "1",
    reason="opt in to real Phase 2 torchrun tests",
)


def run_barrier_demo() -> subprocess.CompletedProcess[str]:
    """Launch four local workers with an explicit loopback rendezvous."""
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun is not installed")

    # Ask the OS for an unused local port instead of relying on the default
    # torchrun port, which may already be occupied by another test or job.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0" if platform.system() == "Darwin" else "lo"
    return subprocess.run(
        [
            torchrun,
            "--nnodes=1",
            "--nproc-per-node=4",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "-m",
            "phase2.barrier_demo",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def json_records(stdout: str) -> list[dict[str, object]]:
    """Extract records even if output from two workers shares one line."""
    return [json.loads(match) for match in re.findall(r"\{[^{}\n]*\}", stdout)]


def test_all_ranks_reach_barrier_and_keep_local_data() -> None:
    result = run_barrier_demo()
    assert result.returncode == 0, result.stdout + result.stderr

    records = json_records(result.stdout)
    before = {
        int(record["rank"]): record
        for record in records
        if record.get("phase") == "before"
    }
    after = {
        int(record["rank"]): record
        for record in records
        if record.get("phase") == "after"
    }

    # Every process must enter and leave the barrier exactly once.
    assert set(before) == {0, 1, 2, 3}, result.stdout
    assert set(after) == {0, 1, 2, 3}, result.stdout

    # A barrier coordinates control flow; it does not copy rank-local tensors.
    for rank in range(4):
        assert before[rank]["values"] == [rank]
        assert after[rank]["values"] == [rank]
        assert float(after[rank]["timestamp"]) >= float(before[rank]["timestamp"])

    waits = {rank: float(after[rank]["wait_seconds"]) for rank in range(4)}

    # The configured sleeps are 0, 1, 2, and 4 seconds. Earlier arrivals wait
    # longer for rank 3, while the final arrival should wait very little. Exact
    # durations vary with process scheduling, so assert ordering and broad
    # bounds rather than exact wall-clock values.
    assert waits[0] > waits[1] > waits[2] > waits[3], waits
    assert waits[0] >= 3.0, waits
    assert waits[3] < 1.5, waits

    # Once rank 3 arrives, all ranks should leave within a short interval.
    departure_times = [float(after[rank]["timestamp"]) for rank in range(4)]
    assert max(departure_times) - min(departure_times) < 1.0, departure_times
