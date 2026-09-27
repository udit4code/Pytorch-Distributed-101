"""Opt-in integration tests for Phase 2 process groups."""

import json
import os
import platform
import re
import shutil
import socket
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED") != "1",
    reason="set RUN_DISTRIBUTED=1 to run real torchrun tests",
)


def run_process_groups(nproc: int) -> subprocess.CompletedProcess[str]:
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun is not installed")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0" if platform.system() == "Darwin" else "lo"
    return subprocess.run(
        [
            torchrun,
            "--nnodes=1",
            f"--nproc-per-node={nproc}",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "-m",
            "phase2.process_groups",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def json_records(stdout: str) -> list[dict[str, object]]:
    """Extract flat JSON records from interleaved worker output."""
    return [json.loads(match) for match in re.findall(r"\{[^{}\n]*\}", stdout)]


def test_pair_subgroups_broadcast_independently() -> None:
    result = run_process_groups(nproc=4)
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

    assert set(before) == {0, 1, 2, 3}, result.stdout
    assert set(after) == {0, 1, 2, 3}, result.stdout

    assert {rank: record["values"] for rank, record in before.items()} == {
        0: [100],
        1: [-1],
        2: [200],
        3: [-1],
    }
    assert {rank: record["values"] for rank, record in after.items()} == {
        0: [100],
        1: [100],
        2: [200],
        3: [200],
    }
    assert {rank: record["group"] for rank, record in after.items()} == {
        0: "A",
        1: "A",
        2: "B",
        3: "B",
    }
    assert {rank: record["src"] for rank, record in after.items()} == {
        0: 0,
        1: 0,
        2: 2,
        3: 2,
    }


def test_pair_group_demo_rejects_wrong_world_size() -> None:
    result = run_process_groups(nproc=2)
    assert result.returncode != 0
    assert "pair-group demo requires world_size=4" in result.stdout + result.stderr
