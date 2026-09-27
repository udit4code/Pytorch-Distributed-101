"""Opt-in integration tests for point-to-point broadcast algorithms."""

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


def run_manual_broadcast(
    algorithm: str, src: int, value: int
) -> subprocess.CompletedProcess[str]:
    """Run one point-to-point broadcast algorithm with four local workers."""
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
            "--nproc-per-node=4",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "-m",
            "phase2.manual_broadcast",
            "--algorithm",
            algorithm,
            "--src",
            str(src),
            "--value",
            str(value),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def json_records(stdout: str) -> list[dict[str, object]]:
    """Extract flat JSON records even when worker writes are interleaved."""
    return [json.loads(match) for match in re.findall(r"\{[^{}\n]*\}", stdout)]


@pytest.mark.parametrize(
    ("algorithm", "src", "value"),
    [
        ("manual", 2, 999),
        ("tree", 0, 100),
    ],
)
def test_point_to_point_broadcast_reaches_every_rank(
    algorithm: str, src: int, value: int
) -> None:
    result = run_manual_broadcast(algorithm, src, value)
    assert result.returncode == 0, result.stdout + result.stderr

    operation = f"{algorithm}_broadcast"
    records = json_records(result.stdout)
    before = [
        record
        for record in records
        if record.get("operation") == operation and record.get("phase") == "before"
    ]
    after = [
        record
        for record in records
        if record.get("operation") == operation and record.get("phase") == "after"
    ]

    # Only the selected source owns the value before communication.
    assert {int(record["rank"]): record["values"] for record in before} == {
        rank: [value if rank == src else -1] for rank in range(4)
    }

    # Direct fan-out and tree propagation must have the same broadcast result.
    assert {int(record["rank"]) for record in after} == {0, 1, 2, 3}
    assert all(record["values"] == [value] for record in after)
