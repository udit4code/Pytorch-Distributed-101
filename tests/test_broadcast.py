"""Opt-in torchrun integration coverage for native broadcast."""

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


def run_broadcast(src: int) -> subprocess.CompletedProcess[str]:
    """Launch four workers using an explicit loopback rendezvous."""
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun is not installed")

    # Ask the OS for a free port. Passing it explicitly avoids --standalone's
    # hostname-based rendezvous, which may not resolve on every local machine.
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
            "phase2.broadcast_demo",
            "--src",
            str(src),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def json_records(stdout: str) -> list[dict[str, object]]:
    """Extract flat JSON records even when worker writes are interleaved."""
    return [json.loads(match) for match in re.findall(r"\{[^{}\n]*\}", stdout)]


@pytest.mark.parametrize(("src", "value"), [(0, 100), (2, 999)])
def test_broadcast_all_ranks_receive_source_value(src: int, value: int) -> None:
    result = run_broadcast(src)
    assert result.returncode == 0, result.stdout + result.stderr

    after = [
        record
        for record in json_records(result.stdout)
        if record.get("operation") == "broadcast" and record.get("phase") == "after"
    ]
    assert {int(record["rank"]) for record in after} == {0, 1, 2, 3}
    assert all(record["values"] == [value] for record in after)
