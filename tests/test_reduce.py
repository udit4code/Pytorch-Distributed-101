"""Opt-in integration tests for the native reduce demonstration."""

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


def run_torchrun(module: str, nproc: int, *args: str) -> subprocess.CompletedProcess[str]:
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
            module,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )


def json_records(stdout: str) -> list[dict[str, object]]:
    """Extract flat JSON records even when worker output shares one line."""
    return [json.loads(match) for match in re.findall(r"\{[^{}\n]*\}", stdout)]


@pytest.mark.parametrize(
    ("operator", "dst", "expected"),
    [
        ("SUM", 0, [10, 100]),
        ("MAX", 2, [4, 40]),
        ("MIN", 3, [1, 10]),
        ("PRODUCT", 1, [24, 240000]),
    ],
)
def test_native_reduce_records_inputs_and_destination_result(
    operator: str, dst: int, expected: list[int]
) -> None:
    result = run_torchrun(
        "phase2.reduce_demo",
        4,
        "--dst",
        str(dst),
        "--operator",
        operator,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    records = json_records(result.stdout)
    before = [record for record in records if record.get("phase") == "before"]
    after = [record for record in records if record.get("phase") == "after"]

    assert {record["rank"]: record["values"] for record in before} == {
        0: [1, 10],
        1: [2, 20],
        2: [3, 30],
        3: [4, 40],
    }
    assert len(after) == 1
    assert after[0]["rank"] == dst
    assert after[0]["values"] == expected
    assert after[0]["reduce_op"] == operator
