"""Opt-in integration tests for point-to-point reduction algorithms."""

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


def test_direct_reduce_defaults_to_sum() -> None:
    result = run_torchrun(
        "phase2.manual_reduce",
        4,
        "--algorithm",
        "manual",
        "--dst",
        "0",
        "--quiet",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    after = [
        record
        for record in json_records(result.stdout)
        if record.get("phase") == "after"
    ]
    assert len(after) == 1
    assert after[0]["values"] == [10]
    assert after[0]["reduce_op"] == "SUM"


@pytest.mark.parametrize(
    ("operator", "dst", "expected"),
    [
        ("SUM", 0, 10),
        ("MAX", 2, 4),
        ("MIN", 3, 1),
        ("PRODUCT", 1, 24),
    ],
)
def test_direct_reduce_supports_all_operators_and_destinations(
    operator: str, dst: int, expected: int
) -> None:
    result = run_torchrun(
        "phase2.manual_reduce",
        4,
        "--algorithm",
        "manual",
        "--dst",
        str(dst),
        "--operator",
        operator,
        "--quiet",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    after = [
        record
        for record in json_records(result.stdout)
        if record.get("phase") == "after"
    ]
    assert len(after) == 1
    assert after[0]["rank"] == dst
    assert after[0]["values"] == [expected]
    assert after[0]["reduce_op"] == operator


@pytest.mark.parametrize(
    ("operator", "nproc", "expected"),
    [
        ("SUM", 4, 10),
        ("SUM", 8, 36),
        ("MAX", 4, 4),
        ("MIN", 4, 1),
        ("PRODUCT", 4, 24),
    ],
)
def test_tree_reduce_supports_all_operators_at_rank_zero(
    operator: str, nproc: int, expected: int
) -> None:
    result = run_torchrun(
        "phase2.manual_reduce",
        nproc,
        "--algorithm",
        "tree",
        "--dst",
        "0",
        "--operator",
        operator,
        "--quiet",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    records = json_records(result.stdout)
    before = [record for record in records if record.get("phase") == "before"]
    after = [record for record in records if record.get("phase") == "after"]

    assert {record["rank"] for record in before} == set(range(nproc))
    assert len(after) == 1
    assert after[0]["rank"] == 0
    assert after[0]["values"] == [expected]
    assert after[0]["reduce_op"] == operator
