"""Opt-in real torchrun subprocess tests; enable with PHASE1_RUN_DISTRIBUTED=1."""

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys

import pytest


def run_torchrun(module: str, nproc: int, *args: str, timeout: int = 30):
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
        capture_output=True, text=True, timeout=timeout, env=env,
    )


pytestmark = pytest.mark.skipif(os.environ.get("PHASE1_RUN_DISTRIBUTED") != "1", reason="opt in to real torchrun tests")


def test_basic_send_recv():
    result = run_torchrun("phase1.send_recv", 2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[10, 20, 30]" in result.stdout


def test_ring_exchange_and_process_isolation():
    result = run_torchrun("phase1.ring", 4)
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = {(int(a), int(b)) for a, b in re.findall(r"rank (\d+) receives (\d+)", result.stdout)}
    assert pairs == {(0, 3), (1, 0), (2, 1), (3, 2)}
    pids = {int(pid) for pid in re.findall(r'"pid": (\d+)', result.stdout)}
    assert len(pids) == 4, result.stdout


def test_ring_gather_reconstructs_all_values():
    result = run_torchrun("phase1.ring", 4, "--scenario", "gather")
    assert result.returncode == 0, result.stdout + result.stderr
    gathered = re.findall(r"rank \d+ gathered \[([^]]+)\]", result.stdout)
    assert len(gathered) == 4
    assert all([int(v.strip()) for v in row.split(",")] == [0, 1, 2, 3] for row in gathered)


def test_invalid_protocol_terminates_with_finite_timeout():
    result = run_torchrun("phase1.failures", 2, "--scenario", "receiver_never_receives", timeout=25)
    assert result.returncode != 0
    assert "Timeout" in result.stdout + result.stderr or "timeout" in result.stdout + result.stderr
