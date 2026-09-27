"""Opt-in real torchrun subprocess tests; enable with RUN_DISTRIBUTED=1."""

import json
import os
import platform
import re
import shutil
import socket
import subprocess

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


def _json_records(output: str) -> list[dict[str, object]]:
    """Decode JSON objects even when workers' stdout writes are adjacent."""
    decoder = json.JSONDecoder()
    records = []
    position = 0
    while position < len(output):
        start = output.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(output, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            records.append(value)
        position = end
    return records


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED") != "1",
    reason="set RUN_DISTRIBUTED=1 to run real torchrun tests",
)


def test_basic_send_recv():
    result = run_torchrun("phase1.send_recv", 2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[10, 20, 30]" in result.stdout


def test_scalar_tensor_message():
    result = run_torchrun("phase1.send_recv", 2, "--scenario", "scalar")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "received from rank 0: 42" in result.stdout


def test_many_to_one_receives_from_each_sender():
    result = run_torchrun("phase1.send_recv", 4, "--scenario", "many_to_one")
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in (1, 2, 3):
        assert f"received from rank {rank}: {rank}" in result.stdout


def test_ordered_bidirectional_exchange():
    result = run_torchrun(
        "phase1.bidirectional", 2, "--scenario", "ordered"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    received = {
        int(rank): int(value)
        for rank, value in re.findall(r"rank (\d+) received \[(\d+)\]", result.stdout)
    }
    assert received == {0: 20, 1: 10}, result.stdout


def test_message_order_maps_by_receive_sequence_not_variable_name():
    result = run_torchrun("phase1.ordering", 2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "receive order=B-first: A_buffer=[2], B_buffer=[1]" in result.stdout

    result = run_torchrun(
        "phase1.ordering", 2, "--receive-order", "A-first"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "receive order=A-first: A_buffer=[1], B_buffer=[2]" in result.stdout


def test_tagged_messages_match_by_tag():
    result = run_torchrun("phase1.ordering", 2, "--scenario", "tags")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "A_buffer=[10], B_buffer=[20]" in result.stdout


def test_async_ring_exchange():
    result = run_torchrun("phase1.async_comm", 4)
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = {
        (int(rank), int(value))
        for rank, value in re.findall(r"rank (\d+) receives (\d+)", result.stdout)
    }
    assert pairs == {(0, 3), (1, 0), (2, 1), (3, 2)}, result.stdout


def test_blocking_and_nonblocking_timestamps_follow_local_event_order():
    result = run_torchrun("phase1.async_comm", 2, "--scenario", "timings")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        record for record in _json_records(result.stdout)
        if "blocking_communication_start" in record
    ]
    assert {record["rank"] for record in records} == {0, 1}, result.stdout
    for record in records:
        assert (
            record["blocking_communication_start"]
            <= record["blocking_communication_complete"]
            <= record["blocking_computation_start"]
            <= record["blocking_computation_end"]
        )
        assert (
            record["nonblocking_communication_start"]
            <= record["nonblocking_computation_start"]
            <= record["nonblocking_computation_end"]
            <= record["nonblocking_wait_start"]
            <= record["nonblocking_communication_complete"]
        )


def test_blocking_and_nonblocking_timestamps_follow_execution_order():
    result = run_torchrun("phase1.async_comm", 2, "--scenario", "timings")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        record for record in _json_records(result.stdout)
        if "blocking_communication_start" in record
    ]
    assert {record["rank"] for record in records} == {0, 1}, result.stdout
    for record in records:
        assert (
            record["blocking_communication_start"]
            <= record["blocking_communication_complete"]
            <= record["blocking_computation_start"]
            <= record["blocking_computation_end"]
        )
        assert (
            record["nonblocking_communication_start"]
            <= record["nonblocking_computation_start"]
            <= record["nonblocking_computation_end"]
            <= record["nonblocking_wait_start"]
            <= record["nonblocking_communication_complete"]
        )


def test_ring_exchange_and_process_isolation():
    result = run_torchrun("phase1.ring", 4)
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = {(int(a), int(b)) for a, b in re.findall(r"rank (\d+) receives (\d+)", result.stdout)}
    assert pairs == {(0, 3), (1, 0), (2, 1), (3, 2)}
    pids = {int(pid) for pid in re.findall(r'"pid": (\d+)', result.stdout)}
    assert len(pids) == 4, result.stdout


def test_ring_circulation_observes_all_other_rank_values():
    result = run_torchrun("phase1.ring", 4, "--scenario", "circulate")
    assert result.returncode == 0, result.stdout + result.stderr
    observations = {
        int(rank): [int(value.strip()) for value in values.split(",")]
        for rank, values in re.findall(r"rank (\d+) saw values \[([^]]+)\]", result.stdout)
    }
    assert observations == {
        0: [0, 3, 2, 1],
        1: [1, 0, 3, 2],
        2: [2, 1, 0, 3],
        3: [3, 2, 1, 0],
    }, result.stdout


def test_safe_ring_exchange():
    result = run_torchrun("phase1.ring", 4, "--scenario", "safe")
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = {
        (int(rank), int(value))
        for rank, value in re.findall(r"rank (\d+) receives (\d+)", result.stdout)
    }
    assert pairs == {(0, 3), (1, 0), (2, 1), (3, 2)}, result.stdout


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


def test_phase2_capstone_global_metric():
    result = run_torchrun("phase2.algorithms", 4, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    capstone = [r for r in records if "global_mean_loss" in r]
    assert capstone == [{"global_mean_loss": 5.0, "total_examples": 10, "world_size": 4}]
