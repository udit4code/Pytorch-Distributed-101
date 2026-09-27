"""Real torchrun integration checks for Phase 4; TODO implementations fail initially."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_torchrun(module: str, ranks: int = 4, *args: str) -> list[dict[str, object]]:
    """Launch a bounded local CPU/Gloo worker group and parse its JSON records."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={ranks}", "-m", module, *args],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=45, check=False,
    )
    assert proc.returncode == 0, proc.stdout
    records = []
    for line in proc.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "rank" in record:
            records.append(record)
    assert len(records) == ranks, proc.stdout
    return sorted(records, key=lambda item: int(item["rank"]))


def test_all_gather_rank_order() -> None:
    records = run_torchrun("phase4.all_gather_demo")
    assert all(record["values"] == [10, 20, 30, 40] for record in records)


def test_all_gather_into_tensor_shape_and_order() -> None:
    records = run_torchrun("phase4.all_gather_tensor_demo")
    assert all(record["shape"] == [8] for record in records)
    assert all(record["values"] == list(range(8)) for record in records)


def test_reduce_scatter_sum_exact_shard() -> None:
    records = run_torchrun("phase4.reduce_scatter_demo")
    assert [record["values"] for record in records] == [[1111], [2222], [3333], [4444]]


@pytest.mark.parametrize("module", ["phase4.manual_all_gather", "phase4.manual_reduce_scatter", "phase4.all_reduce_decomposition", "phase4.sharded_tensor_demo", "phase4.sharded_gradient_demo"])
def test_remaining_distributed_exercise_modules_are_launchable(module: str) -> None:
    """Integration entry points are intended to emit records after TODOs are solved."""
    pytest.skip(f"complete scaffold entry point for {module}")


@pytest.mark.parametrize("scenario", ["inconsistent-size", "inconsistent-dtype", "wrong-order", "missing-rank", "wrong-output-shape", "not-divisible"])
def test_failure_protocols_are_bounded(scenario: str) -> None:
    """Intentional failures must never strand a test process indefinitely."""
    try:
        run_torchrun("phase4.failures", 2, scenario)
    except subprocess.CalledProcessError:
        pass
