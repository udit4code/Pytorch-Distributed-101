"""Bounded failure experiments. Use torchrun and a finite process-group timeout."""

import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from phase1.distributed import cleanup_process_group, setup_process_group

SCENARIOS = (
    "receiver_never_receives",
    "sender_never_sends",
    "wrong_source",
    "shape_mismatch",
    "dtype_mismatch",
    "rank_exits_early",
    "circular_wait",
)
WAIT_TIMEOUT_SECONDS = 8
IDLE_SECONDS = WAIT_TIMEOUT_SECONDS + 4


def _log(
    scenario: str,
    operation: str,
    *,
    source: int | None,
    destination: int | None,
    tensor: torch.Tensor | None,
) -> None:
    """Print the protocol details needed to diagnose a stalled rank."""
    print(
        json.dumps(
            {
                "scenario": scenario,
                "rank": dist.get_rank(),
                "pid": os.getpid(),
                "source": source,
                "destination": destination,
                "operation": operation,
                "shape": list(tensor.shape) if tensor is not None else None,
                "dtype": str(tensor.dtype) if tensor is not None else None,
                "timestamp": time.time(),
            }
        ),
        flush=True,
    )


def _wait_for_timeout(work, scenario: str, operation: str) -> None:
    """Wait briefly for an operation designed to lack a matching peer."""
    rank = dist.get_rank()
    try:
        work.wait(timeout=timedelta(seconds=WAIT_TIMEOUT_SECONDS))
    except RuntimeError as exc:
        raise RuntimeError(
            f"scenario={scenario} rank={rank}: {operation} timed out "
            f"(timeout={WAIT_TIMEOUT_SECONDS}s; expected failure): {exc}"
        ) from exc
    raise RuntimeError(
        f"scenario={scenario} rank={rank}: {operation} completed unexpectedly; "
        "the backend may have buffered this message"
    )


def _require_world_size(scenario: str, world_size: int) -> None:
    required = 3 if scenario == "wrong_source" else 2
    if world_size != required:
        raise ValueError(
            f"scenario={scenario} requires exactly {required} ranks; got {world_size}"
        )


def run_scenario(scenario: str) -> None:
    rank, size = dist.get_rank(), dist.get_world_size()
    _require_world_size(scenario, size)

    # Make the blocked transfers larger than a trivial scalar so a send is
    # less likely to finish solely because a backend eagerly buffered it.
    payload = torch.ones(4 * 1024 * 1024, dtype=torch.float32)
    rank0 = 0
    rank1 = 1

    if scenario == "receiver_never_receives":
        if rank == rank0:
            _log(scenario, "isend", source=rank, destination=rank1, tensor=payload)
            work = dist.isend(tensor=payload, dst=rank1)
            _wait_for_timeout(work, scenario, "send to receiver that never calls recv")
        else:
            _log(scenario, "skip_recv", source=rank0, destination=rank, tensor=payload)
            # Stay alive while rank 0's bounded send waits for a matching recv.
            time.sleep(IDLE_SECONDS)

    elif scenario == "sender_never_sends":
        if rank == rank0:
            _log(scenario, "skip_send", source=rank, destination=rank1, tensor=payload)
            time.sleep(IDLE_SECONDS)
        else:
            _log(scenario, "irecv", source=rank0, destination=rank, tensor=payload)
            work = dist.irecv(tensor=torch.empty_like(payload), src=rank0)
            _wait_for_timeout(work, scenario, "receive from sender that never calls send")

    elif scenario == "wrong_source":
        expected_source = 2
        if rank == rank0:
            _log(
                scenario,
                "irecv_wrong_source",
                source=expected_source,
                destination=rank,
                tensor=torch.empty(1, dtype=torch.int64),
            )
            work = dist.irecv(
                tensor=torch.empty(1, dtype=torch.int64), src=expected_source
            )
            _wait_for_timeout(work, scenario, "receive from rank 2 while rank 1 sends")
        elif rank == 1:
            message = torch.tensor([rank], dtype=torch.int64)
            _log(scenario, "isend", source=rank, destination=rank0, tensor=message)
            # Keep the send handle alive while rank 0 waits for the wrong source.
            work = dist.isend(tensor=message, dst=rank0)
            time.sleep(IDLE_SECONDS)
            work.wait()
        else:
            _log(scenario, "idle_no_send", source=rank, destination=rank0, tensor=None)
            time.sleep(IDLE_SECONDS)

    elif scenario in ("shape_mismatch", "dtype_mismatch"):
        if scenario == "shape_mismatch":
            outgoing = torch.ones(4, dtype=torch.float32)
            incoming = torch.empty(3, dtype=torch.float32)
        else:
            # Equal element counts but different dtypes produce incompatible
            # byte sizes for this Gloo message (float32 vs int64).
            outgoing = torch.ones(2, dtype=torch.float32)
            incoming = torch.empty(2, dtype=torch.int64)

        if rank == rank0:
            _log(scenario, "send", source=rank, destination=rank1, tensor=outgoing)
            dist.send(tensor=outgoing, dst=rank1)
            raise RuntimeError(f"scenario={scenario}: mismatched send unexpectedly completed")
        else:
            _log(scenario, "recv", source=rank0, destination=rank, tensor=incoming)
            try:
                dist.recv(tensor=incoming, src=rank0)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"scenario={scenario} rank={rank}: receiver detected the "
                    f"incompatible tensor contract: {exc}"
                ) from exc
            raise RuntimeError(
                f"scenario={scenario} rank={rank}: incompatible receive unexpectedly completed"
            )

    elif scenario == "rank_exits_early":
        if rank == rank0:
            receive_buffer = torch.empty(1, dtype=torch.int64)
            _log(
                scenario,
                "irecv_waiting_for_rank_exit",
                source=rank1,
                destination=rank,
                tensor=receive_buffer,
            )
            work = dist.irecv(tensor=receive_buffer, src=rank1)
            _wait_for_timeout(work, scenario, "receive from rank 1 after it exits")
        else:
            _log(scenario, "exit_early", source=rank, destination=None, tensor=None)
            # Deliberately skip process-group cleanup to model abrupt worker loss.
            os._exit(17)

    elif scenario == "circular_wait":
        peer = 1 - rank
        # Both ranks issue a large send and wait for it before posting recv.
        # Each send therefore waits for the receive that the peer has not posted.
        circular_payload = torch.ones(4 * 1024 * 1024, dtype=torch.float32)
        _log(scenario, "isend_then_wait", source=rank, destination=peer, tensor=circular_payload)
        work = dist.isend(tensor=circular_payload, dst=peer)
        _wait_for_timeout(work, scenario, "send-first circular wait before recv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    args = parser.parse_args()
    setup_process_group(timeout_seconds=10)
    try:
        run_scenario(args.scenario)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
