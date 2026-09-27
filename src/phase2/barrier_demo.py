"""Barrier timing and synchronization-versus-data experiment."""
import time
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record

# Why do we need barriers ?
# Broadcast and reduce move/transform application data.
# But, it is barrier that establishes a coordination point.
# Synchronisation is not same as data exchange.
def run() -> None:
    setup_process_group()
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank])
        before_sleep = time.time()
        time.sleep({0: 0, 1: 1, 2: 2, 3: 4}.get(rank, rank))
        before = time.time()
        rank_record(rank, "barrier", tensor, phase="before", timestamp_before_sleep=before_sleep)
        # async_op=False is the default blocking form. This Python call returns
        # only after every rank in the default process group has entered the
        # barrier and the barrier operation has completed.
        #
        # With the delays above, the approximate timeline is:
        #
        # rank 0: t=0s -> barrier ---------------- waits about 4s --\
        # rank 1: t=1s ------> barrier ----------- waits about 3s ---+
        # rank 2: t=2s ------------> barrier ----- waits about 2s ---+-> continue
        # rank 3: t=4s ----------------------> barrier, little wait -/
        #
        # The barrier coordinates progress; it does not copy application data.
        # Each rank's local tensor therefore remains [rank].
        #
        # With async_op=True, the call returns a Work handle after launching
        # the operation. The barrier is not complete merely because a handle
        # was returned: call work.wait() before relying on barrier completion.
        # This example uses CPU tensors with Gloo, so no GPU is involved.
        dist.barrier(async_op=False)
        after = time.time()
        rank_record(rank, "barrier", tensor, phase="after", wait_seconds=after-before)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    run()
