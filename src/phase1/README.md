# PyTorch Distributed — Phase 1: Point-to-point communication

## 1. Goal

Build the protocol-level mental model for communicating tensor data between independent processes. This phase uses CPU tensors, Gloo, Python 3.11+, and `torchrun`; it does not use collective APIs, DDP/FSDP, CUDA, or NCCL. The communication functions in the exercises are intentionally unfinished. Implement each `TODO: IMPLEMENT` only after predicting what the experiment will do.

Run these commands from the repository root (the directory containing `pyproject.toml`). Set up the project environment once:

```bash
uv sync --group dev
```

Use `uv run -- ...` for the commands below. This runs them in the project’s `.venv` with `src/phase1` on the import path. If you prefer an activated virtual environment, activate `.venv` first and omit the `uv run --` prefix. Examples use a 30-second process-group timeout; intentionally broken experiments should also be run under an outer shell/test timeout.

## 2. Point-to-point mental model

Each rank owns separate memory.

```text
┌───────────────┐             ┌───────────────┐
│ Rank 0        │             │ Rank 1        │
│               │             │               │
│ tensor A      │             │ recv buffer   │
│               │             │               │
└───────┬───────┘             └──────▲────────┘
        │                            │
        │          send             │
        └───────────────────────────┘
```

`send` does not mean rank 1 gets access to rank 0's tensor object. It transfers tensor data between independent processes. The receiver owns a separate destination tensor.

## 3. send / recv

```bash
uv run -- torchrun --standalone --nproc-per-node=2 -m phase1.send_recv
```

Complete `sender(dst)` and `receiver(src)` in `send_recv.py` with `dist.send()` and `dist.recv()`. The receiver must allocate a compatible destination tensor before `recv()` because the receive operation writes bytes into storage already allocated by that process.

### Experiment: metadata is a contract

```bash
uv run -- torchrun --standalone --nproc-per-node=2 -m phase1.send_recv --scenario metadata
```

The sender has four float values; the receiver intentionally allocates a three-element buffer. Use a bounded run and inspect the error. Then try a four-element integer buffer. **Does `dist.send()` transmit a Python/PyTorch object with arbitrary metadata, or does sender and receiver need to agree on the communication contract?** Do not answer before running the experiment.

### Scalar and rank messages

```bash
uv run -- torchrun --standalone --nproc-per-node=2 -m phase1.send_recv --scenario scalar
```

Encode `42` and rank IDs as one-element tensors. Do not use `send_object_list`. To also exercise the rank-ID extension, use four ranks:

```bash
uv run -- torchrun --standalone --nproc-per-node=4 -m phase1.send_recv --scenario scalar
```

### Many senders, one receiver

```bash
uv run -- torchrun --standalone --nproc-per-node=4 -m phase1.send_recv --scenario many_to_one
```

Rank 0 receives from sources 1, 2, and 3 explicitly. Output order across processes is not a contract. What happens if rank 0 receives in order 1,2,3 but rank 3 sends first?

## 4. Matching communication operations

```text
Rank 0                       Rank 1

send(dst=1)  ──────────────> recv(src=0)
```

These operations form a protocol. If one side never performs the expected matching operation, progress may stop. That is the entry point to distributed deadlock.

## 5. Blocking behavior

Complete `bidirectional.py`. First inspect the intentionally naive pattern, where both ranks send before receiving. Predict: **Could this deadlock?** Then implement the ordered version (rank 0 sends then receives; rank 1 receives then sends). Sender and receiver operations must match by peer, tensor contract, and protocol order.

## 6. Deadlocks

Run deliberately broken cases only in bounded subprocesses. For example:

```bash
uv run -- torchrun --standalone --nproc-per-node=2 -m phase1.failures --scenario receiver_never_receives
```

`failures.py` accepts `receiver_never_receives`, `sender_never_sends`, `wrong_source`, `shape_mismatch`, `dtype_mismatch`, `rank_exits_early`, and `circular_wait`; complete them one by one and note the rank, PID, source, destination, operation, shape, dtype, and scenario from trace events. A process-group timeout is not an instant cancellation mechanism, so keep an outer test timeout too.

## 7. Ring communication

```bash
uv run -- torchrun --standalone --nproc-per-node=4 -m phase1.ring
```

For each rank, `next_rank = (rank + 1) % world_size` and `prev_rank = (rank - 1 + world_size) % world_size`. A single exchange should make rank 0 receive 3, rank 1 receive 0, rank 2 receive 1, and rank 3 receive 2. Complete `ring_exchange`. Then compare `broken_ring_exchange` (send first) with `safe_ring_exchange`; decide a deterministic order that removes the circular wait. Use an external timeout for the broken variant.

For repeated circulation, each rank forwards the most recently received rank ID. After three steps, rank 0 should have seen `[0, 3, 2, 1]`. Complete `circulate` and track the values by hand for another rank.

## 8. Nonblocking communication

Complete `async_comm.py` with `dist.isend()` and `dist.irecv()`. They return asynchronous work handles. The rough sequence is:

```text
launch communication
      ↓
communication is in flight
      ↓
possibly do independent work
      ↓
wait for completion
```

Complete both waits and keep the independent CPU work trivial. Compare the recorded start/end timestamps for blocking and nonblocking variants; do not make broad performance claims from a MacBook CPU run. GPU overlap belongs in a later phase.

## 9. Message ordering

In `ordering.py`, send A then B and experiment with receiving into B's buffer then A's buffer. Variable names have no meaning to the transport layer. Try the optional tag experiment only if Gloo in your installed PyTorch supports the needed behavior cleanly; tags distinguish protocol messages but do not replace rank and ordering agreement.

## 10. Failure experiments

Every failure must be bounded. Run the failure module under `torchrun` with a subprocess timeout. Use JSON trace output to diagnose which rank was waiting for which peer and tensor contract.

## 11. Ring-gather capstone

Complete `ring_gather(local_tensor)` using only `send`, `recv`, `isend`, and `irecv`. Do not use `all_gather`, `broadcast`, `gather`, `all_reduce`, or another collective. With local values `[0]`, `[1]`, `[2]`, `[3]`, every rank must reconstruct `[0, 1, 2, 3]`. Draw the payload movement for every ring step before coding.

## 12. Questions

Answer these in your notes without looking up a solution first:

1. What is point-to-point communication?
2. How is `send/recv` different from shared memory?
3. Why must the receiver generally know tensor shape and dtype?
4. What happens if rank 0 sends but rank 1 never receives?
5. What happens if rank 1 waits for data that rank 0 never sends?
6. Why can two ranks doing `send()` to each other create deadlock?
7. How can communication ordering avoid deadlock?
8. What is the difference between blocking and nonblocking communication?
9. What does `isend()` return?
10. Why do we eventually need `.wait()`?
11. What guarantees do we need about message ordering?
12. Why does the communication protocol need agreement between ranks?
13. How would you debug a program where rank 3 hangs but all other ranks appear fine?
14. What is a ring topology?
15. Why are ring algorithms attractive for distributed collectives?
16. If every rank passes data to its neighbor, how many steps are required for every rank to see every other rank's data?
17. How could you implement `all_gather` using only send/recv?
18. What is the difference between logical topology and physical hardware topology?
19. Why does understanding send/recv help when learning NCCL collectives later?
20. Why are distributed bugs often protocol bugs rather than ordinary algorithm bugs?

## 13. Interview checkpoint

Answer only after completing the exercises:

> Four ranks form a ring. Rank `i` sends to `(i+1) % world_size`. If every rank performs blocking `send()` first and `recv()` second, what potential problem exists?

> Rank 0 has a tensor with 1 billion floats and wants every other rank to obtain it. Could you implement this using repeated point-to-point communication? What are the drawbacks?

> Suppose rank 2 crashes while rank 1 is waiting inside `recv(src=2)`. What happens conceptually?

> Why would real collective libraries implement optimized algorithms rather than having users manually loop over `send/recv`?

> Describe how you could implement an AllGather-like operation using a ring.

> What is the difference between `dist.send()` and `dist.isend()` from the caller's execution perspective?

> Why does asynchronous communication potentially matter for distributed training?

## More questions to keep in mind

What happens if rank 0 sends but rank 1 never receives? What if rank 1 waits for data rank 0 never sends? Why must both ranks agree on sender/receiver ordering and tensor shape? How would you diagnose one stuck rank from structured traces?

## Running tests

Run the local tests with:

```bash
uv run --group dev pytest
```

The real multi-process integration tests are opt-in and launch `torchrun` subprocesses:

```bash
PHASE1_RUN_DISTRIBUTED=1 uv run --group dev pytest tests/test_integration.py
```

Some integration cases will fail until their exercise TODOs are implemented. Each subprocess has a timeout so a protocol deadlock does not hang pytest indefinitely.
