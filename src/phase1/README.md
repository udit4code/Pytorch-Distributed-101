# PyTorch Distributed — Phase 1: Point-to-point communication

## 1. Goal

Build the protocol-level mental model for communicating tensor data between independent processes. This phase uses CPU tensors, Gloo, Python 3.11+, and `torchrun`; it does not use collective APIs, DDP/FSDP, CUDA, or NCCL. The communication functions in the exercises are intentionally unfinished. Implement each `TODO: IMPLEMENT` only after predicting what the experiment will do.

Run these commands from the repository root (the directory containing `pyproject.toml`). Set up the project environment once:

```bash
uv sync --group dev
```

Use `uv run -- ...` for the commands below. This runs them in the project’s `.venv` with `src/phase1` on the import path. If you prefer an activated virtual environment, activate `.venv` first and omit the `uv run --` prefix. Examples use a 30-second process-group timeout; intentionally broken experiments should also be run under an outer shell/test timeout.

On macOS, select the loopback interface for local Gloo runs before launching examples:

```bash
export GLOO_SOCKET_IFNAME=lo0
```

Keep this terminal open while running the commands below. PyTorch uses `GLOO_SOCKET_IFNAME` to select the network interface for Gloo communication. The local launch commands use an explicit IPv4 rendezvous address (`127.0.0.1`) so they do not depend on hostname resolution.

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
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv
```

Complete `sender(dst)` and `receiver(src)` in `send_recv.py` with `dist.send()` and `dist.recv()`. The receiver must allocate a compatible destination tensor before `recv()` because the receive operation writes bytes into storage already allocated by that process.

### Experiment: metadata is a contract

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario metadata
```

The sender has four float values; the receiver intentionally allocates a three-element buffer. Use a bounded run and inspect the error. Then try a four-element integer buffer. **Does `dist.send()` transmit a Python/PyTorch object with arbitrary metadata, or does sender and receiver need to agree on the communication contract?** Do not answer before running the experiment.

### Scalar and rank messages

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario scalar
```

Encode `42` and rank IDs as one-element tensors. Do not use `send_object_list`. To also exercise the rank-ID extension, use four ranks:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario scalar
```

### Many senders, one receiver

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario many_to_one
```

Rank 0 receives from sources 1, 2, and 3 explicitly. Output order across processes is not a contract. What happens if rank 0 receives in order 1,2,3 but rank 3 sends first?

## 4. Matching communication operations

```text
Rank 0                       Rank 1

send(dst=1)  ──────────────> recv(src=0)
```

These operations form a protocol. If one side never performs the expected matching operation, progress may stop. That is the entry point to distributed deadlock.

## 5. Blocking behavior

Run the ordered exchange with exactly two ranks:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.bidirectional --scenario ordered
```

Then predict whether the naive pattern could deadlock before trying it:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.bidirectional --scenario naive
```

In the naive pattern, both ranks send before receiving. Use an external timeout for this experiment; buffering can affect whether a small message visibly hangs. The ordered version has rank 0 send then receive, while rank 1 receives then sends. Sender and receiver operations must match by peer, tensor contract, and protocol order.

## 6. Deadlocks

Each failure scenario has an eight-second work wait and a ten-second process-group timeout. Still run it in a bounded subprocess because process-group timeouts do not instantly cancel every worker. All scenarios require two ranks except `wrong_source`, which requires three: rank 0 waits for rank 2 while rank 1 sends. For example:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.failures --scenario receiver_never_receives
```

For `wrong_source`, change `--nproc-per-node=2` to `--nproc-per-node=3`. The other scenarios are `sender_never_sends`, `shape_mismatch`, `dtype_mismatch`, `rank_exits_early`, and `circular_wait`. Read each JSON trace record's rank, PID, source, destination, operation, shape, dtype, and scenario to see where the protocol diverges.

## 7. Ring communication

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring
```

For each rank, `next_rank = (rank + 1) % world_size` and `prev_rank = (rank - 1 + world_size) % world_size`. A single exchange should make rank 0 receive 3, rank 1 receive 0, rank 2 receive 1, and rank 3 receive 2. Study `ring_exchange`: rank 0 sends first, and the other ranks receive before forwarding. This rank-ordered protocol uses only point-to-point `send` and `recv`, and breaks the circular wait. Then compare `broken_ring_exchange` (every rank sends first) with `safe_ring_exchange`. Use an external timeout for the broken variant.

Run the safe implementation explicitly with:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario safe
```

The broken pattern is deliberately unsafe; bound it with GNU `timeout` (available as `gtimeout` on macOS when GNU coreutils is installed):

```bash
GLOO_SOCKET_IFNAME=lo0 gtimeout 15s uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario broken
```

For repeated circulation, each rank forwards the most recently received rank ID. `circulate()` repeats the safe exchange and records the initial value plus each received value. Run it with:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario circulate
```

After three steps, rank 0 should have seen `[0, 3, 2, 1]`. Track the values by hand for another rank: each step moves every payload one neighbor clockwise.

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

`ring_gather(local_tensor)` uses only nonblocking point-to-point sends and receives. Do not add `all_gather`, `broadcast`, `gather`, `all_reduce`, or another collective. With local values `[0]`, `[1]`, `[2]`, `[3]`, every rank should reconstruct `[0, 1, 2, 3]`. Draw the payload movement for every ring step and trace the `send_index` and `receive_index` calculations in the implementation.

Run the capstone with four ranks:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario gather
```

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

On macOS, prefix each test command with `GLOO_SOCKET_IFNAME=lo0` to make Gloo use the loopback interface. This avoids Gloo's automatic network-interface selection, which on some Macs can produce IPv6 hostname-resolution warnings or prevent local workers from connecting. The integration-test helper also sets this variable for its `torchrun` subprocesses. On Linux, use `GLOO_SOCKET_IFNAME=lo` instead.

Run the send/recv unit tests from the repository root:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run --group dev pytest tests/test_send_recv.py
```

Run the complete unit suite:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run --group dev pytest
```

The real multi-process integration tests are opt-in. Their test helper sets the Gloo interface and selects a free port with `--master-addr=127.0.0.1`, avoiding the hostname-resolution problem seen with `torchrun --standalone` on some Macs:

```bash
GLOO_SOCKET_IFNAME=lo0 PHASE1_RUN_DISTRIBUTED=1 uv run --group dev pytest tests/test_integration.py
```

Some integration cases will fail until their exercise TODOs are implemented. Each subprocess has a timeout so a protocol deadlock does not hang pytest indefinitely.
