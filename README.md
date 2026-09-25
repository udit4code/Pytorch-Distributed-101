# PyTorch Distributed Phase 1: Point-to-point communication

This tutorial starts with independent Python processes and builds toward a ring-based gather using only tensor sends and receives. It is written for someone new to distributed systems, but the exercises and interview prompts aim at the reasoning expected in a Senior ML Engineer loop.

You will use Python 3.11+, PyTorch, CPU tensors, Gloo, `torchrun`, and real local worker processes. The Phase 1 code does not use GPUs, NCCL, DDP/FSDP, or collective communication APIs. A few optional exercises remain unfinished; they are identified below.

## Learning path

Work through the sections in order. For every experiment, first write down what you expect each rank to do, then run it, read the output, and explain any difference.

1. Understand processes, ranks, and independent memory.
2. Send one tensor from rank 0 to rank 1.
3. Match operations and reason about blocking and deadlock.
4. Move tensors around a ring and circulate values.
5. Build a gather-like algorithm from point-to-point operations.
6. Compare blocking and nonblocking communication.
7. Debug message order and deliberate protocol failures.
8. Practice explaining the design tradeoffs out loud.

## 1. Setup and commands

Run commands from the repository root: the directory containing `pyproject.toml`.

Create or synchronize the project environment once:

```bash
uv sync --group dev
```

On macOS, direct local Gloo traffic over the loopback interface in the terminal where you run examples:

```bash
export GLOO_SOCKET_IFNAME=lo0
```

This tells Gloo which network interface to use. On some Macs, automatic interface selection or hostname resolution produces IPv6 lookup warnings or prevents local workers from connecting. On Linux, the loopback interface is commonly named `lo`, so use `export GLOO_SOCKET_IFNAME=lo` there.

The commands below use an explicit local rendezvous address and port. The address `127.0.0.1` means “this computer.” If port `29500` is already occupied, change it to another free port consistently for that launch. The integration-test helper chooses a free port itself.

`uv run -- ...` runs the command in the project environment, where the `phase1` package is importable from `src/`. If you have activated this project’s virtual environment, you can omit `uv run --`.

## 2. Begin with processes, not tensors

A process is a running program with its own interpreter and memory. When `torchrun` launches four workers, it starts four processes running the same Python module. They do not share ordinary Python variables.

```text
torchrun
  ├── Python process: rank 0, PID 41001, its own memory
  ├── Python process: rank 1, PID 41002, its own memory
  ├── Python process: rank 2, PID 41003, its own memory
  └── Python process: rank 3, PID 41004, its own memory
```

The numbers are illustrative; actual PIDs differ each run.

| Term | Meaning in this project |
| --- | --- |
| Worker | One participating Python process |
| Rank | That process’s identity in the distributed process group, from `0` to `world_size - 1` |
| World size | Number of processes in the group |
| Local rank | A worker’s index on its machine; useful in multi-machine jobs, but not currently used by these Phase 1 examples |
| PID | Operating-system process ID; useful for confirming process isolation, but not the distributed rank |
| Process group | The set of workers that PyTorch has connected for distributed communication |

Each worker runs `setup_process_group()` in [distributed.py](./distributed.py). It initializes PyTorch’s default group with the `gloo` backend and a finite timeout. `torchrun` supplies the rendezvous environment to each child process. `cleanup_process_group()` destroys the group when the example finishes.

The process lifecycle in the executable modules follows this shape:

```text
torchrun starts workers
        ↓
each worker joins the Gloo process group
        ↓
each worker reads its own rank and runs its branch of the protocol
        ↓
workers finish or report an error
        ↓
each worker cleans up the group
```

### First launch: one process per rank

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv
```

`--nproc-per-node=2` starts two local processes. `-m phase1.send_recv` runs the same module in both. Each process reads its own rank from the initialized process group and takes a different branch.

## 3. What point-to-point communication means

Point-to-point communication transfers a tensor from one named process to another named process. In PyTorch, the blocking operations in this phase are `dist.send()` and `dist.recv()`; the asynchronous forms are `dist.isend()` and `dist.irecv()`.

Each side has separate memory:

```text
Rank 0 process                              Rank 1 process
┌─────────────────────┐                    ┌─────────────────────┐
│ tensor A = [10,20]  │                    │ receive buffer      │
│                     │                    │ [?, ?]              │
└──────────┬──────────┘                    └──────────▲──────────┘
           │                                           │
           └──────────── send data ───────────────────┘
                              recv writes here
```

Rank 1 does not gain access to rank 0’s Python tensor object. Rank 0 has a tensor in its address space; rank 1 allocates its own destination tensor. Communication copies the tensor data according to a protocol agreed by the ranks.

### The receive buffer is part of the protocol

`dist.recv(tensor=buffer, src=peer)` writes into storage that already exists in the receiving process. The receiver chooses the buffer, so it needs to know what shape and dtype to expect. Do not treat `send()` as Python object serialization that automatically constructs a matching tensor on the other side.

The sender and receiver need a compatible contract:

```text
sender:   destination=1, tensor shape=[3], dtype=int64
receiver: source=0,      buffer shape=[3], dtype=int64
```

The one-element tensors used in scalar examples are still tensors. This phase intentionally avoids `send_object_list` so you can see that the communication payload is tensor data.

## 4. Basic blocking `send()` and `recv()`

Code: [send_recv.py](./send_recv.py), functions `sender()` and `receiver()`.

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv
```

Protocol:

```text
Rank 0                                  Rank 1
value = [10, 20, 30]                   buffer = empty(3, int64)
send(value, dst=1)  ─────────────────> recv(buffer, src=0)
```

Rank 0 calls `sender(1)` and rank 1 calls `receiver(0)`. `sender()` creates an `int64` tensor `[10, 20, 30]`. `receiver()` creates a three-element `int64` buffer before calling `recv()`.

The helper signatures also allow the caller to pass a tensor/buffer explicitly. `many_to_one()` uses that form so each sender can send its rank ID and rank 0 can allocate a one-element receive buffer.

### Experiment: make the contract disagree

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario metadata
```

Rank 0 sends four `float32` values; rank 1 allocates only three. This is a deliberately invalid experiment. Observe the backend error and the process exit. Then edit the receiver to use four elements but `int64`, and observe what your installed Gloo/PyTorch build does.

Before running it, answer: **Does `send()` construct an arbitrary Python/PyTorch object at the receiver, or must the two ranks agree on the tensor communication contract?** The experiment is intended to test your prediction.

### Scalar payload

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario scalar
```

Rank 0 sends the integer `42` encoded as `torch.tensor([42], dtype=torch.int64)`; rank 1 receives into a one-element `int64` tensor. The current `scalar_message()` implementation is reliable as a two-rank example. Its rank-ID extension sends additional messages when launched with more than two ranks, but does not currently post matching receives for all of them; do not use the four-rank form until that code is completed.

### Many senders, one receiver

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.send_recv --scenario many_to_one
```

Protocol:

```text
Rank 1 sends [1] ──┐
Rank 2 sends [2] ──┼──> Rank 0 receives explicitly from 1, then 2, then 3
Rank 3 sends [3] ──┘
```

Rank 0 loops over source ranks and passes each source to `receiver(src, buffer)`. The sending workers may reach `send()` in any order. Rank 0’s explicit source order controls which matching message it receives at each step. The current helper prints the received list and `many_to_one()` also prints its scalar value, so each payload appears in two output lines. Lines from different processes may interleave; line order is not a correctness guarantee.

Explore: if rank 3 sends first while rank 0 is receiving from rank 1, what does rank 0’s `recv(src=1)` do? What changes if rank 0 receives from any source rather than naming one?

## 5. Matching operations form a protocol

A send is not a broadcast. It names a destination; the matching receive names a source.

```text
Rank 0                                      Rank 1
send(tensor, dst=1) ─────────────────────> recv(buffer, src=0)
```

For the intended transfer, both processes must agree on the peers and compatible tensor expectations. If one side never performs its operation, the other may wait. If each side waits for an operation that the other side will not perform, the job may deadlock.

### What “blocking” means here

`dist.send()` and `dist.recv()` are blocking calls from the caller’s point of view: the Python process does not advance past the call until that operation has made the required progress or raised an error. The exact buffering details belong to the backend; do not assume a small send always waits for the receiver to enter `recv()` or always completes immediately.

This is why a deadlock is a **protocol** problem. Each rank’s local code may look reasonable, but the combined sequence can have no operation that lets the ranks make progress.

## 6. Bidirectional exchange and deadlock reasoning

Code: [bidirectional.py](./bidirectional.py).

### Ordered exchange: run this first

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.bidirectional --scenario ordered
```

```text
Rank 0                                Rank 1
send [10] ─────────────────────────> recv into buffer
recv [20] <───────────────────────── send [20]
```

Rank 0 sends and then receives. Rank 1 receives first and then sends. Expected observations: rank 0 receives `[20]`; rank 1 receives `[10]`.

### Naive send-first exchange

Predict first, then try this only with an outer timeout:

```bash
gtimeout 15s uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.bidirectional --scenario naive
```

Both ranks call blocking `send()` before `recv()`:

```text
Rank 0 sends to rank 1 and waits?     Rank 1 sends to rank 0 and waits?
                    ↖             ↙
                       cycle
```

If each send waits for its peer to post a receive, neither process reaches `recv()`. The small messages may be buffered by a backend, so successful completion does not prove the ordering is generally safe. `gtimeout` is GNU coreutils’ timeout command; on macOS it is commonly named `gtimeout`. If unavailable, use a test runner or another external process timeout rather than running an intentionally unsafe pattern without a bound.

Interview habit: draw both ranks’ next operation side by side. Ask, “What event lets each blocked operation complete?” If the answer depends on an operation that the peer cannot reach, look for a wait cycle.

## 7. Ring topology and one exchange

A ring is a **logical** neighbor relationship. For `world_size = N`:

```python
next_rank = (rank + 1) % world_size
prev_rank = (rank - 1 + world_size) % world_size
```

The modulo wraps the first and last rank together. For four ranks:

```text
       0 ───> 1
       ▲      │
       │      ▼
       3 <─── 2
```

This drawing describes who sends to whom. It does not claim that the computer’s physical network has the same shape.

Code: [ring.py](./ring.py), functions `ring_exchange()`, `broken_ring_exchange()`, and `safe_ring_exchange()`.

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario exchange
```

Each rank starts with `torch.tensor([rank])`. After one step, the previous rank’s value is in the local receive buffer:

| Rank | Sends its value to | Receives from | Expected received value |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 3 | 3 |
| 1 | 2 | 0 | 0 |
| 2 | 3 | 1 | 1 |
| 3 | 0 | 2 | 2 |

`ring_exchange()` uses only blocking point-to-point calls. Rank 0 sends first. Every other rank first receives from its predecessor, then sends its own input to its successor. The last rank’s send matches rank 0’s receive. This explicit rank-ordered chain avoids the all-ranks-send-first circular wait and works for odd and even ring sizes.

`safe_ring_exchange()` repeats that safe order as a separate comparison:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario safe
```

`broken_ring_exchange()` has every rank send first, then receive. Its behavior can depend on message size and backend buffering. Do not use its output as proof that send-first is safe. Bound the run:

```bash
GLOO_SOCKET_IFNAME=lo0 gtimeout 15s uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario broken
```

The command can exit with a timeout or finish if the backend buffers the small payload. The point is to reason about the protocol, not to benchmark Gloo buffering.

## 8. Circulate values over multiple steps

Code: `circulate()` in [ring.py](./ring.py).

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario circulate
```

Unlike a single exchange, each process sends the value it most recently received on the next iteration. With four ranks:

```text
step 0:  R0=0  R1=1  R2=2  R3=3
step 1:  R0 gets 3, R1 gets 0, R2 gets 1, R3 gets 2
step 2:  R0 gets 2, R1 gets 3, R2 gets 0, R3 gets 1
step 3:  R0 gets 1, R1 gets 2, R2 gets 3, R3 gets 0
```

Therefore rank 0 records `[0, 3, 2, 1]`. Every rank calls the same safe ring exchange at each step. Three steps are enough for every rank to see its own value plus the other three values in this four-rank example.

Try changing `steps` when calling `circulate()` from a worker program. Predict what values have arrived after zero, one, two, and `world_size - 1` steps.

## 9. Ring-gather capstone

An all-gather-like result means every rank ends up with one tensor from every source rank. This implementation deliberately builds that result without calling a collective. It uses only `irecv()` and `isend()`.

Code: `ring_gather()` in [ring.py](./ring.py).

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ring --scenario gather
```

Each rank begins with one local shard:

```text
R0 owns [0]    R1 owns [1]    R2 owns [2]    R3 owns [3]
```

The algorithm keeps one result slot per **source rank**. In each round, a rank sends the shard it currently holds clockwise and receives a shard from its previous neighbor. It stores the incoming shard in the slot for that shard’s original source. With four ranks, the indices move like this:

| Round | Rank 0 sends / receives | Rank 1 sends / receives | Rank 2 sends / receives | Rank 3 sends / receives |
| ---: | --- | --- | --- | --- |
| 0 | send 0 / receive 3 | send 1 / receive 0 | send 2 / receive 1 | send 3 / receive 2 |
| 1 | send 3 / receive 2 | send 0 / receive 3 | send 1 / receive 0 | send 2 / receive 1 |
| 2 | send 2 / receive 1 | send 3 / receive 2 | send 0 / receive 3 | send 1 / receive 0 |

After `world_size - 1` rounds, every rank has every source’s shard. The list is ordered as source ranks `[0, 1, 2, 3]`, regardless of the order each rank received them.

Why nonblocking operations here? Each rank posts a receive and send before waiting. If all ranks used blocking sends first, they could form a circular wait. `isend()` and `irecv()` return work handles. The code waits for both before using the receive buffer or moving to a round that may reuse the sent tensor. The send and receive buffers are still owned by their local process; no tensor object is shared.

Protocol contract: every rank must supply a tensor with a compatible shape and dtype. This function allocates each receive buffer with `empty_like(local_tensor)`; it does not negotiate arbitrary metadata with other ranks.

Senior-level exploration: draw the movement for five ranks, check the index formulas by hand, and state how many messages each rank sends and receives. Then consider how a production implementation would account for message size, topology, memory bandwidth, and failure handling.

## 10. Nonblocking communication

Code: `async_ring_exchange()` in [async_comm.py](./async_comm.py).

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 -m phase1.async_comm
```

The function follows this sequence:

```text
allocate local receive buffer
          ↓
irecv(buffer, source=previous rank) ─┐
isend(value, destination=next rank) ─┴─ communication may be in flight
          ↓
do tiny independent CPU work
          ↓
wait for receive and send work handles
          ↓
read received tensor / reuse send tensor safely
```

`isend()` and `irecv()` return work handles that represent operations in progress. A returned handle does not mean the tensor data is ready to read or safe to overwrite. The function calls `.wait()` on both handles before returning. It also keeps the original send tensor alive and unchanged through the wait.

This CPU demonstration teaches execution order, not performance. The included computation is intentionally tiny; do not infer GPU communication/computation overlap or speedup from this MacBook run.

Compare blocking and nonblocking event order with:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.async_comm --scenario timings
```

`compare_timestamps()` reports separate timestamps for a blocking exchange and a nonblocking exchange. The blocking computation starts only after the blocking communication call returns; in the nonblocking path, the tiny CPU computation is placed after communication launch and before `.wait()`. The values use each process's own monotonic clock, so compare event order within one rank, not absolute timestamps across ranks. These are execution-semantics observations, not a benchmark.

## 11. Message ordering and variable names

Code: `disagreeing_order()` in [ordering.py](./ordering.py).

Rank 0 sends A, then B. Rank 1 can choose which local buffer it passes to the first receive:

```bash
# Rank 1 receives into B_buffer first
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ordering

# Rank 1 receives into A_buffer first
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ordering --receive-order A-first
```

With the default B-first receive order, rank 1’s first receive gets the first message sent (A’s payload), even though that receive writes into `B_buffer`. The second receive gets B’s payload and writes it into `A_buffer`. The variable names are meaningful only to the local Python program; they are not message labels.

The function always returns `(A_buffer, B_buffer)`. With A-first, expect `A_buffer=[1], B_buffer=[2]`; with B-first, expect `A_buffer=[2], B_buffer=[1]`.

### Optional tags

Run the tagged-message experiment with:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.ordering --scenario tags
```

Rank 0 launches `[10]` with tag 10 and `[20]` with tag 20. Rank 1 receives tag 20 first, then tag 10. The tag is part of the match request, so rank 1 can request a particular logical message. It does not communicate the tensor's shape or dtype. The example uses nonblocking sends so rank 0 can launch both tagged messages before waiting; otherwise, a blocking send for tag 10 could wait while rank 1 is asking for tag 20.

This optional experiment depends on tag support in the installed backend. If it errors, keep the main phase focused on peer, operation order, and tensor contract rather than assuming tags are portable across every backend/version.

## 12. Bounded failure experiments

Code: [failures.py](./failures.py). The process group has a ten-second timeout, and work-handle waits use eight seconds. Still use an outer command or test timeout: distributed timeouts help diagnose missing progress but are not a universal instant-kill mechanism.

Example:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 -m phase1.failures --scenario sender_never_sends
```

Run `wrong_source` with three ranks; other scenarios require two:

```bash
uv run -- torchrun --nnodes=1 --nproc-per-node=3 --master-addr=127.0.0.1 --master-port=29500 -m phase1.failures --scenario wrong_source
```

| Scenario | Protocol problem illustrated | Expected observation |
| --- | --- | --- |
| `receiver_never_receives` | Rank 0 sends to rank 1, which deliberately never posts the matching receive | The send work should time out; buffering can affect the exact backend error |
| `sender_never_sends` | Rank 1 waits for rank 0, which deliberately does not send | The receive work times out |
| `wrong_source` | Rank 0 waits specifically for rank 2 while rank 1 sends to rank 0 | Rank 0’s receive from rank 2 does not match rank 1’s message |
| `shape_mismatch` | Sender provides four floats; receiver provides room for three | Gloo reports an incompatible message/buffer contract or the operation fails |
| `dtype_mismatch` | Sender uses two `float32`; receiver expects two `int64` values | The byte-size/tensor contract is incompatible; observe the installed backend’s error |
| `rank_exits_early` | Rank 1 exits without responding while rank 0 waits | The launcher reports worker failure and/or rank 0’s receive fails |
| `circular_wait` | Both ranks issue a large send and wait before posting receive | Send work should time out if the backend cannot complete without matching receives |

Every trace record is JSON and includes `scenario`, `rank`, `pid`, `source`, `destination`, `operation`, `shape`, `dtype`, and `timestamp`. Compare ranks’ traces to see which side was waiting for what. The generic helper `trace_event()` in [distributed.py](./distributed.py) instead records one `peer` field for its send/receive event format.

Run one scenario at a time. Some outcomes depend on backend buffering or failure timing; the code reports an unexpected completion instead of claiming every invalid protocol must fail in exactly the same way on every PyTorch build.

## 13. Code map: concept to implementation

| Concept | Code to read | Command or test |
| --- | --- | --- |
| Group setup, cleanup, trace records | [distributed.py](./distributed.py) | Any `torchrun` command below |
| One blocking send and receive | [send_recv.py](./send_recv.py): `sender`, `receiver` | `-m phase1.send_recv` |
| Many-to-one explicit sources | [send_recv.py](./send_recv.py): `many_to_one` | `-m phase1.send_recv --scenario many_to_one` with 4 ranks |
| Ordered and naive bidirectional protocols | [bidirectional.py](./bidirectional.py) | `--scenario ordered` or bounded `--scenario naive` |
| One ring step and safe ordering | [ring.py](./ring.py): `ring_exchange`, `safe_ring_exchange` | `-m phase1.ring --scenario exchange` or `safe` |
| Potential send-first ring wait | [ring.py](./ring.py): `broken_ring_exchange` | `-m phase1.ring --scenario broken` with external timeout |
| Forward received values repeatedly | [ring.py](./ring.py): `circulate` | `-m phase1.ring --scenario circulate` |
| Gather-like algorithm with P2P only | [ring.py](./ring.py): `ring_gather` | `-m phase1.ring --scenario gather` |
| Nonblocking ring transfer | [async_comm.py](./async_comm.py): `async_ring_exchange` | `-m phase1.async_comm` |
| Blocking vs. nonblocking event sequence | [async_comm.py](./async_comm.py): `compare_timestamps` | `-m phase1.async_comm --scenario timings` |
| Receive order vs. buffer variable name | [ordering.py](./ordering.py): `disagreeing_order` | `-m phase1.ordering [--receive-order A-first]` |
| Tag-based message matching | [ordering.py](./ordering.py): `tagged_messages` | `-m phase1.ordering --scenario tags` |
| Bounded failure cases | [failures.py](./failures.py): `run_scenario` | `-m phase1.failures --scenario NAME` |

## 14. Tests and what they prove

The unit tests run without launching a distributed job:

```bash
uv run --group dev pytest
```

The real multi-process integration tests use `torchrun` subprocesses. They are opt-in so a normal `pytest` run does not launch many workers:

```bash
PHASE1_RUN_DISTRIBUTED=1 uv run --group dev pytest tests/test_integration.py
```

On macOS, include the loopback interface setting:

```bash
GLOO_SOCKET_IFNAME=lo0 PHASE1_RUN_DISTRIBUTED=1 uv run --group dev pytest tests/test_integration.py
```

The integration helper selects a free port, uses `127.0.0.1`, and sets Gloo's interface based on the OS. Tests check basic/scalar/many-to-one messages, the ordered exchange, receive ordering and tagged matching, async ring exchange and timing-event order, blocking ring behavior, circulation, ring gather, distinct worker PIDs, and one bounded invalid protocol. The invalid-protocol test expects a nonzero worker exit. Each subprocess has an outer timeout.

The `naive` bidirectional and `broken` ring cases are not pass/fail correctness tests: their visible behavior can depend on buffering. Predict and inspect them as bounded experiments.

## 15. Questions to answer before an interview

Do not memorize one-line definitions. Use the code to reason through each rank’s next operation and draw tensor movement.

1. What is point-to-point communication?
2. How is `send`/`recv` different from shared memory?
3. Why does a receiver allocate a tensor buffer?
4. What must sender and receiver agree on about a message?
5. What could happen if a sender never gets a matching receive?
6. What could happen if a receiver waits for a sender that never sends?
7. Why can two ranks that both send first form a deadlock?
8. What does a deterministic communication order change?
9. What is the difference between blocking and nonblocking calls from the caller’s point of view?
10. What does an `isend()` or `irecv()` work handle tell you? What does it not tell you?
11. Why must you wait before reading a receive buffer or reusing a send tensor?
12. What message-order assumptions does the ordering exercise rely on?
13. How would you diagnose a rank that appears stuck while others printed output?
14. What is a ring topology, and how does modulo arithmetic identify neighbors?
15. How many steps does it take in this ring for a rank to see every source’s shard?
16. Draw `ring_gather()` for four ranks and explain each round’s send and receive indices.
17. What is the difference between logical topology and physical network topology?
18. What costs might make a manually written P2P algorithm slower or more fragile than a tuned library operation?
19. Why does understanding these primitives help you reason about higher-level distributed training later?
20. Why are many distributed failures protocol/control-flow bugs rather than arithmetic bugs?

## 16. Senior MLE interview checkpoint

Try answering aloud after the exercises. The prompts intentionally do not include model answers.

> Four ranks form a ring. Rank `i` sends to `(i + 1) % world_size`. What can happen if every rank uses blocking `send()` first and `recv()` second? Does a small-message run that finishes prove the pattern is safe?

> Rank 0 has a tensor with one billion floats and wants every other rank to obtain it. Could repeated point-to-point sends do this? What would you measure or optimize?

> Rank 2 crashes while rank 1 is waiting in `recv(src=2)`. What does rank 1 observe, and what would you inspect in the job logs?

> Describe a ring-based gather without calling a collective. What is sent at each step, and how does a rank know which source slot to fill?

> Explain the caller-visible difference between `dist.send()` and `dist.isend()`. Which tensors must stay alive, and when is the receive buffer safe to read?

> When might asynchronous communication help a training step? What evidence would you need before claiming overlap improved end-to-end throughput?

> A job works with two ranks but hangs with four. Which rank-specific logs, protocol assumptions, and tensor contracts would you inspect first?

## 17. A practical study routine

For each scenario, keep a small notebook entry:

```text
Scenario:
Ranks and world size:
Each rank's initial tensor:
Operation order on each rank:
Expected matching peer and buffer:
Observed output/error:
What I learned about the protocol:
```

The goal is not simply to make the command exit. The goal is to be able to draw the protocol, predict where each process can block, explain the tensor movement, and identify which evidence would distinguish a communication mismatch from a normal algorithm bug.

## Phase 2: collective communication

Phase 2 builds on the process and point-to-point exercises above. See the [Phase 2 guide](src/phase2/README.md) for broadcast, reduce, barrier, subgroups, protocol ordering, and the capstone. The communication-critical code is intentionally left as `# TODO: IMPLEMENT` exercises.
