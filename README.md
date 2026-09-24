# pytorch-distributed-phase0

A first-principles, CPU-only learning project for PyTorch distributed training. It uses `torchrun` and the `gloo` backend; no CUDA or other distributed framework is needed. The central implementation steps are intentionally left as `TODO: IMPLEMENT` exercises. The tests describe expected behavior and are intentionally incomplete/failing until you fill in those steps.

## Setup

Use Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/), sync the project and its test dependency into a local virtual environment:

```bash
uv python pin 3.11
uv sync
```

`uv.lock` records the resolved dependency versions. Run project commands inside the environment with `uv run`, for example:

```bash
uv run pytest
uv run torchrun --standalone --nproc-per-node=4 -m phase0.hello_distributed
```

On Apple Silicon, uv installs the compatible PyTorch wheel for the selected Python version; this project uses CPU/Gloo and does not require CUDA.

## Exercise 1: process-group lifecycle

Implement setup and cleanup in `src/phase0/distributed.py`. Let `torchrun` provide rendezvous details, and initialize the default group with Gloo. Calling setup outside a `torchrun` environment should give a useful error. Cleanup should tolerate a group that was never initialized.

## Exercise 2: rank identity

Implement `get_rank_info()` in `src/phase0/rank_info.py`. Run rank and world-size lookups through `torch.distributed`; read `LOCAL_RANK` from the environment and the PID from the OS.

## Exercise 3: launch workers

From the repository root, run:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run torchrun \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m phase0.hello_distributed
```

Complete the lifecycle in `hello_distributed.py`. Output order is nondeterministic. Each worker should print its rank information and its own counter value. A rank is a distributed worker identity; it is not a PID. In a one-node launch, local ranks commonly match global ranks, but they describe different things.

## Exercise 4: process-local memory

The hello exercise creates a normal Python dictionary in every worker. Give each process a distinct counter value, such as 100 plus its rank. Observe that each independent OS process has its own dictionary.

> If these processes do not share Python memory, how can distributed training synchronize gradients?

Write down your hypothesis before opening the discussion below.

<details>
<summary>Discussion</summary>

Distributed training communicates values between processes using collectives such as all-reduce. For example, ranks can sum gradients and divide by the world size so each worker applies the same averaged gradient. The communication is explicit; ordinary Python objects remain process-local.

</details>

## Exercise 5: barrier and synchronization

Implement `barrier_demo()` in `src/phase0/synchronization.py`. Run one worker longer than another before the barrier. Compare the timestamps: a rank that reaches the barrier early waits until every rank in the group has entered that barrier.

## Exercise 6: incompatible operations and cleanup

Use `src/phase0/failures.py` to build a small, bounded experiment in which ranks execute incompatible collective sequences. Observe the resulting timeout or distributed error. A collective is a protocol shared by the group: every rank must participate in compatible operations in a compatible order. Also exercise cleanup from a `finally` block.

## Tests

Run unit tests with:

```bash
pytest
```

The core tests are expected to fail while the implementation TODOs remain. After completing the exercises, opt into the real multi-process integration check with:

```bash
PHASE0_RUN_DISTRIBUTED=1 pytest tests/test_integration.py
```

The integration check compares rank sets and distinct PIDs; it does not depend on output order.
