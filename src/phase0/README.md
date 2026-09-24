# PyTorch Distributed: First Principles to Interview Ready

Lecture notes for learning distributed training from the ground up. We begin with ordinary operating-system processes, then build toward the communication patterns used in data-parallel training. The examples use CPU workers and PyTorch's Gloo backend so you can focus on the ideas without needing a GPU.

## What you will learn

By the end, you should be able to explain and demonstrate:

- How a launcher starts workers and gives each one a distributed identity ?
- Why each worker has its own Python memory, model, and data iterator ?
- How a process group connects workers and how collectives move tensor values ?
- Why collective order must match across all participating ranks ?
- How barriers, timeouts, and cleanup fit into a reliable worker lifecycle ?
- How the same ideas support synchronous data-parallel training ?

## 1. Start with processes

A process is a running program with its own memory and Python interpreter. Starting the same Python module four times creates four independent processes. A normal dictionary created in one worker is not visible to the other workers. Each process has its own copy.

Distributed training makes these processes cooperate through explicit communication. A **worker** is one process participating in the job. A **rank** is that worker's integer identity within a process group. With four workers, the global ranks are `0`, `1`, `2`, and `3`.

```text
torchrun
  ├── worker process, rank 0
  ├── worker process, rank 1
  ├── worker process, rank 2
  └── worker process, rank 3
```

The rank is not a process ID. The PID is assigned by the operating system and is useful for debugging. `WORLD_SIZE` is the number of workers in the group.

## 2. Set up the project

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

The lockfile pins dependencies. Running commands through `uv run` uses this project's virtual environment.

## 3. Launching workers with `torchrun`

`torchrun` starts one Python process per worker and sets environment variables that let them find each other. Each process then runs the same module independently. For this module launch, the package is found because the project is installed in the environment.

On Linux, the basic single-machine example is:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m phase0.hello_distributed
```

On macOS, this project's Gloo example may need the loopback interface and an explicit local rendezvous address. The following is the command verified for this repository on macOS:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run torchrun \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m phase0.hello_distributed
```

If port `29500` is already in use, choose another free port. `GLOO_SOCKET_IFNAME` is the network interface Gloo should use (`lo0` on macOS, often `lo` on Linux). IPv6 reverse-lookup warnings can appear even when a local launch succeeds; check whether workers actually reach the program output and whether the command exits successfully.

### The launch environment

Each worker receives values like these:

| Variable | Meaning | Example with two local workers |
| --- | --- | --- |
| `RANK` | Global worker number in the process group | `0` or `1` |
| `LOCAL_RANK` | Worker number on this machine | `0` or `1` |
| `WORLD_SIZE` | Total worker count | `2` |
| `MASTER_ADDR` | Address of the rendezvous host | `127.0.0.1` |
| `MASTER_PORT` | Rendezvous port | `29500` |

On a single machine, global rank and local rank often happen to match. On a multi-machine job, global rank is unique across the whole job while local rank starts over on each machine. Keep the distinction clear in code and interview answers.

## 4. Joining a process group

A **process group** is the set of processes allowed to communicate with one another. In this project, `setup_process_group()` checks that the torchrun environment exists and initializes PyTorch's default group with the Gloo backend.

```python
dist.init_process_group(backend="gloo")
```

Because `torchrun` provides the rendezvous environment variables, PyTorch's default initialization method is `env://`; it does not need to be written explicitly. Initialization is a coordination point: every worker must join the same group before group communication can proceed. PyTorch documents this environment-based setup in its [`torch.distributed` reference](https://docs.pytorch.org/docs/stable/distributed.html#environment-variable-initialization).

Gloo is a communication backend suited to CPU communication and is used here for portability. In GPU training, NCCL is commonly used for CUDA tensor collectives. The right backend depends on devices, platform, and deployment environment.

Call `dist.is_initialized()` before relying on the group. Call `dist.destroy_process_group()` during cleanup so the process-group resources are released. The example cleanup helper is safe when initialization did not happen.

## 5. Rank identity and process-local state

`get_rank_info()` gathers four useful values:

- `rank`: global distributed identity from PyTorch.
- `local_rank`: local worker identity from the launch environment.
- `world_size`: number of workers in the group.
- `pid`: operating-system process ID.

In `hello_distributed.main()`, each worker creates a regular dictionary and changes its own counter using its rank. You should see one record per worker, with distinct rank and PID values. The print order is nondeterministic; rank 1 may print before rank 0.

This is a key distinction: **Python objects are process-local; tensor communication is explicit.** If rank 0 changes its dictionary, rank 1's dictionary does not change. To share training information, workers communicate tensors through collectives or other explicit mechanisms.

## 6. Collectives: the communication protocol

A **collective** is an operation in which a group of ranks participates. Examples include:

- `barrier`: wait until every rank has reached the same point.
- `all_reduce`: combine tensor values across ranks and give the result to every rank.
- `broadcast`: send a tensor from one source rank to all other ranks.
- `all_gather`: collect values from every rank on every rank.

For example, if two workers call `all_reduce` with values 2 and 5 using the sum operation, both receive 7. The values travel because each worker participates in the same collective; an ordinary Python variable is not shared.

### Why gradient averaging works

Suppose each worker processes a different mini-batch and calculates a gradient:

```text
rank 0: g₀        rank 1: g₁
```

An all-reduce sum followed by division by the number of workers produces the mean gradient on each worker:

```text
g = (g₀ + g₁) / 2
```

Each worker can then apply the same optimizer update to its model replica. This is the core communication idea behind synchronous data-parallel training. The full PyTorch DDP module automates gradient synchronization hooks and other details; this exercise builds the underlying mental model first. See [PyTorch's distributed overview](https://docs.pytorch.org/docs/stable/distributed.html).

## 7. Synchronization with a barrier

`barrier_demo()` prints timestamps, sleeps for a rank-dependent duration, then calls `dist.barrier()`. A barrier returns only after all members of the group have entered it. A fast rank waits for a slower rank.

```text
rank 0: sleep 0s ─────── arrive ─────────────── leave
rank 1: sleep 1s ─────────────── arrive ─────── leave
                                  barrier
```

This is useful for understanding synchronization and debugging. It is not a general fix for race conditions, and adding barriers everywhere can slow a job. Put a barrier in a program only when every rank is meant to rendezvous at that point. See the [PyTorch barrier reference](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.barrier).

To observe it, call `barrier_demo(delay_seconds=1)` from a worker function after the process group is initialized. Every rank must call the function.

## 8. Matching collective order and diagnosing hangs

Collectives form a protocol. All ranks in the participating group need to call compatible operations in compatible order. Consider this broken sequence:

```text
rank 0: barrier()
rank 1: all_reduce(tensor)
```

The ranks disagree about what operation comes next. One or both workers may report a distributed error or wait until a timeout. If a worker raises an exception before a collective while its peers enter that collective, the peers can wait too. This is why distributed failures often look like hangs instead of a clean Python exception.

The `mismatched_collective_demo()` function deliberately demonstrates an incompatible sequence with an asynchronous wait limit. Run it only as a multi-process experiment and expect a distributed error. The `cleanup_demo()` function shows process-group teardown in `finally`.

### Debug a job systematically

1. Confirm the launcher started the expected number of processes.
2. Print rank, local rank, world size, and PID early in each worker.
3. Check that all ranks take compatible branches and call collectives in the same order.
4. Find the first rank that stops making progress; inspect its exception and logs.
5. Check rendezvous address, port, and network interface if the group never initializes.
6. Use bounded timeouts in small experiments. A timeout is a diagnostic; it does not repair an inconsistent collective sequence.
7. Ensure cleanup runs when worker code raises, and terminate/restart the full job after a broken collective.

For test runs, the integration test selects a free loopback port and sets the correct loopback interface for macOS or Linux. Run it with:

```bash
PHASE0_RUN_DISTRIBUTED=1 uv run pytest tests/test_integration.py -vv -s
```

The test captures worker output and compares rank/PID/counter sets, not print order.

## 9. The worker lifecycle pattern

The example in `hello_distributed.py` follows the essential shape:

```python
try:
    setup_process_group()
    # Read rank/world information and do distributed work.
finally:
    cleanup_process_group()
```

Every worker runs this lifecycle. `finally` makes cleanup happen after normal work or an exception. In a production training program, the `try` body also includes device setup, data loading, model construction, training, checkpoint handling, and error reporting. Coordinated failure handling needs more than a `finally` block, but reliable teardown is a sound baseline.

## 10. Interview practice

Try answering these aloud before reading the short answers.

**Q: What is the difference between rank, local rank, world size, and PID?**  
Rank identifies a worker globally in its process group. Local rank identifies it on one host, often to select a local device. World size is the total worker count. PID identifies an OS process and is not a distributed identity.

**Q: Does each worker share the same Python model object or dictionary?**  
No. Each process has its own memory and typically its own model replica. Workers communicate selected data explicitly, often with tensor collectives.

**Q: How can workers train equivalent model replicas if their memory is separate?**  
They start from compatible parameters, compute gradients on local data, synchronize gradients (commonly with all-reduce in data parallelism), and apply equivalent optimizer updates.

**Q: Why can a distributed program hang when one worker crashes?**  
Other workers may be waiting in a collective for participation that will never arrive. Timeouts and launcher failure reporting help make this visible, but correct control flow must keep ranks coordinated.

**Q: What does a barrier guarantee?**  
Every member that returns from the barrier has reached that barrier call. It does not mean arbitrary Python state is shared, nor is it a substitute for the correct data communication operation.

**Q: What must be true for an all-reduce?**  
All participating ranks must issue compatible operations in a compatible order, with tensors that satisfy the backend's requirements (including compatible shape and type).

**Q: What does `torchrun` do and what does `init_process_group` do?**  
`torchrun` launches worker processes and supplies rendezvous configuration. `init_process_group` uses that configuration to connect each worker to the distributed group used by collective operations.

## 11. Source map and practice sequence

| File | Concept |
| --- | --- |
| `src/phase0/distributed.py` | Process-group initialization and cleanup |
| `src/phase0/rank_info.py` | Global/local worker identity |
| `src/phase0/hello_distributed.py` | Worker entry point and process-local Python state |
| `src/phase0/synchronization.py` | Arrival times and barriers |
| `src/phase0/failures.py` | Mismatched collectives and cleanup |
| `tests/` | Expected behavior and opt-in multi-process checks |

Recommended order: run the hello example, explain every field in its output, draw the barrier timeline, predict the mismatched-collective outcome, then answer the interview questions without looking. Finally, make one worker take a different branch around a collective and explain why the other workers cannot make progress.
