# PyTorch Distributed Phase 2: Collectives

This phase builds on [Phase 0](../phase0/README.md) (processes, ranks, groups,
`torchrun`, and Gloo) and [Phase 1](../phase1/README.md) (point-to-point
messaging and ring-gather). The implementations are complete, so the notes can
be read alongside runnable examples. Draw what every process does before
running an example, then compare that prediction with its logs.

## 1. Goal

**Code:** [process-group lifecycle and structured logging](./distributed.py),
[all Phase 2 modules](./), and [the capstone](./algorithms.py).

Understand the global problem each collective solves, who participates, where the result appears, what synchronization is involved, and how point-to-point messages could realize the same semantics. This is not an API memorization exercise.

Environment: Python 3.11+, CPU tensors, PyTorch distributed, Gloo, local `torchrun`. No accelerator or later-phase collectives are needed.

From the repository root:

```bash
uv sync --group dev
```

The commands below use the macOS loopback interface, `lo0`. On Linux, replace
`GLOO_SOCKET_IFNAME=lo0` with `GLOO_SOCKET_IFNAME=lo`.

Run examples with bounded process-group timeouts. Real distributed tests launch
subprocesses and are opt-in through the repository-wide `RUN_DISTRIBUTED`
switch. The same variable applies to every phase. For example, the
[native broadcast tests](../../tests/test_broadcast.py) run with:

```bash
RUN_DISTRIBUTED=1 uv run pytest tests/test_broadcast.py
```

Each distributed test also places an outer timeout around `torchrun`, because
a process-group timeout does not catch every invalid collective protocol.

## 2. From point-to-point to collectives

**Code:** [manual broadcast](./manual_broadcast.py),
[manual reduction](./manual_reduce.py), and
[shared distributed helpers](./distributed.py).

Phase 1 transferred between named peers:

```text
Rank 0 ───────────────> Rank 1
          send/recv
```

Phase 2 uses coordinated operations over a process group:

```text
          Collective operation
               Rank 0
              /  |  \
          Rank 1 Rank 2 Rank 3
```

A collective is not “rank 0 calls a function that modifies other ranks.” Every member enters a compatible operation; the runtime coordinates communication; the operation's semantics determine where results appear.

## 3. Broadcast semantics

**Code:** [native broadcast demonstration](./broadcast_demo.py) and
[broadcast integration tests](../../tests/test_broadcast.py).

Broadcast distributes one rank's tensor to all members of the group. The `src` rank supplies the input; all ranks, including source, call the collective. Source's tensor remains available and each non-source tensor buffer receives its contents. `src=0` is a convention in examples, not a special capability of rank zero.

The [broadcast demonstration](./broadcast_demo.py) begins with source value
100 and placeholders elsewhere.

Run a four-rank broadcast from rank zero:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29621 \
  -m phase2.broadcast_demo \
  --src 0
```

Before the collective, rank zero has `[100]` and the other ranks have `[-1]`.
After it, every rank should have `[100]`.

Run the same experiment with rank two as the source:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29622 \
  -m phase2.broadcast_demo \
  --src 2
```

Rank two starts with `[999]`; after the collective, every rank should have
`[999]`.

Finally, delay rank two by three seconds and compare the before/after
timestamps:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29623 \
  -m phase2.broadcast_demo \
  --src 0 \
  --delay-rank 2 \
  --delay-seconds 3
```

A participant that has not entered can hold up completion. Collective
completion semantics are backend/operation-specific; do not equate broadcast
with a barrier.

## 4. Manual broadcast

**Code:** [`manual_broadcast`](./manual_broadcast.py) and its command-line
driver in the same module.

`manual_broadcast(tensor, src)` uses only blocking `dist.send` and `dist.recv`.
All world ranks call the function. Non-root processes allocate compatible
buffers before calling. For P ranks and N bytes, the source sends `P - 1`
copies and originates `(P - 1) * N` bytes. If rank zero transmits a 10 GB
tensor directly to 1,023 peers, it must originate 10,230 GB of data and perform
1,023 blocking sends.

Use a fixed, agreed tensor shape and dtype. Since this function receives a tensor buffer rather than a shape descriptor, non-root buffers must already have the correct layout.

Run the direct point-to-point implementation with:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29624 \
  -m phase2.manual_broadcast \
  --algorithm manual \
  --src 0 \
  --value 100
```

The before records contain `[100]` on R0 and `[-1]` elsewhere. Every after
record contains `[100]`.

## 5. Tree broadcast

**Code:** [`tree_broadcast`](./manual_broadcast.py) and the
[broadcast benchmark](./broadcast_benchmark.py).

`tree_broadcast` handles source zero and a power-of-two world size. It doubles
the set of informed ranks in each round. For four ranks:

```text
step 0: R0 has X
step 1: R0 ──> R1       (R0,R1 have X)
step 2: R0 ──> R2; R1 ──> R3
final:  R0 R1 R2 R3 have X
```

Extending the drawing to eight ranks gives three tree rounds, compared with
seven sequential sends from the source in naive blocking fan-out. This is the
communication-depth argument; the measured localhost result is discussed
below.

Run the four-rank tree implementation with:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29625 \
  -m phase2.manual_broadcast \
  --algorithm tree \
  --src 0 \
  --value 100
```

### Broadcast benchmark

**Code:** [benchmark runner and measurement logic](./broadcast_benchmark.py).

The [benchmark runner](./broadcast_benchmark.py) compares direct fan-out and
binary-tree implementations after process-group initialization. It performs
warmups, checks the received tensor, and reports the slowest rank's elapsed
time for each trial.

Run four ranks with a one-million-element `float32` tensor (about 4 MB):

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29630 \
  -m phase2.broadcast_benchmark \
  --algorithm both \
  --elements 1000000 \
  --warmups 2 \
  --trials 5
```

Repeat with `--nproc-per-node=2`, `4`, and `8`, using a fresh port for each
run. Also try several payload sizes such as `1`, `10000`, and `1000000`
elements. The tree still sends `P - 1` messages overall, but distributes those
sends across informed ranks and reduces communication depth from roughly
`P - 1` source sends to `log2(P)` rounds.

Small localhost runs may show direct fan-out winning because process
scheduling and extra tree coordination can outweigh its theoretical benefit.
Treat this as an algorithm experiment rather than evidence about multi-node
GPU performance. Native `dist.broadcast` uses backend-specific optimized
algorithms and should be the production comparison.

#### Detailed local experiment

**Code:** [trial synchronization, timing, validation, and reporting](./broadcast_benchmark.py).

The experiment varied both dimensions that affect the algorithms:

- World sizes: 2, 4, and 8 local Gloo ranks.
- Payloads: 1, 4, 16, and 64 MiB of `float32` data.
- Two independent `torchrun` launches for every world-size/payload pair.
- Three warmups followed by 15 measured trials in each launch, giving 30
  measured samples per algorithm in each table row.
- The first launch ran direct fan-out first; the second ran tree first. This
  reduces systematic bias from always benchmarking one algorithm first.
- Each trial synchronized ranks before starting. Reported time is the maximum
  local duration across ranks, because completion is limited by the slowest
  participant.

The table reports pooled median latency and the 10th-to-90th percentile range.
Positive improvement means tree was faster; negative improvement means direct
fan-out was faster.

| Ranks | Payload | Direct median [p10–p90] | Tree median [p10–p90] | Tree improvement |
|---:|---:|---:|---:|---:|
| 2 | 1 MiB | 0.220 ms [0.200–0.245] | 0.224 ms [0.206–0.295] | -1.8% |
| 2 | 4 MiB | 0.641 ms [0.627–0.689] | 0.644 ms [0.615–0.688] | -0.5% |
| 2 | 16 MiB | 3.099 ms [2.837–3.278] | 3.103 ms [2.852–3.591] | -0.1% |
| 2 | 64 MiB | 5.040 ms [4.946–7.131] | 5.096 ms [4.973–7.180] | -1.1% |
| 4 | 1 MiB | 0.554 ms [0.302–0.625] | 0.570 ms [0.286–0.670] | -2.9% |
| 4 | 4 MiB | 1.849 ms [1.096–1.903] | 1.975 ms [1.864–2.073] | -6.8% |
| 4 | 16 MiB | 4.266 ms [3.931–5.383] | 4.344 ms [3.872–6.061] | -1.8% |
| 4 | 64 MiB | 17.140 ms [15.895–20.186] | 18.814 ms [16.132–21.743] | -9.8% |
| 8 | 1 MiB | 0.770 ms [0.647–1.214] | 0.657 ms [0.593–1.250] | +14.7% |
| 8 | 4 MiB | 2.511 ms [2.260–4.085] | 2.366 ms [2.173–4.663] | +5.8% |
| 8 | 16 MiB | 10.625 ms [9.816–11.470] | 12.664 ms [11.952–15.093] | -19.2% |
| 8 | 64 MiB | 40.984 ms [38.024–43.466] | 59.602 ms [53.528–65.621] | -45.4% |

#### Does this support the hypothesis?

**Code:** [the two algorithms being compared](./manual_broadcast.py) and
[the benchmark harness](./broadcast_benchmark.py).

Only partially. The two-rank case is a sanity check: both implementations
perform the same single `R0 -> R1` transfer, and their medians differ by at
most 1.8%. At eight ranks, the tree improved median latency for the 1 and 4
MiB payloads. That is consistent with the expected benefit of replacing seven
sequential source sends with three communication rounds.

The result reverses for larger payloads. At eight ranks, tree was 19.2% slower
for 16 MiB and 45.4% slower for 64 MiB. All ranks run on one host, so the tree's
simultaneous sends compete for the same loopback transport, memory bandwidth,
CPU time, and memory-copy resources. The topology has no independent network
links for the tree to exploit. Forwarding a large tensor through intermediate
ranks can therefore add contention rather than increase useful aggregate
bandwidth.

The measurements support a narrower conclusion: on this machine, the tree can
reduce latency for small messages as rank count grows, while direct fan-out is
better for the tested large messages. They do not prove that tree broadcast is
universally faster. Testing the distributed-systems hypothesis requires
multiple hosts with independent links, repeated runs, and comparison with the
backend's optimized `dist.broadcast` implementation.

## 6. Reduce semantics

**Code:** [native reduce demonstration](./reduce_demo.py) and
[native reduce integration tests](../../tests/test_reduce.py).

Reduce combines each member's corresponding tensor elements using an operator and leaves the result at the designated destination. Everyone participates. For inputs `[1,10]`, `[2,20]`, `[3,30]`, `[4,40]`, SUM to rank zero gives `[10,100]`. Do not depend on non-destination buffers after reduce.

Try `SUM`, `MAX`, `MIN`, and `PRODUCT` where backend/dtype supports them. For `1,5,3,2`, predict each result first. Reduction operators are generally expected to be associative for regrouping into efficient trees. SUM is central to aggregating gradients and metrics (we study only the semantics needed here).

Run the native SUM reduction with:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29640 \
  -m phase2.reduce_demo \
  --operator SUM \
  --dst 0
```

R0 begins with `[1, 10]` and finishes with `[10, 100]`. Only R0's final
buffer is part of the reduction result.

## 7. Manual reduction

**Code:** [direct and tree reduction implementations](./manual_reduce.py) and
[manual reduction integration tests](../../tests/test_manual_reduce.py).

`manual_reduce` and `tree_reduce` implement reduction using point-to-point
messages. Both default to `SUM` and also accept `MAX`, `MIN`, and `PRODUCT`.
The older `manual_reduce_sum` and `tree_reduce_sum` names remain as SUM-only
wrappers. Every rank must use the same operator, tensor shape, and dtype.

For four ranks containing `[1]`, `[2]`, `[3]`, and `[4]`:

| Operator | Destination result | Elementwise combine |
|---|---:|---|
| `SUM` | `[10]` | `accumulator + incoming` |
| `MAX` | `[4]` | larger corresponding value |
| `MIN` | `[1]` | smaller corresponding value |
| `PRODUCT` | `[24]` | `accumulator * incoming` |

These operators are associative, so partial results may be grouped into a tree.
Integer PRODUCT can overflow for large values, and backend/dtype support should
always be checked when using native collectives.

### Direct SUM dry run: four-rank execution

**Code:** [`manual_reduce`](./manual_reduce.py).

Suppose each rank starts with `rank + 1` and the destination is rank zero:

```text
R0=[1]  R1=[2]  R2=[3]  R3=[4]
```

All four processes call the same function, but their `rank` makes them follow
different branches:

| Execution step | Rank behavior | Destination accumulator |
|---:|---|---:|
| 1 | R0 clones its local `[1]` into `result` | `[1]` |
| 2 | Loop reaches source 0; R0 skips receiving from itself | `[1]` |
| 3 | R1 sends `[2]`; R0 receives it and runs `result.add_([2])` | `[3]` |
| 4 | R2 sends `[3]`; R0 receives and adds it | `[6]` |
| 5 | R3 sends `[4]`; R0 receives and adds it | `[10]` |
| 6 | R0 copies `result` back into its input tensor | `[10]` |

Ranks 1, 2, and 3 return after their blocking sends complete. Their local
tensors remain `[2]`, `[3]`, and `[4]` in this implementation, but reduce
semantics only guarantee the final result at the destination.

Run this example with:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29650 \
  -m phase2.manual_reduce \
  --algorithm manual \
  --operator SUM \
  --dst 0
```

The CLI enables a detailed trace by default. Each line begins with the rank and
shows when it waits, sends, receives, updates its accumulator, or finishes.
Because workers write concurrently, lines from different ranks can appear in a
different order between runs. Add `--quiet` when only the structured
before/after records are needed.

### Tree SUM dry run: eight-rank execution

**Code:** [`tree_reduce`](./manual_reduce.py).

The tree version currently requires destination zero and a power-of-two world
size. With rank-local values `[1]` through `[8]`, every active receiver holds a
partial sum. A sender transfers its partial sum exactly once and then exits the
loop so its contribution cannot be counted again.

```text
Initial: R0=1 R1=2 R2=3 R3=4 R4=5 R5=6 R6=7 R7=8

step=1: R1 -> R0, R3 -> R2, R5 -> R4, R7 -> R6
        R0=3,       R2=7,       R4=11,      R6=15

step=2: R2 -> R0, R6 -> R4
        R0=10,      R4=26

step=4: R4 -> R0
        R0=36
```

Here is how the loop condition produces those pairs:

| `step` | Sender condition | Senders | Destinations | Partial sums afterward |
|---:|---|---|---|---|
| 1 | `rank % 2 == 1` | 1, 3, 5, 7 | 0, 2, 4, 6 | R0=3, R2=7, R4=11, R6=15 |
| 2 | `rank % 4 == 2` | 2, 6 | 0, 4 | R0=10, R4=26 |
| 4 | `rank % 8 == 4` | 4 | 0 | R0=36 |

Trace rank 6 through the code: at `step=1`, it is a receiver, calculates
`source=7`, receives `[8]`, and changes its tensor from `[7]` to `[15]`. At
`step=2`, `6 % 4 == 2`, so it sends `[15]` to rank 4 and breaks. Rank zero
never becomes a sender; it receives and accumulates at every round until it
holds `[36]`.

Run the eight-rank tree example with:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=8 \
  --master-addr=127.0.0.1 \
  --master-port=29651 \
  -m phase2.manual_reduce \
  --algorithm tree \
  --operator SUM \
  --dst 0
```

For the tree run, the trace also prints the round number, `step`, partner rank,
payload, updated partial sum, and the point where each sender leaves the tree.

To try another operator, keep the same topology and change only `--operator`:

```bash
# Four ranks reduce [1], [2], [3], [4] to MAX=[4] at rank 2.
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29654 \
  -m phase2.manual_reduce \
  --algorithm manual \
  --operator MAX \
  --dst 2
```

Omitting `--operator` selects `SUM`.

### What complexity improves?

**Code:** [direct and tree implementations used in the comparison](./manual_reduce.py).

Let `P` be the number of ranks and `N` the tensor size.

| Property | Direct reduction | Tree reduction |
|---|---:|---:|
| Sequential communication depth | `P - 1`, or `O(P)` | `log2(P)`, or `O(log P)` |
| Total messages | `P - 1`, or `O(P)` | `P - 1`, or `O(P)` |
| Total tensor data transferred | `(P - 1) * N`, or `O(PN)` | `(P - 1) * N`, or `O(PN)` |
| Receives/combine operations performed by root | `P - 1` | `log2(P)` |

For eight ranks, direct reduction makes the destination receive seven tensors
sequentially. The tree needs three rounds because `log2(8) = 3`. Under an ideal
model where independent pairs communicate concurrently, the critical-path
communication changes from approximately
`(P - 1) * (latency + N / bandwidth)` to
`log2(P) * (latency + N / bandwidth)`.

The total message count and total bytes do not improve: both algorithms still
send seven tensors for eight ranks. The tree improves communication depth and
distributes the selected combine operation and network work across ranks.
Actual speedup depends on whether the hardware can run the same-round transfers
concurrently; ranks on one laptop often compete for shared CPU and memory
bandwidth.

## 8. Barrier

**Code:** [barrier timing demonstration](./barrier_demo.py) and
[barrier integration tests](../../tests/test_barrier.py).

`dist.barrier()` makes each member wait until all members of its group reach
that point. It does not copy tensors or make Python objects identical. The
[barrier demonstration](./barrier_demo.py) delays ranks by 0, 1, 2, and 4
seconds and records before/after timestamps and wait duration. Rank zero
arrives first but proceeds only when rank three arrives at about four seconds.
A slow worker can make every faster worker idle at a synchronization point:
the straggler problem.

Run the four-rank barrier demonstration from the repository root:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29670 \
  -m phase2.barrier_demo
```

On Linux, replace `GLOO_SOCKET_IFNAME=lo0` with
`GLOO_SOCKET_IFNAME=lo`. Rank zero arrives first and should wait roughly four
seconds, while rank three arrives last and should wait very little.

## 9. Synchronization vs communication

**Code:** [barrier](./barrier_demo.py), [broadcast](./broadcast_demo.py), and
[reduce](./reduce_demo.py) demonstrations.

Broadcast and reduce move/transform application data. Barrier establishes a coordination point. For example, rank-local tensors `[0]`, `[1]`, `[2]`, `[3]` stay different after a barrier. Synchronization is not data exchange.

## 10. Process groups

**Code:** [pair-subgroup creation and broadcasts](./process_groups.py) and
[process-group integration tests](../../tests/test_process_groups.py).

### Why have groups when WORLD already exists?

**Code:** [`create_pair_groups` and `run_pair_broadcasts`](./process_groups.py).

Start with four independent processes launched by `torchrun`. After
`dist.init_process_group`, each process has a unique global rank and all four
belong to the default group, `WORLD`:

```text
WORLD = {R0, R1, R2, R3}
```

A collective is defined over a group. If these ranks call a WORLD broadcast,
all four must participate and all four receive the same source value. That is
correct when the operation concerns the entire job, but many distributed
algorithms need communication among only selected workers.

For example, suppose R0/R1 are one data-parallel replica and R2/R3 are another:

```text
WORLD   = {R0, R1, R2, R3}  # setup, global coordination, global rank space
Group A = {R0, R1}          # one independent collective sequence
Group B = {R2, R3}          # another independent collective sequence
```

Using WORLD for the pair broadcasts would couple unrelated ranks: R2 and R3
would have to enter R0's broadcast, and one WORLD broadcast could not deliver
100 to the first pair while independently delivering 200 to the second pair.
The two subgroups provide separate participation scopes. A collective on Group
A requires only R0 and R1; R2 and R3 do not call it or wait for it.

Process groups do not create processes, share Python memory, or automatically
choose a physical network topology. They define which existing ranks
participate in a collective and provide the communication context in which
collective ordering must agree. WORLD remains useful as the initial group and
global rank namespace from which smaller groups are created.

### Pair-group assumptions and execution

**Code:** [logged pair-group demonstration](./process_groups.py).

This exercise intentionally assumes exactly four world ranks. Every world rank
calls `dist.new_group([0, 1])` and then `dist.new_group([2, 3])` in the same
order. Consistent creation order ensures that all processes agree about the two
communication contexts. After creation, each rank calls a broadcast only on
the group to which it belongs.

The `src` argument is a global rank even when `group` limits participation:

| Rank | Subgroup | Initial tensor | Subgroup source | Final tensor |
|---:|---|---:|---:|---:|
| 0 | A `{0,1}` | `[100]` | R0 | `[100]` |
| 1 | A `{0,1}` | `[-1]` | R0 | `[100]` |
| 2 | B `{2,3}` | `[200]` | R2 | `[200]` |
| 3 | B `{2,3}` | `[-1]` | R2 | `[200]` |

Run the logged demonstration with:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29680 \
  -m phase2.process_groups
```

On Linux, use `GLOO_SOCKET_IFNAME=lo`. The logs show WORLD initialization,
group membership, source ranks, and tensors before and after each subgroup
broadcast. In larger training systems, groups commonly represent data,
tensor, pipeline, or expert-parallel communication scopes.

## 11. Collective ordering

**Code:** [ordering and participation failure experiments](./failures.py).

Within one process group, every rank must execute compatible collectives in the
same order. Think of the group as one distributed program whose instruction
pointer is copied across processes. The first collective called by every rank
must describe one compatible operation, then the second collective must do the
same, and so on.

For example, this sequence is valid:

| Collective slot | R0 | R1 | R2 | R3 |
|---:|---|---|---|---|
| 0 | Broadcast | Broadcast | Broadcast | Broadcast |
| 1 | Barrier | Barrier | Barrier | Barrier |

The processes need not enter a slot at exactly the same instant. A fast rank
may wait for a slow rank. They must eventually enter the same protocol with
compatible arguments.

## 12. Failure modes

**Code:** [bounded negative experiments](./failures.py) and
[process-group timeout setup](./distributed.py).

`failures.py` deliberately breaks four parts of that protocol. These are
negative experiments, so a nonzero exit, timeout, or apparent hang is the
expected result. Run them one at a time from the repository root. Each command
uses a different rendezvous port so a previous failed launch cannot collide
with the next one.

```bash
# macOS; use GLOO_SOCKET_IFNAME=lo on Linux.
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29690 \
  -m phase2.failures ordering

GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29691 \
  -m phase2.failures missing

GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29692 \
  -m phase2.failures shape

GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29693 \
  -m phase2.failures dtype
```

The process group is configured with an eight-second timeout. That bounded the
ordering and missing-participant runs on the tested macOS/Gloo environment,
but it did not stop either metadata-mismatch run. Use an outer subprocess
timeout in automation. When running manually, press `Ctrl-C` if a negative
experiment remains stuck.

### How to read the failure output

**Code:** [the four failure branches](./failures.py).

Four workers write to the same terminal, so tracebacks can be interleaved and
their order can change between runs. Read the output in this order:

1. Find the first rank-level exception. It is usually closest to the actual
   failed collective.
2. Identify the collective named by its Python stack frame, such as
   `dist.broadcast` or `dist.barrier`.
3. Read the Gloo message. `Timed out waiting ... for send/recv` describes what
   the transport was waiting for; it does not mean the Python code explicitly
   called `dist.send` or `dist.recv`.
4. Treat the final `ChildFailedError` as `torchrun` reporting that at least one
   child process failed. It is a launcher summary, not the root cause.
5. If the run was manually interrupted, the final `SignalException` with
   signal 2 means `torchrun` received `SIGINT` from `Ctrl-C`. It says how the
   experiment was stopped, not why the collective became stuck.

### Case 1: incompatible collective order (`ordering`)

**Code:** [`ordering` scenario](./failures.py).

The code creates this protocol:

| Collective slot | R0 | R1 | R2 | R3 |
|---:|---|---|---|---|
| 0 | Broadcast | Broadcast | Barrier | Broadcast |
| 1 | Barrier | Barrier | Broadcast | Barrier |

At slot zero, R0, R1, and R3 wait for a broadcast involving the whole WORLD
group. R2 waits for a WORLD barrier. Neither operation can collect all four
participants, and no rank can advance to repair the mismatch because each is
blocked in its current operation.

The observed rank-level errors after about eight seconds included:

```text
rank 0: dist.broadcast(...) -> Timed out waiting 8000ms for send operation
rank 2: dist.barrier(...)   -> Timed out waiting 8000ms for recv operation
ranks 1 and 3: dist.barrier(...) -> Timed out waiting 8000ms for recv operation
```

R1 and R3 entered `broadcast` first, but their traceback points at the later
`barrier`. A collective may return at different moments on different ranks;
the eventual traceback is where that rank noticed the broken protocol, not
necessarily where the bug began. The full sequence across all ranks is the
unit that must be inspected.

**Failure mode:** distributed deadlock until the backend timeout converts it
to exceptions and `torchrun` terminates the job.

**Why it matters:** real training loops often contain rank-dependent branches.
If one branch changes collective order, the whole job can stop even though
every individual process is still alive.

**Key lesson:** all members of a process group must follow one globally
consistent collective sequence. Log the rank, group, operation, and sequence
number around collectives when diagnosing an ordering bug.

### Case 2: missing participant (`missing`)

**Code:** [`missing` scenario](./failures.py).

R0, R1, and R2 call a barrier on WORLD. R3 skips it:

| Rank | Action |
|---:|---|
| 0 | Enter WORLD barrier |
| 1 | Enter WORLD barrier |
| 2 | Enter WORLD barrier |
| 3 | Skip barrier and continue to cleanup |

A four-rank barrier records arrivals until its group membership condition is
satisfied: four arrivals are required. Three arrivals can never complete it.
R3 calling `destroy_process_group()` does not count as arriving at the barrier.

The observed errors after about eight seconds were:

```text
ranks 0 and 1: dist.barrier() -> Timed out waiting 8000ms for recv operation
rank 2:        dist.barrier() -> Timed out waiting 8000ms for send operation
torchrun: ChildFailedError
```

Which rank reports a send or receive timeout is an implementation detail of
the barrier algorithm. The useful fact is that every traceback points to the
same incomplete WORLD barrier.

**Failure mode:** the participating ranks wait forever in the abstract
protocol; the configured backend timeout eventually aborts this run.

**Why it matters:** an early return, exception, exhausted data loader, or
rank-only code path can silently remove one worker from a later collective and
strand every remaining worker.

**Key lesson:** group membership defines an obligation to participate. If only
a subset should communicate, create and use a subgroup instead of conditionally
skipping a WORLD collective.

### Case 3: incompatible tensor shape (`shape`)

**Code:** [`shape` scenario](./failures.py).

Broadcast is in-place. It does not first broadcast a Python tensor description
and allocate a matching result. Every non-source rank supplies its own receive
buffer before entering the collective. The ranks therefore need a shared
contract for element count, shape, dtype, device, source rank, and group.

This experiment violates the element-count part of the contract:

| Rank | Buffer before broadcast from R0 |
|---:|---|
| 0, 1, 3 | four `float32` elements (16 bytes) |
| 2 | eight `float32` elements (32 bytes) |

Conceptually, R0 offers one 16-byte payload while R2 participates with a
32-byte destination. The collective has no application-level rule saying
whether R2 should receive four values, wait for eight values, resize itself,
or preserve its remaining values.

On the tested macOS/Gloo build, this run printed no rank-level exception and
did not terminate after 15 seconds, even though the process-group timeout was
eight seconds. It had to be interrupted. The final log was the launcher's
`SignalException: ... signal: 2`, caused by that interrupt. Other PyTorch
versions, backends, and devices may instead report an error or fail in another
way.

**Failure mode:** undefined collective protocol behavior, observed here as a
hang-like stall that was not converted into a useful backend timeout.

**Why it matters:** shapes can diverge through uneven batches, rank-dependent
model paths, or incorrect padding. A metadata mistake can look like a network
problem because the communication layer only sees incompatible buffers.

**Key lesson:** validate or establish tensor metadata before the collective.
For dynamic data, communicate the metadata first, allocate a compatible
buffer, and then broadcast the payload.

### Case 4: incompatible tensor dtype (`dtype`)

**Code:** [`dtype` scenario](./failures.py).

This experiment gives R0, R1, and R3 one `float32` element but gives R2 one
`int64` element:

| Rank | Logical element count | Dtype | Buffer size |
|---:|---:|---|---:|
| 0, 1, 3 | 1 | `float32` | 4 bytes |
| 2 | 1 | `int64` | 8 bytes |

The matching shape does not make the buffers compatible. A collective
transports tensor storage; it does not perform an implicit numeric cast from
the source dtype to each receiver's dtype. R0 therefore supplies a four-byte
value while R2 declares an eight-byte destination with a different
interpretation.

The observed behavior matched the shape case: no rank-level exception and no
termination after 15 seconds. After `Ctrl-C`, the launcher reported SIGINT
rather than a dtype diagnostic. Backend behavior can differ, so a clean error
must not be relied upon.

**Failure mode:** undefined collective protocol behavior, observed here as an
unbounded stall until the outer launcher was interrupted.

**Why it matters:** mixed precision, integer counters, and model state often
put several dtypes in the same program. Accidentally choosing a different
buffer on one rank can stop the complete job or produce an invalid result on a
backend that does not reject the mismatch.

**Key lesson:** dtype is part of the wire protocol. Convert tensors explicitly
to one agreed dtype before entering the collective.

### Combined observation

**Code:** [all failure scenarios](./failures.py).

| Scenario | Broken assumption | Observed result on this machine | Main prevention |
|---|---|---|---|
| `ordering` | Same operation at each collective slot | Gloo timeout, then `ChildFailedError` | Keep one group-wide operation order |
| `missing` | Every group member participates | Gloo timeout, then `ChildFailedError` | Use the correct group and control flow |
| `shape` | Compatible element count and shape | Stalled past the process-group timeout; manually interrupted | Agree on metadata before payload |
| `dtype` | Compatible dtype and byte layout | Stalled past the process-group timeout; manually interrupted | Cast explicitly to one dtype |

The general rule is to treat a collective as a distributed protocol call.
Its contract includes the process group, position in the collective sequence,
operation, source or destination rank, tensor metadata, and reduction operator
where applicable. Validate those invariants at application boundaries, include
rank and operation context in logs, configure a process-group timeout, and put
an independent outer timeout around negative tests. A timeout limits damage;
it does not make an invalid protocol correct or guarantee a clear diagnosis.

## 13. Training-related examples

**Code:** [mini training coordinator](./algorithms.py),
[collective trace records](./tracing.py), and
[structured rank records](./distributed.py).

- Broadcast: rank zero owns tensor-encoded configuration (seed and step count), then all workers receive it.
- Reduce: each worker contributes loss sum and example count; rank zero gets global totals and computes total loss / total examples. Do not average local means unless every rank has equal example counts.
- Rank-zero logging: reduce fits when only one process needs the aggregate.
- Barrier: establish a phase boundary, understanding that the slowest worker governs progress.

The [tracing module](./tracing.py) defines a small JSON event record (`rank`,
operation, group, tensor shape, timestamp, phase), with no logging framework.

## 14. Capstone: mini coordinator

**Code:** [capstone implementation](./algorithms.py) and
[capstone integration test](../../tests/test_integration.py).

`algorithms.run_capstone()` uses broadcast, barrier, and reduce to coordinate a
small four-rank job. Rank zero starts with seed 123 and 5 steps. Every rank
performs deterministic, rank-varying toy work, all ranks synchronize before
metric aggregation, and rank zero prints the global result.

Run it from the repository root:

```bash
GLOO_SOCKET_IFNAME=lo0 uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29710 \
  -m phase2.algorithms
```

On Linux, use `GLOO_SOCKET_IFNAME=lo`.

### Capstone dry run

**Code:** [`run_capstone`](./algorithms.py).

1. **One-to-many:** R0 starts with configuration `[123, 5]`; R1, R2, and R3
   start with `[0, 0]`. All four ranks call broadcast with `src=0`, after which
   every rank has seed 123 and 5 steps.
2. **Local work:** rank `r` processes `r + 1` examples and produces loss sum
   `(r + 1) * 5`.
3. **Everyone waits:** all ranks enter the barrier. A rank can leave only after
   R0, R1, R2, and R3 have all arrived.
4. **Many-to-one:** every rank packs `[local_loss_sum, local_examples]` into a
   `float64` tensor and calls SUM reduce with `dst=0`.

| Rank | Configuration after broadcast | Local loss sum | Local examples | Reduction contribution |
|---:|---|---:|---:|---|
| 0 | `[123, 5]` | 5 | 1 | `[5, 1]` |
| 1 | `[123, 5]` | 10 | 2 | `[10, 2]` |
| 2 | `[123, 5]` | 15 | 3 | `[15, 3]` |
| 3 | `[123, 5]` | 20 | 4 | `[20, 4]` |

R0 receives the elementwise sum `[50, 10]` and computes `50 / 10 = 5.0`.
Reducing the loss sum and example count is correct even when ranks process
different numbers of examples. Averaging four local means would generally be
wrong because it would give a rank with one example the same weight as a rank
with four examples.

The expected output is:

```json
{"global_mean_loss": 5.0, "total_examples": 10, "world_size": 4}
```

## 15. Paper-and-pencil exercises

**Code to compare with the derivations:** [broadcast](./broadcast_demo.py),
[reduce](./reduce_demo.py), [barrier](./barrier_demo.py),
[manual broadcast](./manual_broadcast.py), and
[manual reduction](./manual_reduce.py).

1. With R0=[5], R1=[9], R2=[2], R3=[7], what does every rank hold after broadcast from rank 2?
2. With those same inputs, what does rank 3 hold after SUM reduce to rank 3? Which rank is guaranteed the final result?
3. Explain the directional difference between broadcast and reduce.
4. Rank 0 reaches a barrier at t=1s and rank 3 at t=8s. When can rank 0 continue, approximately? What does this show about stragglers?
5. With 1,024 ranks and a 1 GB tensor, why could direct root-to-everyone sending be undesirable? What topology may improve scalability?

The completed semantics summary is:

| Operation | Input location | Result location |
|---|---|---|
| Broadcast | Source rank | Every rank in the process group |
| Reduce | Every rank in the process group | Destination rank only |

### Answers

**Code:** [native collective examples](./broadcast_demo.py),
[reduce example](./reduce_demo.py), and [tree broadcast](./manual_broadcast.py).

1. Broadcast uses R2 as the source, so R0, R1, R2, and R3 all finish with
   `[2]`. Every group member participates, including the source.
2. SUM is `5 + 9 + 2 + 7 = 23`, so R3 holds `[23]`. Only the destination R3 is
   guaranteed to hold the reduced result; the other output buffers must not be
   used as though they contain the global sum.
3. Broadcast moves one source rank's value outward to every group member.
   Reduce moves every member's contribution inward, combines the values with
   an operator, and guarantees the result only at the destination.
4. R0 can leave at approximately `t=8s`, after the final participant reaches
   the barrier. It waits about seven seconds. The example shows that a single
   straggler determines when synchronous workers can continue.
5. Direct fan-out makes the root send 1 GB to each of 1,023 peers: 1,023 GB of
   root-originated traffic and 1,023 sequential root sends in the simple
   blocking implementation. A tree broadcast lets informed ranks forward the
   tensor, reducing ideal communication depth from `O(P)` to `O(log P)` rounds.

## 16. Questions and interview checkpoint

**Code reference:** [native broadcast](./broadcast_demo.py),
[manual broadcasts](./manual_broadcast.py), [native reduce](./reduce_demo.py),
[manual reductions](./manual_reduce.py), [barrier](./barrier_demo.py),
[process groups](./process_groups.py), [failure modes](./failures.py), and
[the combined capstone](./algorithms.py).

Answer these after the exercises, without looking at the code:

1. What makes an operation collective? Does one rank calling broadcast cause it automatically? Which ranks participate?
2. What does `src` mean? Is rank zero special? What happens to source and non-source buffers?
3. How can send/recv implement broadcast? Why does naive fan-out scale poorly? How does a tree reduce communication depth?
4. What does reduce do, where is its result, and why are other ranks' output buffers unspecified?
5. What is a reduction operator? Why is associativity useful? Why is SUM common in ML?
6. What does barrier guarantee? Does it move tensors? Why do stragglers matter?
7. Why must ranks use compatible collective order? What can happen if one rank skips or tensors disagree in shape/dtype?
8. What is a process group? Can groups run independent collectives? Why will subgroups matter for tensor, pipeline, and expert parallelism?
9. Why might global metrics use reduce? When would broadcast be natural?
10. How would you choose between a manual send/recv protocol and a native collective?

Senior MLE prompts:

- 64 workers need a seed loaded by rank zero: what pattern fits?
- Every rank counts successful examples and only rank zero logs the global count: what pattern fits and why?
- R0: Broadcast→Reduce; R1: Broadcast→Reduce; R2: Reduce→Broadcast; R3: Broadcast→Reduce. What protocol problem do you expect?
- Why can a low-payload barrier be slow? Why can one slow worker limit synchronous training throughput?
- Why use groups `{0..3}` and `{4..7}`?
- Why might a tree broadcast beat root direct fan-out?
- Predict the semantic difference between Reduce and AllReduce; this phase's
  code provides Reduce as the concrete reference point.
- Draw an eight-rank point-to-point broadcast and an eight-rank reduction tree, showing every round.

Phase 2 readiness check: derive broadcast propagation from R0 to seven peers,
derive reduction in reverse, and draw a barrier timeline with an early rank
waiting for the final arrival. If each drawing can be explained from the code
and its logs, the core Phase 2 model is in place.
