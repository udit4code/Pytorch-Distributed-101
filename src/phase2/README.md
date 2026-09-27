# PyTorch Distributed Phase 2: Collectives

This phase builds on Phase 0 (processes, ranks, groups, `torchrun`, Gloo) and Phase 1 (point-to-point messaging and ring-gather). Work through each exercise by drawing what every process does before running it. The communication logic is intentionally left as `# TODO: IMPLEMENT` in the code.

## 1. Goal

Understand the global problem each collective solves, who participates, where the result appears, what synchronization is involved, and how point-to-point messages could realize the same semantics. This is not an API memorization exercise.

Environment: Python 3.11+, CPU tensors, PyTorch distributed, Gloo, local `torchrun`. No accelerator or later-phase collectives are needed.

From the repository root:

```bash
uv sync --group dev
```

The commands below use the macOS loopback interface, `lo0`. On Linux, replace
`GLOO_SOCKET_IFNAME=lo0` with `GLOO_SOCKET_IFNAME=lo`.

Run examples with bounded process-group timeouts. The real subprocess tests are opt-in with `PHASE2_RUN_DISTRIBUTED=1 pytest tests/test_broadcast.py`; each subprocess also has an outer timeout. Fill in each TODO, then replace/enable the corresponding scaffold assertions.

## 2. From point-to-point to collectives

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

Broadcast distributes one rank's tensor to all members of the group. The `src` rank supplies the input; all ranks, including source, call the collective. Source's tensor remains available and each non-source tensor buffer receives its contents. `src=0` is a convention in examples, not a special capability of rank zero.

`broadcast_demo.py` begins with source value 100 and placeholders elsewhere.

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

Complete `manual_broadcast(tensor, src)` using only blocking `dist.send` and `dist.recv`. All world ranks call the function. Non-root processes allocate compatible buffers before calling. For P ranks and N bytes, reason about the direct source's send volume: how many copies must it send? What if rank zero must transmit a 10 GB tensor directly to 1,023 peers?

Use a fixed, agreed tensor shape and dtype. Since this function receives a tensor buffer rather than a shape descriptor, non-root buffers must already have the correct layout.

## 5. Tree broadcast

Complete `tree_broadcast` for source zero and power-of-two world size. For four ranks, double the set of informed ranks each round:

```text
step 0: R0 has X
step 1: R0 ──> R1       (R0,R1 have X)
step 2: R0 ──> R2; R1 ──> R3
final:  R0 R1 R2 R3 have X
```

Draw eight ranks and compare the number of communication rounds with naive fan-out. Do not over-optimize or benchmark localhost performance.

### Broadcast benchmark scaffold

`broadcast_benchmark.py` compares the direct fan-out and binary-tree
implementations after process-group initialization. It performs warmups,
checks the received tensor, and reports the slowest rank's elapsed time for
each trial.

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

Reduce combines each member's corresponding tensor elements using an operator and leaves the result at the designated destination. Everyone participates. For inputs `[1,10]`, `[2,20]`, `[3,30]`, `[4,40]`, SUM to rank zero gives `[10,100]`. Do not depend on non-destination buffers after reduce.

Try `SUM`, `MAX`, `MIN`, and `PRODUCT` where backend/dtype supports them. For `1,5,3,2`, predict each result first. Reduction operators are generally expected to be associative for regrouping into efficient trees. SUM is central to aggregating gradients and metrics (we study only the semantics needed here).

## 7. Manual reduction

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

`dist.barrier()` makes each member wait until all members of its group reach that point. It does not copy tensors or make Python objects identical. `barrier_demo.py` delays ranks by 0, 1, 2, and 4 seconds and records before/after timestamps and wait duration. Predict when the fastest rank proceeds. A slow worker can make every faster worker idle at a synchronization point: the straggler problem.

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

Broadcast and reduce move/transform application data. Barrier establishes a coordination point. For example, rank-local tensors `[0]`, `[1]`, `[2]`, `[3]` stay different after a barrier. Synchronization is not data exchange.

## 10. Process groups

### Why have groups when WORLD already exists?

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

Within a group, ranks must execute compatible collectives in compatible order. If some ranks do Broadcast then Reduce while another does Reduce then Broadcast, operations can mismatch, error, or time out. A rank skipping a collective has similar consequences. `failures.py` provides bounded ordering, missing participant, shape, and dtype experiments; backend behavior varies, so protocol violations can produce errors, timeouts, or hang-like behavior.

## 12. Failure modes

Treat every collective as a distributed protocol with shared assumptions: group membership, call order, root, tensor shape, and dtype. Do not assume mismatches always produce a clean Python exception. Use finite process-group and test timeouts. The negative examples are experiments, not production patterns.

## 13. Training-related examples

- Broadcast: rank zero owns tensor-encoded configuration (seed and step count), then all workers receive it.
- Reduce: each worker contributes loss sum and example count; rank zero gets global totals and computes total loss / total examples. Do not average local means unless every rank has equal example counts.
- Rank-zero logging: reduce fits when only one process needs the aggregate.
- Barrier: establish a phase boundary, understanding that the slowest worker governs progress.

`tracing.py` defines a small JSON event record (`rank`, operation, group, tensor shape, timestamp, phase), with no logging framework.

## 14. Capstone: mini coordinator

Complete `algorithms.run_capstone()` using only broadcast, barrier, and reduce. Rank zero starts with seed 123 and 5 steps; every rank performs deterministic, rank-varying toy work; all synchronize before metric aggregation; rank zero prints JSON with world size, total examples, and global mean loss. The scaffold uses counts 1,2,3,4 and per-rank loss sums `(rank+1)*5`, so expected totals are 10 examples and mean loss 5.0. Explain the flow as one-to-many, everyone waits, many-to-one.

## 15. Paper-and-pencil exercises

1. With R0=[5], R1=[9], R2=[2], R3=[7], what does every rank hold after broadcast from rank 2?
2. With those same inputs, what does rank 3 hold after SUM reduce to rank 3? Which rank is guaranteed the final result?
3. Explain the directional difference between broadcast and reduce.
4. Rank 0 reaches a barrier at t=1s and rank 3 at t=8s. When can rank 0 continue, approximately? What does this show about stragglers?
5. With 1,024 ranks and a 1 GB tensor, why could direct root-to-everyone sending be undesirable? What topology may improve scalability?

Fill in before looking up the semantics:

| Operation | Input location | Result location |
|---|---|---|
| Broadcast | ? | ? |
| Reduce | ? | ? |

## 16. Questions and interview checkpoint

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
- Predict the semantic difference between Reduce and AllReduce. Do not implement any later-phase operation.
- Draw an eight-rank point-to-point broadcast and an eight-rank reduction tree, showing every round.

Mandatory derivations before Phase 2 is complete: draw broadcast propagation from R0 to seven peers; draw reduction in reverse; draw a barrier timeline with an early rank waiting for the last arrival. Do not move to Phase 3 until you can derive these patterns on paper.
