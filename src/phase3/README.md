# Phase 3: Why synchronous data parallel SGD needs AllReduce

Start with single-process SGD. For one minibatch, compute
`g = grad_theta L(theta, B)` and `theta_(t+1) = theta_t - eta * g`.
With four workers and different local batches, rank `r` computes
`g_r = grad_theta L(theta, B_r)`.
If the local batches are equal sized
and each local loss is a mean, the global-batch mean gradient is
`g = (g_0 + g_1 + g_2 + g_3) / 4`.
Every rank has its own model replica and independently runs `optimizer.step()`.
So every rank needs the same aggregate gradient: the communication shape is
many-to-everyone. That is exactly the result AllReduce provides.

This phase intentionally leaves the learning-critical operations unfinished.
Each exercise function has a detailed `TODO: IMPLEMENT` comment; its tests
specify behavior and should fail until the corresponding exercise is done.

## Start here: a live four-process AllReduce

Run every command from the repository root. This exercise uses four independent
Python processes on the same machine. `torchrun` starts them and assigns each a
rank: `0`, `1`, `2`, or `3`. The ranks join one Gloo process group before they
communicate. Each process runs the same Python module, but `dist.get_rank()`
returns a different rank in each process.

### First principles: what should the result be?

The code creates one local tensor per process. Rank `r` contributes
`[r + 1, 10 * (r + 1)]`, so the four independent inputs are:

```text
rank 0: [1, 10]
rank 1: [2, 20]
rank 2: [3, 30]
rank 3: [4, 40]
```

An AllReduce with SUM adds values at matching tensor positions across all
processes and makes the completed sum available to every process:

```text
first position: 1 + 2 + 3 + 4       = 10
second position: 10 + 20 + 30 + 40 = 100
```

Thus, after the collective, **each rank** should hold `[10, 100]`. It is not
just rank 0 receiving the answer. Each process has a local tensor; the
collective overwrites that tensor with the global sum. For synchronous data
parallel training, this is useful because each process needs the same combined
gradient before it updates its own model replica.

### Run the demonstration

Make sure PyTorch is installed in the active Python environment, then run:

```bash
GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.all_reduce_demo
```

Each process prints a JSON record before and after AllReduce. In the records
whose `"phase"` is `"after"`, check that ranks 0 through 3 all report
`"values": [10, 100]`. The records may appear in a different order because the
processes print independently.

### Test the live four-process behavior

The integration test launches a real `torchrun` subprocess; it does not mock
`torch.distributed`. From the repository root, run:

```bash
RUN_DISTRIBUTED=1 python -m pytest -q tests/test_all_reduce.py::test_all_reduce_demo_returns_expected_sum_on_every_rank
```

This test checks that the command exits successfully, observes output from all
four ranks, and verifies every rank's result is `[10, 100]`. If pytest is not
installed in the active environment, install it with `python -m pip install
pytest` and retry. A normal `python -m pytest` run skips tests marked as real
distributed integration tests unless `RUN_DISTRIBUTED=1` is set.

The remaining Phase 3 exercises are intentionally unfinished. Their tests
describe expected behavior, but those tests are expected to fail until you
implement the corresponding TODOs.

## 1. Goal

Derive the need for AllReduce from the SGD update, implement it from simpler
communication primitives, then compare a manual distributed update against a
single-process reference. No DDP, FSDP, CUDA, or NCCL is used in this phase.

## 2. From Reduce to AllReduce

**Reduce** sends all contributions to one destination. **AllReduce** first
reduces and then makes the result available on every rank. Ask yourself: why
is Reduce alone insufficient when every process owns an independent model
replica and will execute its own optimizer step? Write down your answer after
the manual experiment.

## 3. AllReduce semantics

Every participant contributes a tensor; after the collective, every participant
holds the elementwise reduction result. Exercise 1 starts with rank-local values
`[1,10]`, `[2,20]`, `[3,30]`, `[4,40]`; SUM must leave `[10,100]` everywhere.
Experiment with SUM, MAX, MIN, and PRODUCT on `1,5,3,2`, predicting first.

## 4. Why SGD needs AllReduce

Each rank sees a different minibatch and therefore generally computes a
different gradient. Synchronous replicated SGD needs the aggregate gradient on
each replica before the local optimizer step. AllReduce matches that shape.

## 5. Gradient averaging

First define what each rank has computed. Let rank `r` process `n_r` examples,
and let `grad_(r,i)` be the gradient from its `i`th example. If each rank's
loss is the mean over its local examples, its local gradient is also a mean:

```text
g_r = (grad_(r,1) + grad_(r,2) + ... + grad_(r,n_r)) / n_r
```

### Equal local batch sizes

Suppose there are `P` ranks and every rank processes the same number `n` of
examples. The global batch contains `P * n` examples. Its mean gradient is
the sum of all per-example gradients divided by `P * n`:

```text
g_global
  = (sum of every rank's per-example gradients) / (P * n)
  = (g_0 + g_1 + ... + g_(P-1)) / P
```

So the implementation has two steps: AllReduce SUM the rank-local mean
gradients, then divide the result by `P` (the world size). Every rank receives
the same average. This matches the gradient from one process computing mean
loss over the concatenated global batch. Averaging is necessary because a SUM
would be `P` times larger and would change the SGD update size for the same
learning rate.

This equal-rank average assumes equal local sample counts and mean-reduced
local losses. Before implementing it, predict what happens if those assumptions
do not hold.

### Paper-and-pencil derivation 1: equal batches

For two equally sized local batches, each local mean gradient gives every
example within that batch weight `1 / |B_0|`. Since the global batch has twice
as many examples, each local mean contributes half of the global mean:

`g_global = grad(L_(B_0 union B_1)) = (grad(L_B0) + grad(L_B1)) / 2 = (g_0 + g_1) / 2`.

For `P` equal-sized rank-local batches, generalize this to
`g_global = (1 / P) * sum(g_r for r = 0..P-1)`.

Then draw four replicas each calling `optimizer.step()` and explain why each
needs the same gradient.

## 6. Unequal local batches

If local sample counts differ, each rank's mean represents a different number
of examples. Give each local mean gradient weight proportional to its count.
For rank `r`, multiply its local mean by `n_r` to recover its local sum:

```text
local_gradient_sum_r = n_r * g_r
```

Sum these local gradient sums across ranks, then divide by the total number of
examples:

```text
g_global
  = sum(n_r * g_r for r = 0..P-1) / sum(n_r for r = 0..P-1)
```

For example, say four ranks process `4, 8, 2, and 6` examples. Suppose their
local mean gradients (shown here as scalars for easy arithmetic) are `1, 2, 3,
and 8`. Then:

```text
weighted sum = 4*1 + 8*2 + 2*3 + 6*8 = 86
sample count = 4 + 8 + 2 + 6 = 20
global mean gradient = 86 / 20 = 4.3
```

The unweighted rank mean would be `(1 + 2 + 3 + 8) / 4 = 3.5`, which is
different because it gives each rank equal weight instead of each example
equal weight. In code, AllReduce the weighted gradient sums and AllReduce the
sample counts, then divide the former by the latter. Both collectives must be
called by every rank in the same order.

An empty rank contributes a zero gradient sum and count zero. Its local mean
gradient is undefined, so use a zero tensor for its contribution. The total
sample count across the process group must still be positive.

### Run the gradient averaging check

This launches four Gloo processes. The equal-batch check uses local gradients
`1, 2, 3, 4`, so each rank should report `2.5`. The weighted check uses counts
`0, 1, 2, 3` and local means `0, 2, 4, 6`; its global result is `14/3` on every
rank. Rank 0 has no examples, so the check also verifies that its placeholder
gradient is zeroed before aggregation.

```bash
RUN_DISTRIBUTED=1 python -m pytest -q tests/test_gradient_sync.py::test_equal_and_unequal_gradient_aggregation
```

The test parses each rank's JSON record and checks both results. To run the
same worker check directly, use:

```bash
GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.gradient_sync --self-check
```

### Paper-and-pencil derivation 2

Without looking above, derive the weighted formula from the definition of each
rank's local mean. Explain why an empty rank contributes zero and why a globally
empty batch must be rejected.

## 7. Manual AllReduce

The module implements two ways to assemble AllReduce. Both use four processes
and the same rank-local input tensors:

```text
rank 0: [1, 10]
rank 1: [2, 20]
rank 2: [3, 30]
rank 3: [4, 40]
```

The elementwise SUM is `[10, 100]`. After either function returns, every rank's
input tensor should contain `[10, 100]`.

### Method A: Reduce, then Broadcast

`naive_all_reduce_sum` asks the distributed backend to do two collectives. All
ranks call the same operations in the same order:

1. `dist.reduce(..., dst=0, op=SUM)` combines all four tensors at rank 0. Rank
   0's tensor becomes `[10, 100]`; only rank 0 is guaranteed to have the sum.
2. `dist.broadcast(..., src=0)` copies rank 0's tensor to every other rank.
   Now ranks 0, 1, 2, and 3 all hold `[10, 100]`.

The backend performs the reduction and broadcast communication. This version
is short and directly demonstrates that AllReduce can be composed from the
Phase 2 Reduce and Broadcast operations.

### Method B: point-to-point send and recv

`point_to_point_all_reduce_sum` builds both phases explicitly using only
blocking `send` and `recv`. Rank 0 is the root in this example:

1. Rank 0 copies its `[1, 10]` into an accumulator.
2. Ranks 1, 2, and 3 each send their original tensor to rank 0. Rank 0 receives
   one sender at a time and adds it to its accumulator:

   ```text
   start:          [1, 10]  (rank 0's contribution)
   after rank 1:   [3, 30]
   after rank 2:   [6, 60]
   after rank 3:   [10, 100]
   ```

3. Rank 0 copies `[10, 100]` into its input tensor and sends that result to
   ranks 1, 2, and 3. Each peer receives the result into its own tensor.
4. The function returns on each rank with `[10, 100]`.

The non-root ranks block in their send until rank 0 receives their contribution,
then block in their receive until rank 0 sends the sum. Rank 0 receives every
contribution before sending results. These matched operations provide the
ordering, so this protocol does not need a barrier.

### How the implementations differ

| Reduce + Broadcast | Point-to-point |
| --- | --- |
| Calls `dist.reduce` and `dist.broadcast`. | Calls only `dist.send` and `dist.recv`. |
| The backend performs the reduction and distributes the result. | Rank 0 explicitly receives, adds, and sends tensors. |
| Shorter and closer to the collective API. | More code; exposes the message flow and ordering directly. |
| Demonstrates composition of collectives. | Demonstrates that AllReduce can be built as a communication algorithm. |

Run both methods on four real processes and compare each result against native
AllReduce:

```bash
GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.manual_all_reduce --self-check
```

Each rank prints a JSON record with `"phase": "passed"` if both manual
results equal `[10, 100]`. The focused pytest integration test is:

```bash
RUN_DISTRIBUTED=1 python -m pytest -q tests/test_manual_all_reduce.py::test_manual_reduce_broadcast_and_point_to_point_match_native
```

Do not call Reduce, Broadcast, or AllReduce inside the point-to-point function;
the native AllReduce is used only by the self-check as the expected reference.

## 8. Data partitioning

Data parallel training requires both replicated model state and partitioned
input data. Inspect the `DistributedSampler` indices for `0..15` over four
ranks. With a divisible dataset and no shuffle, expect no overlap and complete
coverage. For non-divisible lengths, understand sampler padding or `drop_last`.

With shuffling, run two epochs and inspect rank 0's indices with and without
`sampler.set_epoch(epoch)`. The same seed plus epoch should make each epoch
reproducible while changing its ordering.

## 9. Model replication

Compare same-seed model construction on every rank with different seeds. Ask:
if gradients are averaged but replicas start from different weights, do they
necessarily become identical? Broadcast rank 0's parameters and verify exact
agreement before training. Parameter broadcast here intentionally excludes
buffers; consider what a model with mutable buffers would require.

## 10. Manual distributed SGD

Build the deterministic regression dataset and MLP. Each rank gets a distinct
shard, computes local mean loss and gradients, AllReduces each present gradient,
divides by world size for equal shard sizes, and calls plain SGD. No DDP hooks.
Run with:

```bash
torchrun --standalone --nproc-per-node=4 -m phase3.distributed_sgd
```

`--sync-gradients=false` is an experiment: identical initial replicas see
different data, step on different gradients, and diverge. Parameter consistency
is checked every training step.

## 11. Correctness against single-process SGD

Compare one process on a global batch of 16 with four processes on four shards
of four examples. Compare gradients before stepping and parameters after
stepping within a stated floating-point tolerance. This equivalence assumes the
same initial parameters, examples, loss reduction, optimizer, learning rate,
and update timing. Data placement changes; the mathematical update should not.

### Paper-and-pencil derivation 3

Explain why Reduce alone cannot provide the needed result to four independent
replicas. What changes if a rank has zero samples, or if loss reduction is sum?

## 12. Communication cost

Use the rough communication model `T ≈ α + βn`,
where `alpha` is startup latency, `beta` is time per byte, and `n` is payload
size. Why could a
thousand tiny collectives cost more than fewer larger ones? This motivates
gradient buckets; it is not a precise network performance model.

## 13. Stragglers

Delay one rank before a collective and inspect timestamps. Synchronous progress
must wait for every participant, so iteration time is constrained by the
slowest worker. Why can adding workers stop improving throughput as local
computation shrinks while communication and synchronization remain?

## 14. Async communication

Launch `all_reduce(..., async_op=True)`, do independent CPU work, then call
`wait()` before consuming the result. What could go wrong if the tensor is read
before completion? Gloo on localhost is for semantic learning here; do not
infer useful performance overlap from the toy timing.

## 15. Why naive gradient synchronization is inefficient

One AllReduce per parameter is easy to understand but can create many small
collectives. Count calls for a model with 100 parameter tensors. Optionally
flatten a small set of gradients, AllReduce one buffer, then restore them.
Compare many small messages with fewer larger messages; do not add autograd
hooks or attempt full DDP bucketing.

## 16. Failure modes

Try mismatched collective ordering: rank 0 calls A then B while rank 1 calls B
then A. A backend matches protocol order, not Python variable names. Try a
missing rank and a short process-group timeout. These are destructive to that
worker group, so run only in bounded subprocesses. Logs should identify rank,
PID, operation, tensor shape, and step. The straggler exercise is a slow but
valid collective; ordering and missing-rank exercises are protocol failures.

Also compare parameter synchronization after independent optimizer steps with
gradient synchronization before the step. Test plain SGD, then reason about
momentum and Adam: optimizer state and update ordering affect equivalence.

## 17. Capstone

The capstone uses an `nn.Sequential(Linear(4,16), ReLU(), Linear(16,1))`,
deterministic synthetic regression data, rank-0 parameter broadcast, local
shards, manual gradient SUM and correct averaging, and plain SGD. Verify
replicas after each step and compare to the single-process reference on the
same effective global batches. Rank 0 emits one JSON training record per step,
including global loss, world size, local batch size, and global batch size.

Exercise 16: for world size 8, local batch size 16, and 4 accumulation steps,
derive the effective global batch size. Relate the result to how often the
gradient is averaged; accumulation implementation is an optional extension.

## 18. Senior MLE interview checkpoint

Answer these without looking at code:

- Four GPUs process 32 examples each. What is the effective batch size?
- Why AllReduce rather than Reduce when each GPU owns a replica?
- What is AllReduced in classic DDP training? Are parameters AllReduced every iteration?
- What goes wrong if each worker steps before synchronizing gradients?
- If workers see identical data, under which assumptions could they omit synchronization, and why would that defeat data parallelism?
- What happens when rank 3 is 500 ms slower each iteration?
- Why can throughput stop scaling as world size grows?
- Why are per-parameter collectives inefficient, and what do buckets address?
- When could asynchronous AllReduce help? Why not launch all communication only at backward's end?
- Distinguish local gradient SUM from MEAN across workers.
- How should aggregation change for different local batch sizes?
- Why is distributed sampling a correctness concern as well as a performance choice?

## Definition of done

Without DDP, implement `torchrun → identical replicas → different local
minibatches → forward/backward → AllReduce → identical global gradients →
optimizer.step() → identical replicas`. Numerically prove that one process on a
global batch and `P` processes on equivalent local batches produce the same
SGD update under matching assumptions. The key reasoning is mathematical:
synchronous replicated SGD requires each rank to obtain the same aggregate
gradient, and AllReduce has precisely those semantics.
