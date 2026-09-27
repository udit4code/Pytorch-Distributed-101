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
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.all_reduce_demo
```

Each process prints a JSON record before and after AllReduce. In the records
whose `"phase"` is `"after"`, check that ranks 0 through 3 all report
`"values": [10, 100]`. The records may appear in a different order because the
processes print independently.

### Test the live four-process behavior

The integration test launches a real `torchrun` subprocess; it does not mock
`torch.distributed`. From the repository root, run:

```bash
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_all_reduce.py::test_all_reduce_demo_returns_expected_sum_on_every_rank
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
local losses. When sample counts differ, use the count-weighted derivation below.

### Derivation 1: global-batch gradient for equal local batches

Let each of two ranks process `n` examples. For an example `i`, write its loss
as `ell_i` and its gradient as `grad_i`. The mean loss on the concatenated
batch is

```text
L_global = (sum(ell_i over B_0) + sum(ell_i over B_1)) / (2*n)
```

Regroup the two sums by dividing each by `n`:

```text
L_global
  = (1/2) * [ (sum(ell_i over B_0) / n)
            + (sum(ell_i over B_1) / n) ]
  = (L_0 + L_1) / 2
```

Differentiation is linear, so differentiating this equality gives

```text
grad(L_global) = (grad(L_0) + grad(L_1)) / 2 = (g_0 + g_1) / 2
```

For `P` ranks with the same local batch size `n`, the global batch has `P*n`
examples. Repeating the same regrouping gives

```text
L_global = (L_0 + L_1 + ... + L_(P-1)) / P
g_global = (g_0 + g_1 + ... + g_(P-1)) / P
```

AllReduce SUM gives every rank the numerator; dividing by `P` gives every
rank `g_global`. Every replica needs it because each process owns its own
parameters and optimizer and will independently run `optimizer.step()`. If
only rank 0 received the sum, the other ranks would step with stale or local
gradients and their model copies would diverge.

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
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_gradient_sync.py::test_equal_and_unequal_gradient_aggregation
```

The test parses each rank's JSON record and checks both results. To run the
same worker check directly, use:

```bash
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.gradient_sync --self-check
```

### Derivation 2: global-batch gradient for unequal local batches

Suppose rank `r` has `n_r` examples and computes a local mean gradient:

```text
g_r = (1 / n_r) * sum(grad_(r,i) for i = 1..n_r)
```

Multiply both sides by `n_r`; this recovers that rank's sum of per-example
gradients:

```text
n_r * g_r = sum(grad_(r,i) for i = 1..n_r)
```

The global mean includes every example once, then divides by the total example
count. Substituting the local sums gives

```text
g_global
  = [sum over ranks of (n_r * g_r)] / [sum over ranks of n_r]
```

For example, with 2 examples on rank 0 and 8 on rank 1, suppose their local
mean gradients are `g_0 = 2` and `g_1 = 8`. The correct example-weighted result
is `(2*2 + 8*8) / (2+8) = 68/10 = 6.8`. The unweighted rank mean is
`(2+8)/2 = 5`, which gives a rank with 2 examples the same influence as the
rank with 8 examples.

An empty rank has no defined local mean. Treat its contribution as a zero
gradient sum and a sample count of zero. The global denominator must be
positive; if every rank is empty, the global mean is undefined and the step
must be rejected.

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
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.manual_all_reduce --self-check
```

Each rank prints a JSON record with `"phase": "passed"` if both manual
results equal `[10, 100]`. The focused pytest integration test is:

```bash
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_manual_all_reduce.py::test_manual_reduce_broadcast_and_point_to_point_match_native
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
necessarily become identical? The update is `theta_r_new = theta_r_old - eta*g`.
If every rank receives the same `g` but starts from a different `theta_r_old`,
the same update is added to different starting values; gradient averaging does
not remove the initial difference. Broadcast rank 0's parameters first so every
replica starts from the same state. Parameter broadcast here intentionally
excludes buffers; consider what a model with mutable buffers would require.

The real four-rank self-check verifies same-seed equality, observes that
different seeds produce divergence, broadcasts rank 0's parameters, then
verifies equality again:

```bash
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_parameter_consistency.py::test_broadcast_parameters_makes_replicas_equal
```

## 10. Manual distributed SGD

Build the deterministic regression dataset and MLP. Each rank gets a distinct
shard, computes local mean loss and gradients, AllReduces each present gradient,
divides by world size for equal shard sizes, and calls plain SGD. No DDP hooks.

### A hand-worked two-rank update

First reduce the protocol to one parameter, `w`, and one example per rank. Both
replicas start with `w = 0`, use squared error, and use learning rate `0.1`:

```text
rank 0 has (x=1, target=2): loss = (w*x - target)^2
rank 1 has (x=2, target=0): loss = (w*x - target)^2
```

For one squared-error example, the gradient with respect to `w` is
`2 * (w*x - target) * x`. At `w = 0`:

```text
rank 0 gradient = 2 * (0*1 - 2) * 1 = -4
rank 1 gradient = 2 * (0*2 - 0) * 2 =  0
```

AllReduce SUM gives `-4` to both ranks. The local batches have equal size, so
divide by two: both ranks now have gradient `-2`. Each replica applies the same
SGD update:

```text
w_new = w - learning_rate * gradient
      = 0 - 0.1 * (-2)
      = 0.2
```

The single-process mean loss over those same two examples has gradient `-2`
at `w=0`, so it also updates `w` to `0.2`. Without gradient synchronization,
rank 0 would update to `0.4` while rank 1 would remain at `0`; the replicas
would diverge immediately. This small calculation is the same sequence used by
the MLP below, where each gradient tensor has many elements instead of one.

### Run the four-rank MLP example

The dataset has 16 examples. Rank 0 takes rows 0–3, rank 1 takes rows 4–7,
rank 2 takes rows 8–11, and rank 3 takes rows 12–15. Each local loss is a mean
over four examples. The per-parameter AllReduce SUM followed by division by
four produces the gradient of the mean loss over all 16 examples.

Run with:

```bash
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.distributed_sgd --steps 3 --self-check
```

Every run checks replica equality after each step. With `--self-check`, it also
fails if the distributed parameters differ from rank 0's single-process
reference trained on the full 16-example batch. Rank 0 emits one JSON record
per step, including loss, batch sizes, and maximum parameter differences.

The opt-in integration test runs the same comparison and checks its output:

```bash
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_distributed_sgd.py::test_one_and_multiple_steps_match_reference_and_stay_synchronized
```

`--sync-gradients=false` is a separate experiment: identical initial replicas
see different data, step on different gradients, and diverge. Verify that with:

```bash
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.distributed_sgd --steps 1 --sync-gradients=false
```

## 11. Correctness against single-process SGD

Compare one process on a global batch of 16 with four processes on four shards
of four examples. Compare gradients before stepping and parameters after
stepping within a stated floating-point tolerance. This equivalence assumes the
same initial parameters, examples, loss reduction, optimizer, learning rate,
and update timing. Data placement changes; the mathematical update should not.

### Derivation 3: why every replica needs the aggregate

Start from the local SGD update on rank `r`:

```text
theta_r_after = theta_r_before - learning_rate * gradient_r
```

The replicas start with the same parameters. To keep them equal after the
step, they must apply the same update, which requires the same aggregate
gradient on every rank. Reduce sends the aggregate to only one destination;
the other replicas cannot apply that global update from the Reduce result.
AllReduce returns the aggregate to every participant, so each independent
optimizer can make the same update.

If a rank has zero samples, it contributes a zero gradient **sum** and count
zero. Weighting by sample counts still works as long as the total count is
positive. If local losses use SUM rather than MEAN reduction, the local
gradient is already a gradient sum: AllReduce those sums and divide by the
global sample count to recover the global mean. Do not multiply a local sum by
its sample count again.

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

### What changes when AllReduce is asynchronous?

With the default `async_op=False`, a call such as `dist.all_reduce(tensor)` is
blocking from the caller's point of view: the Python process waits for the
collective to finish before moving to the next line. When it returns, it is
safe to consume the reduced value.

With `async_op=True`, the call starts the collective and quickly returns a
`Work` handle. The handle represents that particular operation. The collective
may still be running while Python executes later statements:

```text
blocking call:
  call AllReduce ───── wait for communication ───── return; read result

asynchronous call:
  launch AllReduce ── return Work ── independent work ── Work.wait() ── read result
       communication may continue in the background ───────────────────┘
```

Asynchronous changes **when the caller waits**, not the mathematical result.
All ranks still have to launch compatible collectives in the same order, and
every rank eventually has to wait before using its result.

### What exactly does `wait()` do?

`work.wait()` tells this process to block until the operation represented by
`work` has completed for this rank. After it returns, the AllReduce result in
the tensor is ready to read. It does not mean that every peer has moved on to
the next Python statement, and it does not excuse ranks from matching the
collective sequence.

Before `wait()`, treat the participating tensor as owned by the in-flight
operation: do not read it as the final result, overwrite it, or reuse its
storage for another operation. Reading too early can observe the original
local value, a partially updated value, or backend-dependent behavior. Waiting
first removes that race.

### Live four-rank example

`async_all_reduce.py` gives rank `r` the scalar `r + 1`, launches SUM, computes
an unrelated sum of squares, waits, and only then prints the reduced tensor.
For four ranks the expected collective result is
`1 + 2 + 3 + 4 = 10` on every rank; the independent CPU computation is `332833500`.

Run from the repository root on macOS:

```bash
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29517 -m phase3.async_all_reduce
```

Each process should print `sum=10`. Output order is nondeterministic because
the ranks print independently.

The independent CPU work is safe because it does not touch the tensor involved
in AllReduce. This example teaches the `Work`/`wait()` lifecycle; it does not
prove useful communication-computation overlap. Gloo on localhost may finish
the collective before, during, or after that small CPU calculation, and its
timing is not representative of a training cluster.

## 15. Why naive gradient synchronization is inefficient

One AllReduce per parameter is easy to understand but can create many small
collectives. Count calls for a model with 100 parameter tensors. Optionally
flatten a small set of gradients, AllReduce one buffer, then restore them.
Compare many small messages with fewer larger messages; do not add autograd
hooks or attempt full DDP bucketing.

## 16. Failure modes

Try mismatched collective ordering: rank 0 calls A then B while rank 1 calls B
then A. A backend matches calls by protocol order, not Python variable names.
This demo uses different tensor shapes so Gloo may report a shape mismatch
immediately; another backend/configuration may instead time out. Either result
shows that ranks issued incompatible collectives.

In the missing-rank demo, the final rank logs its departure and exits the
scenario without entering AllReduce. The remaining ranks wait in the collective
until Gloo reports the failed participant or the process-group timeout expires.
Both failure cases are deliberately run in subprocesses with a short Gloo
timeout and an outer test timeout. Their logs include rank, PID, tensor shape,
operation, and step.

### Stragglers: why one slow rank slows every rank

An AllReduce combines a value contributed by every rank. Rank 0 cannot know the
final sum until rank 3's value has arrived; the same is true for every other
rank. In synchronous data parallel training, workers must also have the same
parameters before they start the next forward/backward pass. So a worker that
finishes its local batch early reaches the collective and waits for the
remaining workers. The collective is a synchronization point because its
result depends on all participants, even though it is not simply a standalone
barrier.

The demo in `failures.py` makes this visible with four ranks. Rank `r` puts
`r + 1` in its tensor, so the reduction result must be `1 + 2 + 3 + 4 = 10`.
Ranks 0, 1, and 2 enter AllReduce immediately. Rank 3 sleeps for the requested
number of seconds before entering. Each rank measures only the time from its
own `before_all_reduce` point to its own `after_all_reduce` point. The
`collective_seconds` values are therefore local durations; compare those
durations, not monotonic timestamps from different processes.

I ran the following three experiments with four local Gloo processes. The
numbers are from one run on localhost, so expect small variation from startup
and scheduling:

| Rank 3 sleep | Ranks 0–2 measured in AllReduce | Rank 3 measured in AllReduce | Result on every rank |
| ---: | ---: | ---: | ---: |
| 0 s | about 2.9–3.2 ms | about 2.8 ms | 10 |
| 0.25 s | about 256 ms | about 0.54 ms | 10 |
| 0.8 s | about 802 ms | about 0.73 ms | 10 |

Why does rank 3 report a short collective duration in the delayed runs? Its
sleep happens *before* its timer for AllReduce starts. During that sleep, the
other three ranks have already entered AllReduce and are waiting for its
contribution. Once rank 3 arrives, Gloo transfers/reduces the small tensors and
the collective completes on all ranks. The slow work is charged to the early
ranks as collective wait time, even though the slow rank did that work outside
the collective.

The per-step model is roughly: each worker does its local work, then workers
need to finish the collective. If local work on rank `r` takes `C_r` seconds,
the group cannot proceed before the slowest rank is ready, so the local-work
part is governed by `max(C_0, C_1, ..., C_(P-1))`, not by the average. The
communication time comes after (or, in more advanced implementations, can
partly overlap with) local computation. In this simple demo, rank 3's inserted
sleep adds close to 0.25 or 0.8 seconds to the early ranks' wait. If a similar
imbalance recurs on every one of 100 synchronous steps, 0.25 seconds of extra
delay per step is roughly 25 seconds during which faster workers cannot start
their next step. This is why one straggler can erase the benefit of otherwise
fast workers.

Repeat the measurements from the repository root on macOS. Keep each port
different if running commands close together:

```bash
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29580 -m phase3.failures --scenario straggler --seconds 0 --timeout-seconds 5
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29581 -m phase3.failures --scenario straggler --seconds 0.25 --timeout-seconds 5
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29582 -m phase3.failures --scenario straggler --seconds 0.8 --timeout-seconds 5
```

First predict what the three ranks that do not sleep will report. Then inspect
their `collective_seconds` and the `sleep_start`/`sleep_end` timestamps emitted
by rank 3. Check that all ranks print the same reduced value. The experiment
demonstrates the cost of waiting; it does not claim that a local Gloo timing is
a network benchmark or that every rank's clocks can be compared.

The focused integration tests keep failure runs bounded:

```bash
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_failures.py
```

The straggler is slow but valid. Mismatched collective ordering and a missing
rank are protocol failures because the workers no longer execute the same
collective protocol.

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

### Reporting a global loss when shard sizes differ

The reported loss must weight every example equally. A mean of rank means does
not do that when ranks process different numbers of examples. For example,
suppose four ranks have these local mean losses and counts:

| Rank | Examples (`n_r`) | Local mean loss (`L_r`) | Local loss sum (`n_r * L_r`) |
| --- | ---: | ---: | ---: |
| 0 | 1 | 1 | 1 |
| 1 | 2 | 2 | 4 |
| 2 | 3 | 3 | 9 |
| 3 | 4 | 4 | 16 |

The mean of rank means is `(1 + 2 + 3 + 4) / 4 = 2.5`. That gives a
one-example rank the same influence as a four-example rank. The example-level
global mean is `(1 + 4 + 9 + 16) / (1 + 2 + 3 + 4) = 30 / 10 = 3.0`.
Each worker therefore AllReduces its local loss sum and its local example
count, then divides the global loss sum by the global count. The
`global_mean_loss` helper follows this procedure. In the current regression
model each example has one scalar target, so the MSE sum divided by example
count is the per-example mean. For multiple target values per example, define
clearly whether the desired metric averages over examples or individual target
elements, and use the matching denominator.

Run its four-rank check (which uses the counts and means in the table) with:

```bash
PYTHONPATH=src GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29520 -m phase3.distributed_sgd --metrics-self-check
```

Run the focused test with:

```bash
PYTHONPATH=src RUN_DISTRIBUTED=1 python -m pytest -q tests/test_distributed_sgd.py::test_global_mean_loss_weights_uneven_shards_and_reaches_every_rank
```

This change makes the *metric* correct for uneven shards. The training loop's
gradient averaging still assumes equal local batch sizes; uneven-batch
training must weight local gradient sums by their example counts as described
in the unequal-batch gradient derivation above.

Exercise 16 worked derivation: with world size `8`, local batch size `16`, and
`4` microbatches accumulated before each optimizer step, each rank contributes
`16 * 4 = 64` examples to one update. Across all ranks, the effective global
batch size is `8 * 16 * 4 = 512` examples. For the global mean gradient, each
rank must accumulate its four microbatch gradient sums (or equivalently
appropriately scaled means); workers then combine rank contributions and
normalize by the total of 512 examples. With equal local batches and correctly
mean-scaled accumulated gradients, AllReduce SUM followed by division by
world size gives the same mean. Accumulation changes how many local examples
contribute before a step; it does not remove the need for cross-rank gradient
synchronization.

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
