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
torchrun --standalone --nproc-per-node=4 -m phase3.all_reduce_demo
```

Each process prints a JSON record before and after AllReduce. In the records
whose `"phase"` is `"after"`, check that ranks 0 through 3 all report
`"values": [10, 100]`. The records may appear in a different order because the
processes print independently.

### Test the live four-process behavior

The integration test launches a real `torchrun` subprocess; it does not mock
`torch.distributed`. From the repository root, run:

```bash
RUN_DISTRIBUTED=1 python -m pytest -q \
  tests/test_all_reduce.py::test_all_reduce_demo_returns_expected_sum_on_every_rank
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

AllReduce SUM yields `sum(g_r for r = 0..P-1)`. Divide by world size to obtain the mean
when ranks process equally sized local batches and local losses are means.
Why is averaging necessary to match a mean loss on the concatenated global
batch? Would it still be correct if ranks processed different numbers of
examples? Explain your assumptions before coding.

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

If rank `r` processes `n_r` examples and computes a local **mean** gradient,
then that gradient is

`g_r = (1 / n_r) * sum(grad_theta(loss_(r,i)) for i = 1..n_r)`.

Therefore, `n_r * g_r` is the sum of its per-example gradients. Add those sums
across ranks and divide by the total number of examples to get the global mean:

`g_global = sum(n_r * g_r for r = 0..P-1) / sum(n_r for r = 0..P-1)`

Equivalently, `g_global` is the sum of all per-example gradients divided by
the total number of examples.

For 2 and 8 samples, explain why the unweighted mean of `g_0` and `g_1` gives the two ranks equal
weight rather than the examples equal weight. The count-weighted expression
gives each example equal weight. If a rank has zero examples, it contributes
zero to both the gradient sum and sample count; the global total must still be
positive. Implement weighted aggregation.

### Paper-and-pencil derivation 2

Starting from `g_r = (sum of rank r's per-example gradients) / n_r`, derive
the weighted formula without looking above. State how you would handle a rank with
zero examples and what to do if the global sample count is zero.

## 7. Manual AllReduce

First assemble Reduce + Broadcast. Then use only send/recv to gather values at
a root and distribute the sum. The simple algorithms are intentionally
inefficient: they show that AllReduce is a communication algorithm, not magic.
Do not use Reduce/Broadcast/AllReduce in the point-to-point version.

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
