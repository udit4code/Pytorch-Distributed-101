# Phase 4: AllGather, ReduceScatter, and Sharded Tensors

## 1. Goal

Understand the semantics and mathematics of sharded collective communication on CPU with Gloo. This phase is about **what** collectives mean. Ring/tree algorithms, communication cost models, NCCL, DDP internals, FSDP APIs, DTensor, and tensor-parallel APIs are later topics.

## 2. Recap: AllReduce

AllReduce combines corresponding values across ranks and makes the complete reduced result available to every rank. Phase 4 derives this semantic result from ReduceScatter and AllGather.

## 3. Replicated vs Sharded vs Partial

### Replicated

Every rank owns the complete logical tensor:

```text
Logical: [A B C D]
R0 [A B C D]   R1 [A B C D]   R2 [A B C D]   R3 [A B C D]
```

### Sharded

Each rank owns only part of the logical tensor:

```text
Logical: [A B C D]
R0 [A]   R1 [B]   R2 [C]   R3 [D]
```

### Partial

Each rank owns a contribution to a value. The logical result is elementwise sum:

```text
R0 [a0 a1 a2 a3]   R1 [b0 b1 b2 b3]
R2 [c0 c1 c2 c3]   R3 [d0 d1 d2 d3]
logical [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
```

Keep logical tensor shape distinct from the physical local tensor shape.

## 4. AllGather

Exercise: `python -m phase4.all_gather_demo` under four-rank torchrun. Each rank starts with `[10]`, `[20]`, `[30]`, or `[40]` and should obtain `[10,20,30,40]` in rank order. The collective call is TODO.

```text
Before: R0 [A]  R1 [B]  R2 [C]  R3 [D]
After:  R0 [A B C D]  R1 [A B C D]  R2 [A B C D]  R3 [A B C D]
```

AllGather collects data from every rank; it performs no arithmetic. Broadcast has one meaningful source and copies that source to all ranks. Why is Broadcast not a replacement for AllGather?

**Mandatory paper exercise 1:** Draw the result for R0 `[A]`, R1 `[B]`, R2 `[C]`, R3 `[D]`. State the logical tensor and each rank's local tensor before and after.

## 5. AllGather shapes and memory

`all_gather_tensor_demo.py` uses `all_gather_into_tensor`. For local shape `[2]` on four ranks, output shape is `[8]`; for local shape `[2,3]`, concatenating along the leading dimension gives `[8,3]`. Derive the output shape explicitly and explain why the buffer needs storage for all ranks. If each rank starts with N/P bytes, derive the approximate per-rank footprint before (`N/P`) and after (`N`).

The demo allocates an output buffer, but its collective call remains TODO.

## 6. Manual AllGather

Implement `manual_all_gather` using only `send` and `recv`. Do not call any gather/broadcast collective. Use a deadlock-safe deterministic schedule, and explain why this is a communication pattern rather than magic.

## 7. ReduceScatter

Each rank contributes a full input; corresponding values are reduced, then disjoint pieces of the reduced tensor are distributed. SUM is the required operation. Other operators are optional and backend dependent.

## 8. ReduceScatter mathematics

```text
INPUTS
R0 [a0 a1 a2 a3]   R1 [b0 b1 b2 b3]
R2 [c0 c1 c2 c3]   R3 [d0 d1 d2 d3]

STEP 1 — conceptual reduce
[a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]

STEP 2 — conceptual scatter
R0 <- result[0]   R1 <- result[1]   R2 <- result[2]   R3 <- result[3]
```

**Mandatory paper exercise 2:** For `[1,2,3,4]`, `[10,20,30,40]`, `[100,200,300,400]`, `[1000,2000,3000,4000]`, derive the reduced `[1111,2222,3333,4444]`, then each rank's one-element shard. For P ranks and N elements, require `N % P == 0`; derive local output size `N/P` (e.g. 16 elements / 4 ranks = 4 each). The demo checks divisibility and allocates the expected output shape; native operation remains TODO.

**Mandatory paper exercise 3:** Derive AllReduce = ReduceScatter + AllGather for four ranks and eight values, writing every rank's ownership after each stage.

## 9. Manual ReduceScatter

Implement `manual_reduce_scatter_sum` with send/recv only (or previously learned reduce plus point-to-point scatter). Do not use native ReduceScatter inside it. Compare results with native semantics.

## 10. AllReduce = ReduceScatter + AllGather

For R0 `[1,2,3,4]`, R1 `[10,20,30,40]`, AllReduce SUM gives `[11,22,33,44]`. ReduceScatter gives R0 `[11,22]`, R1 `[33,44]`; AllGather then gives the full result to both. Implement the decomposition without calling `dist.all_reduce`, then compare elementwise to native AllReduce for two and four ranks.

```text
AllReduce ≈ ReduceScatter + AllGather
```

If the next computation does not need the full tensor, why stop after ReduceScatter instead of reconstructing it?

## 11. Sharded parameters

For `W=[w0,...,w7]` on four ranks, local shards are `[w0,w1]`, `[w2,w3]`, `[w4,w5]`, `[w6,w7]`. Implement/test local shard creation and reconstruct with AllGather. Toy workflow: persist shards, temporarily AllGather a full parameter for computation, then retain owned shards. This trades lower persistent storage for communication and temporary materialization.

## 12. Sharded gradients

Ranks compute partial gradient contributions. ReduceScatter combines corresponding values and leaves each rank with its reduced gradient shard. This naturally matches sharded ownership; AllReduce would retain a full result everywhere.

## 13. Toy sharded optimizer

Capstone scaffold: four ranks, eight parameters, two persistent values per rank. AllGather parameters, compute deterministic local gradient contributions, ReduceScatter gradients, update only the local shard, then AllGather for verification against a single-process SGD reference. No DDP/FSDP/DTensor/DeviceMesh/TensorParallel.

## 14. Memory vs communication trade-off

Sharding can reduce persistent memory while adding collectives and temporary full materialization. It is not automatically faster. **Mandatory paper exercise 4:** One billion FP32 values are about 4 GB (decimal; about 3.73 GiB). Compare replicated storage with evenly sharded storage across eight ranks, ignoring buffers.

## 15. Failure modes

Scenarios scaffolded in `failures.py`: inconsistent sizes/dtypes, wrong collective order, missing rank, invalid output shape, and non-divisible inputs. Run only in subprocesses with bounded process-group timeouts; deliberately broken protocols can terminate workers. Diagnostic tests should assert bounded exit and useful output, never wait indefinitely.

## 16. Capstone

Complete `sharded_gradient_demo.py`; verify reconstructed updated parameters numerically against a single-process equivalent. Run with `torchrun --standalone --nproc-per-node=4 -m phase4.sharded_gradient_demo`.

## 17. Senior MLE interview checkpoint

Answer these before looking up APIs:

- A 40 GB parameter tensor is evenly sharded over 8 GPUs. How much persistent parameter storage per GPU, ignoring overhead?
- A layer needs the entire tensor while each GPU stores one shard. What communication materializes it?
- Every rank computes gradient contributions, but retains only its owned reduced gradient shard. Which collective?
- Compare AllReduce and ReduceScatter by post-operation ownership.
- Why can ReduceScatter + AllGather match AllReduce? Show intermediate states.
- Why might sharded training stop after ReduceScatter?
- For 80 GB parameters and 80 GB gradients on 8 ranks, compare replicated with ideal sharded persistent memory; ignore optimizer state.
- Why can AllGather cause a temporary memory spike?
- What memory/communication trade-off does sharding make?

### Questions I must answer

1. What does AllGather do? 2. How does it differ from Broadcast? 3. Does it perform arithmetic? 4. What does each rank own before it? 5. Afterward? 6. What happens to memory? 7. What is ReduceScatter? 8. Why that name? 9. How does it differ from Reduce? 10. What does each rank own afterward? 11. Why is it useful for sharded gradients? 12. What is replicated? 13. Sharded? 14. Partial? 15. Logical vs local shape? 16. How does a sharded parameter become replicated? 17. Which collective? 18. How can full partial gradients become sharded reduced gradients? 19. Which collective? 20. Why is AllReduce semantically ReduceScatter + AllGather? 21. Why stop after ReduceScatter? 22. What memory benefit? 23. What communication cost? 24. Why isn't sharding always faster? 25. Why do these concepts matter for later FSDP/tensor parallelism?

### Phase boundary

Phase 4 answers what AllGather and ReduceScatter do and why sharded training needs them. Phase 5 covers ring/tree algorithms, rounds, bytes transferred, latency/bandwidth, and scaling. Do not move on until you can derive every rank's tensor contents for AllGather, ReduceScatter, and their composition without documentation.
