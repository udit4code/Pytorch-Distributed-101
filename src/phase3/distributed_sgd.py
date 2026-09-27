"""Manual synchronous data-parallel SGD; deliberately no DDP."""
from __future__ import annotations

import argparse
import json

import torch
import torch.distributed as dist
from torch import nn

from .distributed import cleanup_process_group, setup_process_group
from .single_process_reference import make_dataset, make_model, reference_step


LEARNING_RATE = 0.05
PARAMETER_SYNC_TOLERANCE = 1e-6
REFERENCE_TOLERANCE = 1e-5


def synchronize_gradients(model: nn.Module) -> None:
    """AllReduce each present gradient, then replace it with the rank mean.

    This matches the global-batch mean when all ranks have equal-sized local
    batches and each local loss uses mean reduction. Every rank must have the
    same parameter order and call the collectives in that same order.
    """
    world_size = dist.get_world_size()

    for parameter in model.parameters():
        if parameter.grad is None:
            # Parameters unused in this rank's forward pass contribute nothing.
            # This is safe only if every rank skips the same parameter positions.
            continue

        # After SUM, every rank has the sum of this parameter's local gradients.
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        # Equal local batch sizes mean each rank's local mean has equal weight.
        parameter.grad.div_(world_size)


def _flatten_parameters(model: nn.Module) -> torch.Tensor:
    """Copy model parameters into one vector for a compact comparison."""
    return torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )


def _global_max_parameter_difference(
    model: nn.Module,
    reference_parameters: torch.Tensor | None = None,
) -> float:
    """Compare local parameters with a rank-0 vector using Broadcast and MAX.

    With no explicit reference vector, rank 0's model is the comparison point.
    When provided, only rank 0 supplies the single-process reference vector.
    A MAX AllReduce makes the largest local difference visible to all ranks;
    this check does not use AllGather.
    """
    rank = dist.get_rank()
    local_parameters = _flatten_parameters(model)

    if rank == 0 and reference_parameters is not None:
        root_parameters = reference_parameters
    else:
        root_parameters = local_parameters.clone()

    dist.broadcast(root_parameters, src=0)
    max_difference = (local_parameters - root_parameters).abs().max().reshape(1)
    dist.all_reduce(max_difference, op=dist.ReduceOp.MAX)
    return float(max_difference.item())


def _parse_bool(value: str) -> bool:
    """Parse the explicit true/false values accepted by the CLI."""
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def run(
    steps: int = 10,
    sync_gradients: bool = True,
    self_check: bool = False,
) -> None:
    """Train local shards, optionally checking each update against one process.

    Rank 0 logs one JSON record per step. With ``self_check=True``, the run
    raises if replicas diverge or if distributed parameters differ from a
    single-process full-batch SGD reference beyond the stated tolerances.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if self_check and not sync_gradients:
        raise ValueError("self-check requires gradient synchronization")

    setup_process_group()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank builds the same model. Then rank 0 broadcasts its exact
        # parameter values so initialization does not depend on RNG details.
        model = make_model(seed=123)
        with torch.no_grad():
            for parameter in model.parameters():
                dist.broadcast(parameter, src=0)

        # Only rank 0 needs the reference model: it trains on the full batch.
        reference_model = make_model(seed=123) if rank == 0 else None

        inputs, targets = make_dataset()
        num_samples = inputs.shape[0]
        if num_samples % world_size != 0:
            raise ValueError(
                f"dataset size {num_samples} must be divisible by "
                f"world_size={world_size}"
            )

        samples_per_rank = num_samples // world_size
        start = rank * samples_per_rank
        end = start + samples_per_rank
        local_inputs = inputs[start:end]
        local_targets = targets[start:end]

        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)

            # Compute this rank's mean loss and its local gradients.
            predictions = model(local_inputs)
            loss = nn.functional.mse_loss(
                predictions, local_targets, reduction="mean"
            )
            loss.backward()

            # Synchronous data parallel update: aggregate gradients before
            # each replica applies its local optimizer step.
            if sync_gradients:
                synchronize_gradients(model)
            optimizer.step()

            # The local shard sizes are equal, so the mean of local mean losses
            # is the global mean loss. Every rank receives that metric.
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            global_loss.div_(world_size)

            # Broadcast rank 0's flattened parameters and reduce the largest
            # local difference. No AllGather is needed for this check.
            max_replica_diff = _global_max_parameter_difference(model)

            reference_max_diff = None
            if sync_gradients:
                # One ordinary full-batch SGD step is the mathematical reference.
                if rank == 0:
                    reference_step(
                        reference_model,
                        inputs,
                        targets,
                        learning_rate=LEARNING_RATE,
                    )
                reference_parameters = (
                    _flatten_parameters(reference_model) if rank == 0 else None
                )
                reference_max_diff = _global_max_parameter_difference(
                    model, reference_parameters=reference_parameters
                )

                # The MAX reductions give every rank the same measured value,
                # so every process makes the same pass/fail decision.
                if max_replica_diff > PARAMETER_SYNC_TOLERANCE:
                    raise AssertionError(
                        f"step {step}: replicas differ by {max_replica_diff}"
                    )
                if self_check and reference_max_diff > REFERENCE_TOLERANCE:
                    raise AssertionError(
                        f"step {step}: distributed/reference parameters differ "
                        f"by {reference_max_diff}"
                    )

            # Keep training output concise: rank 0 emits one structured record.
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "global_loss": global_loss.item(),
                            "world_size": world_size,
                            "local_batch_size": samples_per_rank,
                            "global_batch_size": num_samples,
                            "sync_gradients": sync_gradients,
                            "max_replica_diff": max_replica_diff,
                            "reference_max_diff": reference_max_diff,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        cleanup_process_group()


def main() -> None:
    """Parse torchrun options for the training and correctness experiments."""
    parser = argparse.ArgumentParser(description="Manual CPU/Gloo data-parallel SGD.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sync-gradients", type=_parse_bool, default=True)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="fail unless replicas and distributed updates match the reference",
    )
    args = parser.parse_args()
    run(args.steps, args.sync_gradients, args.self_check)


if __name__ == "__main__":
    main()
