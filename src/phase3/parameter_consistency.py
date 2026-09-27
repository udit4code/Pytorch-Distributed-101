"""Model initialization and replica consistency exercises."""

import argparse

import torch
import torch.distributed as dist
from torch import nn

from .distributed import cleanup_process_group, emit, setup_process_group


def broadcast_model_parameters(model: nn.Module, src: int = 0) -> None:
    """Broadcast every parameter from ``src`` so replicas start identically.

    Rank ``src`` supplies the values; every other rank's parameters are
    overwritten in place. All ranks must call this with the same model
    structure and parameter order. Buffers are intentionally out of scope.
    """
    world_size = dist.get_world_size()

    if not 0 <= src < world_size:
        raise ValueError(
            f"src={src} must be in [0, {world_size})"
        )

    # Each rank must visit matching parameters in the same order. no_grad
    # avoids autograd tracking for this initialization-time state copy.
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            dist.broadcast(parameter, src=src)


def assert_parameters_in_sync(
    model: nn.Module,
    atol: float = 0.0,
) -> None:
    """Raise AssertionError if any parameter differs across group members."""

    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol}")

    canonical_rank = 0
    local_mismatch = False
    first_mismatch: tuple[str, float] | None = None

    for name, parameter in model.named_parameters():
        # Clone locally, then replace the clone with rank 0's canonical value.
        # Never broadcast into the live parameter during a consistency check.
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=canonical_rank)

        in_sync = torch.allclose(
            parameter.detach(),
            reference,
            rtol=0.0,
            atol=atol,
        )

        if not in_sync:
            local_mismatch = True
            if first_mismatch is None:
                max_diff = (parameter.detach() - reference).abs().max().item()
                first_mismatch = (name, float(max_diff))

    # Finish every parameter broadcast before any rank raises. Otherwise a
    # mismatching rank could exit while peers block in the next broadcast.
    any_mismatch = torch.tensor([int(local_mismatch)], dtype=torch.int64)
    dist.all_reduce(any_mismatch, op=dist.ReduceOp.MAX)
    if any_mismatch.item():
        if first_mismatch is None:
            raise AssertionError("model parameters differ across process-group ranks")
        name, max_diff = first_mismatch
        raise AssertionError(
            f"Parameter {name!r} is out of sync on rank {dist.get_rank()}: "
            f"max_abs_diff={max_diff}, atol={atol}"
        )


def run_self_check() -> None:
    """Demonstrate seeded initialization, divergence, and parameter broadcast."""
    from .single_process_reference import make_model

    setup_process_group()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 2:
            raise ValueError("the self-check requires at least two ranks")

        # Experiment A: same seed gives every process identical initial weights.
        same_seed_model = make_model(seed=123)
        assert_parameters_in_sync(same_seed_model)

        # Experiment B: different seeds create different initial parameters.
        different_seed_model = make_model(seed=123 + rank)
        try:
            assert_parameters_in_sync(different_seed_model)
        except AssertionError:
            observed_divergence = True
        else:
            raise AssertionError("different seeds unexpectedly produced identical models")

        # Rank 0's model becomes the canonical initialization for all replicas.
        broadcast_model_parameters(different_seed_model, src=0)
        assert_parameters_in_sync(different_seed_model)

        emit({
            "rank": rank,
            "operation": "parameter_consistency_self_check",
            "same_seed_equal": True,
            "different_seed_diverged": observed_divergence,
            "broadcast_equal": True,
            "phase": "passed",
        })
    finally:
        cleanup_process_group()


def main() -> None:
    """Run the real multi-rank initialization consistency experiment."""
    parser = argparse.ArgumentParser(description="Check model replica initialization.")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("pass --self-check to run the multi-rank experiment")
    run_self_check()


if __name__ == "__main__":
    main()
