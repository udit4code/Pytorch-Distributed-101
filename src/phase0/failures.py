"""Exercises for incompatible collectives and process-group cleanup."""

import torch.distributed as dist


def mismatched_collective_demo() -> None:
    """Explore what happens when ranks call incompatible collectives.

    Run only as a multi-process experiment: incompatible collective sequences
    can hang or time out. Keep the timeout short when you implement this.
    """
    # TODO: IMPLEMENT
    # Build an intentional mismatch (for example, one rank calls barrier while
    # another calls a different collective), observe the error/timeout, and
    # explain why every rank must execute a compatible collective sequence.
    raise NotImplementedError("Implement the mismatched-collective experiment")


def cleanup_demo() -> None:
    """Exercise cleanup after successful work and while handling an exception."""
    # TODO: IMPLEMENT
    # Demonstrate try/finally around the process-group lifecycle, then verify
    # cleanup leaves no initialized default group.
    raise NotImplementedError("Implement the cleanup experiment")
