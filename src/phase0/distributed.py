"""Process-group lifecycle helpers. The key distributed steps are exercises."""

import os

import torch.distributed as dist

# Under the hood, when we do torchrun --nproc_per_node=4, torchrun will launch 4 processes, each of which will have the following environment variables set:
# RANK: The rank of the process (0, 1, 2, 3)
# WORLD_SIZE: The total number of processes (4)
# MASTER_ADDR: The address of the master node (localhost or the IP address of the node running the master process)
# MASTER_PORT: The port of the master node (29500)
# After that, every process independently executes dist.init_process_group(backend="gloo", init_method="env://"), which reads the environment variables and initializes the default process group. 
# PyTorch uses these environment variables to make the processes discover each other and creates the default process group containing ranks 0, 1, 2, and 3.
# The default process group is a global singleton that can be accessed via dist.get_default_group().
# Initialization blocks until the participating processes have joined the group, and the default process group is destroyed when the process exits.
# So, the mental model is : torchrun -> creates processes + environment variables and then, 
# init_process_group() -> connects those processes into a distributed group. 
def setup_process_group() -> None:
    """Initialize the default Gloo group using torchrun's rendezvous environment."""
    if dist.is_initialized():
        raise RuntimeError("The default process group is already initialized")

    required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "setup_process_group() expects a torchrun environment; missing: "
            + ", ".join(missing)
        )

    # The default rendezvous method is env://, which reads torchrun's environment.
    dist.init_process_group(backend="gloo")


def cleanup_process_group() -> None:
    """Destroy the default group if it exists; safe to call during cleanup."""
    if dist.is_initialized():
        dist.destroy_process_group()
