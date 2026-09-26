"""Native broadcast demo; launch with torchrun -m phase2.broadcast_demo."""
import argparse
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group, rank_record


def run(src: int = 0, delay_rank: int = -1, delay_seconds: float = 0.0) -> None:
    import time

    # torchrun starts one copy of this program per worker. This connects all
    # workers to the default process group before communication begins.
    setup_process_group()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size() 
        if not (0 <= src < world_size):
            raise ValueError(f"src={src} must be in [0, {world_size})") 
        
        # Only the source rank starts with the value being distributed. Every
        # other rank allocates a same-shape, same-dtype buffer for the result.
        source_value = 999 if src == 2 else 100 
        initial_value = source_value if rank == src else -1
        tensor = torch.tensor([initial_value], dtype=torch.int64)


        # Artificially delay one rank to demonstrate that broadcast
        # waits until every participating rank reaches the collective.
        if rank == delay_rank and delay_seconds > 0.0:
            time.sleep(delay_seconds)

        # The before/after records show that non-source buffers begin at -1 and
        # are overwritten with the source value by broadcast.
        rank_record(rank, "broadcast", tensor, phase="before", src=src)

        # Every rank in the group must make this call in the same collective
        # order, including the source rank. broadcast modifies `tensor` in
        # place, so there is no return value to assign.
        # Collective: every rank must execute this in the same order.
        # The source's tensor is copied into every other rank's tensor.
        dist.broadcast(tensor, src=src)

        rank_record(rank, "broadcast", tensor, phase="after", src=src)
    finally:
        # Destroy the group on success or failure so each worker exits cleanly.
        cleanup_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    run(args.src, args.delay_rank, args.delay_seconds)
