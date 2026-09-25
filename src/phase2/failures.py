"""Bounded collective protocol failure experiments; run only with torchrun."""
import argparse
import torch
import torch.distributed as dist
from .distributed import setup_process_group, cleanup_process_group


def run(scenario: str) -> None:
    setup_process_group(timeout_seconds=8)
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank], dtype=torch.int64)
        if scenario == "ordering":
            if rank == 2:
                dist.barrier()
                dist.broadcast(tensor, src=0)
            else:
                dist.broadcast(tensor, src=0)
                dist.barrier()
        elif scenario == "missing":
            if rank != dist.get_world_size() - 1:
                dist.barrier()
        elif scenario == "shape":
            tensor = torch.zeros(8 if rank == 2 else 4)
            dist.broadcast(tensor, src=0)
        elif scenario == "dtype":
            tensor = torch.zeros(1, dtype=torch.int64 if rank == 2 else torch.float32)
            dist.broadcast(tensor, src=0)
        else:
            raise ValueError(scenario)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("ordering", "missing", "shape", "dtype"))
    run(parser.parse_args().scenario)
