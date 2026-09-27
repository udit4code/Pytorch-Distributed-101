"""DistributedSampler partitioning and epoch-shuffle experiment."""
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler


def indices_for_epoch(dataset: TensorDataset, rank: int, world_size: int,
                      epoch: int, shuffle: bool = False, seed: int = 123) -> list[int]:
    """Return this rank's sampler indices for one reproducible epoch.

    TODO: IMPLEMENT: construct DistributedSampler with explicit replica count,
    rank, shuffle, and seed; call set_epoch(epoch); return its indices. Consider
    drop_last/padding behavior when explaining non-divisible dataset sizes.
    """
    raise NotImplementedError


def make_toy_dataset(size: int = 16) -> TensorDataset:
    """Create a dataset whose sample values reveal their original indices."""
    # TODO: IMPLEMENT: build deterministic tensors and return a TensorDataset.
    raise NotImplementedError
