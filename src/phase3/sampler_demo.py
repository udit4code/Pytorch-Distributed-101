"""DistributedSampler partitioning and epoch-shuffle experiment."""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler


def indices_for_epoch(
    dataset: TensorDataset,
    rank: int,
    world_size: int,
    epoch: int,
    shuffle: bool = False,
    seed: int = 123,
) -> list[int]:
    """Return this rank's sampler indices for one reproducible epoch."""

    # Note that DistributedSampler does not usually hand out contiguous chunks. It conceptually forms one global index sequence and then each rank takes every world_size-th element.
    # With shuffle=True, set_epoch(epoch) matters. All ranks must use the same epoch and seed so they construct the same global permutation, but each rank receives a different slice of that permutation.
    # With drop_last=False, for world_size = 4 and dataset_size = 10, 
    # each rank must receive the same number of samples: Ceil(10/4) = 3 
    # so the sampler needs 12 total slots. It therefore pads the global index sequence by repeating some samples.
    # Across all ranks, we may therefore see duplicates.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )

    # Important when shuffle=True.
    # This changes the deterministic permutation for each epoch.
    sampler.set_epoch(epoch)

    return list(iter(sampler))


def make_toy_dataset(size: int = 16) -> TensorDataset:
    """Create a dataset whose sample values reveal their original indices."""

    # x[i] == i, so when the sampler returns index i,
    # the corresponding sample also visibly contains i.
    x = torch.arange(size, dtype=torch.int64).unsqueeze(1)

    # Optional dummy target.
    y = torch.arange(size, dtype=torch.int64)

    return TensorDataset(x, y)