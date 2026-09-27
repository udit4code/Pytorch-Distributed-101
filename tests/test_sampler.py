"""DistributedSampler contract checks (the real sampler needs no process group)."""
from phase3.sampler_demo import indices_for_epoch, make_toy_dataset


def test_divisible_dataset_is_partitioned_without_overlap():
    dataset = make_toy_dataset(16)
    shards = [indices_for_epoch(dataset, r, 4, 0) for r in range(4)]
    assert all(len(shard) == 4 for shard in shards)
    assert sorted(i for shard in shards for i in shard) == list(range(16))
    assert len(set(i for shard in shards for i in shard)) == 16


def test_set_epoch_changes_deterministic_shuffle():
    dataset = make_toy_dataset(16)
    first = [indices_for_epoch(dataset, 0, 4, 0, shuffle=True)]
    second = [indices_for_epoch(dataset, 0, 4, 1, shuffle=True)]
    assert first[0] != second[0]
