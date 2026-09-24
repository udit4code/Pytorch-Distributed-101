import os

import pytest

from phase0 import rank_info
from phase0.rank_info import RankInfo, get_rank_info


def test_rank_info_is_immutable():
    info = RankInfo(rank=1, local_rank=0, world_size=2, pid=123)
    with pytest.raises((AttributeError, TypeError)):
        info.rank = 2


def test_get_rank_info_uses_distributed_and_torchrun_values(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(rank_info.os, "getpid", lambda: 54321)
    monkeypatch.setattr(rank_info.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rank_info.dist, "get_rank", lambda: 7)
    monkeypatch.setattr(rank_info.dist, "get_world_size", lambda: 8)
    assert get_rank_info() == RankInfo(rank=7, local_rank=3, world_size=8, pid=54321)
