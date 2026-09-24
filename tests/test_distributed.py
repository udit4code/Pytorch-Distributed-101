import pytest

from phase0 import distributed


def test_setup_requires_torchrun_environment(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="torchrun environment"):
        distributed.setup_process_group()


def test_setup_initializes_gloo_from_torchrun_environment(monkeypatch):
    calls = []
    for name, value in {
        "RANK": "1",
        "WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    distributed.setup_process_group()
    assert calls == [{"backend": "gloo"}]


def test_cleanup_is_safe_when_group_is_not_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "destroy_process_group", lambda: calls.append(True))
    distributed.cleanup_process_group()
    assert calls == []


def test_cleanup_destroys_initialized_group(monkeypatch):
    calls = []
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "destroy_process_group", lambda: calls.append(True))
    distributed.cleanup_process_group()
    assert calls == [True]
