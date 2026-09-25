"""Unit-level checks for asynchronous communication entry points."""

from phase1.async_comm import async_ring_exchange, compare_timestamps


def test_async_entry_points_exist():
    assert callable(async_ring_exchange)
    assert callable(compare_timestamps)


def test_timestamp_comparison_is_an_entry_point():
    # This function needs an initialized process group; its behavior is covered
    # by the real torchrun integration test rather than a mocked distributed API.
    assert callable(compare_timestamps)
