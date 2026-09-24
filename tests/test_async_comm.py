"""Scaffolding checks for asynchronous communication exercises."""

from phase1.async_comm import async_ring_exchange, compare_timestamps


def test_async_entry_points_exist():
    assert callable(async_ring_exchange)
    assert callable(compare_timestamps)


def test_timestamp_exercise_names_required_events():
    stamps = compare_timestamps()
    assert set(stamps) == {
        "communication_start", "computation_start", "computation_end",
        "wait_start", "communication_complete",
    }
