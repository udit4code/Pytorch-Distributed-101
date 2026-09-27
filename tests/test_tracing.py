"""Focused tests for trace_event output and exception behavior."""
import re

import pytest

from phase3.tracing import trace_event


def test_trace_event_prints_rank_tagged_start_and_end(capsys):
    with trace_event("forward", rank=2):
        pass

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2

    start = re.fullmatch(
        r"rank=2 event=forward phase=start ts=([0-9]+\.[0-9]+)", lines[0]
    )
    end = re.fullmatch(
        r"rank=2 event=forward phase=end ts=([0-9]+\.[0-9]+) "
        r"duration=([0-9]+\.[0-9]+)",
        lines[1],
    )
    assert start is not None
    assert end is not None
    assert float(end.group(1)) >= float(start.group(1))
    assert float(end.group(2)) >= 0.0


def test_trace_event_logs_end_and_propagates_exception(capsys):
    with pytest.raises(RuntimeError, match="intentional failure"):
        with trace_event("backward", rank=1):
            raise RuntimeError("intentional failure")

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "rank=1 event=backward phase=start" in lines[0]
    assert "rank=1 event=backward phase=end" in lines[1]
