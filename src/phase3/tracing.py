"""Small timing utilities for manual training phase instrumentation."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def trace_event(name: str, rank: int) -> Iterator[None]:
    """Print start/end timestamps around a named local event.

    Emits flushed, rank-tagged start/end records using monotonic timestamps.
    The timestamps and duration describe this process's local timing; do not
    treat monotonic timestamps as synchronized wall-clock times across hosts.
    Exceptions from the enclosed block are not swallowed.
    """
    start = time.monotonic()

    print(
        f"rank={rank} event={name} phase=start ts={start:.9f}",
        flush=True,
    )

    try:
        yield
    finally:
        end = time.monotonic()

        print(
            f"rank={rank} event={name} phase=end ts={end:.9f} "
            f"duration={end - start:.9f}",
            flush=True,
        )
