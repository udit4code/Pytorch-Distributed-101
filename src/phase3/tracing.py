"""Small timing utilities for manual training phase instrumentation."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def trace_event(name: str, rank: int) -> Iterator[None]:
    """Print start/end timestamps around a named local event.

    TODO: IMPLEMENT: emit flushed, rank-tagged start and end records with
    monotonic timestamps even if the enclosed operation raises; re-raise errors.
    """
    raise NotImplementedError
