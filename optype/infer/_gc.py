"""Pause and drain the cyclic garbage collector around exploration runs."""

# ruff: noqa: PLW0603

import gc
from collections.abc import Generator
from contextlib import contextmanager

# pending allocations below this leave collection to the normal gc cadence
_PENDING_MAX = 100_000

# unlocked module state: one exploration at a time, like the collector it toggles
_paused = False
_promoted = False


@contextmanager
def pause_gc() -> Generator[None]:
    """Hold automatic collection off while the trace graph is fully live.

    Young-only sweeps between runs and on exit free the dead cycles without
    scanning the host's older heap.
    """
    global _paused, _promoted
    if _paused or not gc.isenabled():
        yield
        return
    gc.disable()
    _paused = True
    _promoted = False
    try:
        yield
    finally:
        _paused = False
        gc.enable()
        # a drain promoted the live graph to the older generation; sweep it there
        gc.collect(1 if _promoted else 0)


def drain_gc() -> None:
    """A young sweep once enough garbage pends while collection is paused."""
    global _promoted
    if _paused and gc.get_count()[0] > _PENDING_MAX:
        gc.collect(0)
        _promoted = True
