"""Pause and drain the cyclic garbage collector around exploration runs."""

import gc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final, final

_PENDING_MAX: Final = 100_000


@final
class _CyclicGC:
    __slots__ = "_paused", "_promoted"

    _paused: bool
    _promoted: bool  # whether a drain moved the live graph to the older generation

    def __init__(self) -> None:
        self._paused = False
        self._promoted = False

    @contextmanager
    def pause(self) -> Generator[None]:
        """Hold collection off while live, freeing dead cycles with young sweeps."""
        if self._paused or not gc.isenabled():
            yield
            return
        gc.disable()
        self._paused = True
        self._promoted = False
        try:
            yield
        finally:
            self._paused = False
            gc.enable()
            # a drain promoted the live graph to the older generation; sweep it there
            gc.collect(1 if self._promoted else 0)

    def drain(self) -> None:
        """A young sweep once enough garbage pends while collection is paused."""
        if self._paused and gc.get_count()[0] > _PENDING_MAX:
            gc.collect(0)
            self._promoted = True


cyclic_gc: Final = _CyclicGC()
