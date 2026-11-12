"""Pause and drain the cyclic garbage collector around exploration runs."""

import gc
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final, final

_PENDING_MAX: Final = 100_000
# the incremental collector of 3.14.0 to 3.14.4 only sweeps an increment at generation 1
_OLDER_GEN: Final = 2 if (3, 14) <= sys.version_info < (3, 14, 5) else 1


@final
class _CyclicGC:
    __slots__ = "_paused", "_promoted", "_resume"

    _paused: bool
    _promoted: bool  # whether a drain moved the live graph to the older generation
    _resume: bool

    def __init__(self) -> None:
        self._paused = False
        self._promoted = False
        self._resume = True

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
            if self._resume:
                gc.enable()
                # a drain promoted the live graph to the older generation; sweep there
                gc.collect(_OLDER_GEN if self._promoted else 0)

    def stay_paused(self) -> None:
        """Leave collection off after a pause, for a process that exits right after."""
        self._resume = False

    def drain(self) -> None:
        """A young sweep once enough garbage pends while collection is paused."""
        if self._paused and gc.get_count()[0] > _PENDING_MAX:
            gc.collect(0)
            self._promoted = True


cyclic_gc: Final = _CyclicGC()
