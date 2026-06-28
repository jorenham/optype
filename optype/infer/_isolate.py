"""Run inference in a forked subprocess so a native fault can't crash the host."""

# ruff: noqa: BLE001

import faulthandler
import mmap
import multiprocessing as mp
import os
import signal
import warnings
from collections.abc import Callable
from contextlib import closing
from multiprocessing.connection import Connection

import optype.infer._spy as _spy  # noqa: PLR0402
from ._errors import WARN_SKIP_PREFIX, InferError

_MAX_STATE_SIZE = 4096
_TIMEOUT = 60.0  # seconds


def _child(work: Callable[[], object], send: Connection, buf: mmap.mmap) -> None:
    faulthandler.disable()  # report a native crash via InferError, not a C-level dump
    _spy.set_state_buffer(buf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            payload, kind, cause = work(), "ok", None
        except BaseException as exc:
            payload, kind, cause = exc, "error", exc.__cause__

        warns = [(str(w.message), w.category) for w in caught]

        try:
            send.send((kind, payload, cause, warns))
        except Exception:
            # a spy in the exception won't pickle
            fallback = payload if kind == "ok" else InferError(str(payload))
            send.send((kind, fallback, None, warns))


def _read_state(buf: mmap.mmap) -> str:
    buf.seek(0)
    return buf.read().split(b"\x00", 1)[0].decode("utf-8", "replace")


def _crash_error(sig: int, state: str) -> InferError:
    exc = InferError("inference crashed the interpreter")
    try:
        exc.add_note(f"signal: {signal.Signals(sig).name} ({sig})")
    except ValueError:
        exc.add_note(f"signal: {sig}")
    if state:
        exc.add_note(f"spy state: {state}")
    return exc


def isolate[T](work: Callable[[], T]) -> T:
    """Run `work` in a forked subprocess so a native crash becomes an `InferError`.

    Raises:
        InferError: If the child crashes, hangs past the timeout, or returns no result.
    """  # noqa: DOC501
    if not hasattr(os, "fork"):
        return work()

    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    with closing(recv), closing(mmap.mmap(-1, _MAX_STATE_SIZE)) as buf:
        proc = ctx.Process(target=_child, args=(work, send, buf))
        try:
            with warnings.catch_warnings():
                # multi-threaded fork warns; we'd otherwise error on it
                warnings.simplefilter("ignore", DeprecationWarning)
                proc.start()
        except OSError:
            return work()
        finally:
            # close the parent's write end so a dead child yields EOF, not a hang
            send.close()

        if not recv.poll(_TIMEOUT):
            proc.terminate()
            proc.join()
            raise InferError("inference timed out")
        try:
            received = recv.recv()
        except Exception:
            # dead child: EOF or truncated pickle
            received = None
        proc.join()

        if received is None:
            code = proc.exitcode
            if code and code < 0:
                raise _crash_error(-code, _read_state(buf))
            raise InferError("inference produced no result")

        kind, payload, cause, warns = received
        for message, category in warns:
            warnings.warn(message, category, skip_file_prefixes=(WARN_SKIP_PREFIX,))

        if kind == "error":
            raise payload from cause

        return payload
