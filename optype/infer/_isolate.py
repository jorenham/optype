"""Run inference in a forked subprocess so a native fault can't crash the host."""

# ruff: noqa: BLE001

import enum
import faulthandler
import gc
import mmap
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import closing, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

import optype.infer._spy as _spy  # noqa: PLR0402
from ._errors import WARN_SKIP_PREFIX, InferError


class _Status(enum.Enum):
    OK = enum.auto()
    ERROR = enum.auto()


_MAX_STATE_SIZE = 4096

_S_TIMEOUT = 60.0
_S_GRACE = 1.0
_S_POLL = 0.01


def _child(work: Callable[[], object], send: Connection, buf: mmap.mmap) -> None:
    os.setsid()  # own session, so the target's `kill(0)`/`killpg` can't reach the host
    pid = os.getpid()
    faulthandler.disable()  # report a native crash via InferError, not a C-level dump
    _spy.set_state_buffer(buf)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            payload, status, cause = work(), _Status.OK, None
        except BaseException as exc:
            payload, status, cause = exc, _Status.ERROR, exc.__cause__

        warns = [(str(w.message), w.category) for w in caught]

        if os.getpid() != pid:
            os._exit(0)  # the target forked; only the original child may send

        try:
            send.send((status, payload, cause, warns))
        except Exception:
            # a spy in the exception won't pickle
            fallback = payload if status is _Status.OK else InferError(str(payload))
            send.send((status, fallback, None, warns))


def _read_state(buf: mmap.mmap) -> str:
    buf.seek(0)
    return buf.read().split(b"\x00", 1)[0].decode("utf-8", "replace")


def _wait_exit(pid: int) -> bool:
    # `WNOWAIT` keeps the zombie, so the pid (= pgid) can't be recycled before the sweep
    if not hasattr(os, "waitid"):
        return True

    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    for _ in range(round(_S_GRACE / _S_POLL)):
        try:
            if os.waitid(os.P_PID, pid, flags) is not None:
                return True
        except ChildProcessError:
            return True

        time.sleep(_S_POLL)

    return False


def _kill_tree(proc: BaseProcess) -> None:
    # sweep before join: an unreaped child can't have its pid (= pgid) recycled
    if proc.pid is not None:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    proc.kill()
    proc.join()


def _crash_error(sig: int, state: str) -> InferError:
    exc = InferError("inference crashed the interpreter")
    try:
        exc.add_note(f"signal: {signal.Signals(sig).name} ({sig})")
    except ValueError:
        exc.add_note(f"signal: {sig}")
    if state:
        exc.add_note(f"spy state: {state}")
    return exc


def _no_result_error(proc: BaseProcess, buf: mmap.mmap, *, exited: bool) -> InferError:
    code = proc.exitcode
    state = _read_state(buf)
    if exited and code and code < 0:
        return _crash_error(-code, state)

    exc = InferError("inference produced no result")
    if state:
        exc.add_note(f"spy state: {state}")
    if exited:
        exc.add_note(f"exit code: {code}")
    else:
        exc.add_note("the subprocess was killed after breaking its result pipe")
    return exc


def _inline[T](work: Callable[[], T]) -> T:
    """Run `work` in-process, containing exploration debris like a fork child would.

    A leaked explored object (e.g. an unclosed event loop) can raise from its
    deallocator, through a spy or a warnings-as-errors filter; in a fork child that
    noise dies with the process, so it is muted here too (gh-769).
    """

    def mute(_: object, /) -> None: ...

    hook = sys.unraisablehook
    sys.unraisablehook = mute
    try:
        result = work()
        gc.collect()  # finalize the explored garbage while the hook is muted
    finally:
        sys.unraisablehook = hook
    return result


def isolate[T](work: Callable[[], T]) -> T:
    """Run `work` in a forked subprocess so a native crash becomes an `InferError`.

    Raises:
        InferError: If the child crashes, hangs past the timeout, or returns no result.
    """  # noqa: DOC501
    if not hasattr(os, "fork"):
        return _inline(work)

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
            return _inline(work)
        finally:
            # close the parent's write end so a dead child yields EOF, not a hang
            send.close()

        if not recv.poll(_S_TIMEOUT):
            _kill_tree(proc)
            raise InferError("inference timed out")

        exited = True
        try:
            received = recv.recv()
        except Exception:
            # dead child: EOF or truncated pickle
            received = None
            exited = proc.pid is not None and _wait_exit(proc.pid)

        _kill_tree(proc)

        if received is None:
            raise _no_result_error(proc, buf, exited=exited)

        status, payload, cause, warns = received
        for message, category in warns:
            warnings.warn(message, category, skip_file_prefixes=(WARN_SKIP_PREFIX,))

        if status is _Status.ERROR:
            raise payload from cause

        return payload
