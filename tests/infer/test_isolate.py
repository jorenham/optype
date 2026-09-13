"""`isolate`: a forked child or a thread, with a timeout and crash containment."""

# pyright: reportUnusedParameter=false

import contextlib
import gc
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
import warnings
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest

from optype.infer import InferError, InferWarning, _gc, infer
from optype.infer._isolate import _inline, isolate


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


fork_only = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")


type _Isolator = Callable[[Callable[[], Any]], Any]

# without `fork`, `isolate` is `_inline`, so the same cases cover both there
isolators = pytest.mark.parametrize(
    "run",
    [isolate, _inline],
    ids=["isolate", "inline"],
)


@isolators
def test_isolate_returns_value(run: _Isolator) -> None:
    assert run(lambda: "ok") == "ok"


@isolators
def test_isolate_preserves_cause(run: _Isolator) -> None:
    def work() -> None:
        raise InferError("boom") from ValueError("root")

    with pytest.raises(InferError, match="boom") as excinfo:
        run(work)
    assert isinstance(excinfo.value.__cause__, ValueError)


@isolators
def test_isolate_forwards_warning(run: _Isolator) -> None:
    def work() -> str:
        warnings.warn("heads up", InferWarning, stacklevel=1)
        return "ok"

    with pytest.warns(InferWarning, match="heads up"):
        assert run(work) == "ok"


@isolators
def test_isolate_times_out(run: _Isolator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("optype.infer._isolate._S_TIMEOUT", 0.1)
    with pytest.raises(InferError, match="timed out"):
        run(lambda: time.sleep(5))


@fork_only
def test_isolate_reports_signal() -> None:
    with pytest.raises(InferError) as excinfo:
        isolate(lambda: signal.raise_signal(signal.SIGKILL))
    assert any("SIGKILL (9)" in note for note in excinfo.value.__notes__)


@fork_only
def test_isolate_reports_silent_exit() -> None:
    with pytest.raises(InferError, match="no result"):
        isolate(lambda: os._exit(0))


@fork_only
def test_isolate_timeout_reports_spy_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # gh-766: the note names the last spy operation before the block
    monkeypatch.setattr("optype.infer._isolate._S_TIMEOUT", 0.1)

    def block(x: object) -> None:
        repr(x)
        time.sleep(30)

    with pytest.raises(InferError, match="timed out") as excinfo:
        infer(block)
    assert any(
        "spy state: __repr__() -> spy" in note for note in excinfo.value.__notes__
    )


def test_inline_timeout_reports_blocked_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    # gh-766: the note names the frame the abandoned thread stopped in
    monkeypatch.setattr("optype.infer._isolate._S_TIMEOUT", 0.1)
    with pytest.raises(InferError, match="timed out") as excinfo:
        _inline(lambda: time.sleep(5))
    assert any("blocked at: " in note for note in excinfo.value.__notes__)


def test_inline_timeout_resumes_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    # gh-766: the abandoned thread never unwinds its `cyclic_gc.pause`
    monkeypatch.setattr("optype.infer._isolate._S_TIMEOUT", 0.1)

    def work() -> None:
        with _gc.cyclic_gc.pause():
            time.sleep(5)

    enabled = gc.isenabled()
    with pytest.raises(InferError, match="timed out"):
        _inline(work)
    assert gc.isenabled() == enabled
    assert not _gc.cyclic_gc._paused  # ruff: ignore[private-member-access]


@fork_only
def test_isolate_contains_signal_to_own_group() -> None:
    # gh-765: `os.kill(0, ...)` inside the target must not reach the host
    seen: list[int] = []
    previous = signal.signal(signal.SIGUSR1, lambda sig, _: seen.append(sig))

    def work() -> str:
        os.kill(0, signal.SIGUSR1)  # the child inherits (its own copy of) the handler
        return "ok"

    try:
        assert isolate(work) == "ok"
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert not seen


@fork_only
def test_isolate_forked_target_sends_once() -> None:
    # gh-765: a target that forks must not let its fork race the result pipe
    def work() -> str:
        if (pid := os.fork()) == 0:
            return "grandchild"
        os.waitpid(pid, 0)
        return "child"

    assert isolate(work) == "child"


@fork_only
def test_isolate_kills_descendants() -> None:
    # gh-765: an escaped fork of the target dies with the child's process group
    def work() -> int:
        if (pid := os.fork()) == 0:
            time.sleep(60)
            os._exit(0)
        return pid

    pid = isolate(work)
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the target's forked child outlived inference")


@fork_only
def test_isolate_reports_exit_code() -> None:
    with pytest.raises(InferError, match="no result") as excinfo:
        isolate(lambda: os._exit(7))
    assert any("exit code: 7" in note for note in excinfo.value.__notes__)


@fork_only
@pytest.mark.skipif(sys.version_info < (3, 14), reason="`Bdb.start_trace` is 3.14+")
def test_cli_bdb_start_trace() -> None:
    # gh-774: the spy trace function grew the trace without bound, into an OOM
    code = (
        "import bdb, resource;"
        " resource.setrlimit(resource.RLIMIT_AS, (1 << 31, 1 << 31));"
        " from optype.infer import infer; print(infer(bdb.Bdb.start_trace))"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.rstrip().endswith("]) -> None")


@fork_only
def test_infer_isolates_native_crash() -> None:
    # a native callable that faults is contained, not a session crash (#738)
    class _Crash:
        def __call__(self, x: object) -> None:  # ruff: ignore[unused-method-argument]
            signal.raise_signal(signal.SIGKILL)

    with pytest.raises(InferError):
        infer(_Crash())


@isolators
def test_isolate_reports_target_interrupt(run: _Isolator) -> None:
    # gh-773: a `KeyboardInterrupt` raised by the target is a limitation, not a ctrl+C
    def work() -> None:
        raise KeyboardInterrupt

    with pytest.raises(InferError, match="KeyboardInterrupt") as excinfo:
        run(work)
    assert isinstance(excinfo.value.__cause__, KeyboardInterrupt)


@fork_only
def test_isolate_kills_child_on_host_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gh-773: a ctrl+C on the host must not orphan the child in its own session
    pid_file = tmp_path / "pid"

    def work() -> None:
        pid_file.write_text(str(os.getpid()))
        time.sleep(60)

    def poll(*_args: object) -> bool:
        # the interrupt lands while the host waits on the child, once it has started
        for _ in range(500):
            if pid_file.exists() and pid_file.read_text():
                raise KeyboardInterrupt
            time.sleep(0.01)
        pytest.fail("the child never started")

    monkeypatch.setattr(Connection, "poll", poll)
    with pytest.raises(KeyboardInterrupt):
        isolate(work)

    pid = int(pid_file.read_text())
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@fork_only
def test_isolate_detaches_stdio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gh-764: an fd op or a write on 1 in the target must not reach the host's stdout
    (sink := tmp_path / "null").touch()
    monkeypatch.setattr(os, "devnull", str(sink))  # never the real one
    canary = tmp_path / "canary"
    canary.touch()
    canary.chmod(0o644)

    def work() -> None:
        os.write(1, b"leak")
        os.fchmod(1, 0)

    sys.stdout.flush()
    fd = os.open(canary, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(fd, 1)
    try:
        isolate(work)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(fd)
    assert canary.stat().st_mode & 0o777 == 0o644
    assert not canary.read_text()


@fork_only
def test_isolate_with_closed_host_stdio() -> None:
    # gh-764: with the host's stdin and stdout closed, the result pipe takes fd 0 or 1
    code = (
        "import os, sys; from optype.infer._isolate import isolate;"
        " sys.stdout.flush(); os.close(0); os.close(1);"
        " sys.stderr.write(isolate(lambda: 'ok'))"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert out.stderr == "ok"


def test_infer_reports_target_interrupt() -> None:
    # gh-773
    with pytest.raises(InferError, match="KeyboardInterrupt"):
        infer(signal.default_int_handler)


@fork_only
def test_infer_isolates_function_native_crash() -> None:
    # gh-763: cython 3 compiles methods to real functions, so a `FunctionType`
    # can fault in native code just like any other callable
    def crash(x: object) -> None:  # ruff: ignore[unused-function-argument]
        signal.raise_signal(signal.SIGSEGV)

    with pytest.raises(InferError):
        infer(crash)


@fork_only
def test_infer_crash_reports_spy_state() -> None:
    # the spy state note names the last operation before a native crash (#738)
    class _Crash:
        def __call__(self, x: object) -> None:
            repr(x)
            signal.raise_signal(signal.SIGSEGV)

    with pytest.raises(InferError) as excinfo:
        infer(_Crash())
    notes = excinfo.value.__notes__
    assert any("spy state: __repr__() -> spy" in note for note in notes)
