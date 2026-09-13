"""The `optype infer` command line: arguments, errors, warnings, and colors."""

import io
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from typing import override

import pytest

from optype.infer import _color


def _run_cli(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    # the color variables are inherited from the developer's shell, so reset them
    base = {k: v for k, v in os.environ.items() if k not in {"NO_COLOR", "FORCE_COLOR"}}
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, *args],
        capture_output=True,
        text=True,
        check=False,
        env=base | env,
    )


_ADD1 = "[R](x: CanAdd[Literal[1], R]) -> R"
_ADD1_COMPAT = (
    "from typing import Literal\n"
    "from optype import CanAdd\n\n"
    "def f[R](x: CanAdd[Literal[1], R]) -> R: ..."
)

CLI_CASES: list[tuple[tuple[str, ...], str]] = [
    (("-m", "optype.infer", "lambda x: x + 1"), _ADD1),
    (
        ("-m", "optype", "infer", "lambda x: x * 2"),
        "[R](x: CanMul[Literal[2], R]) -> R",
    ),
    (("-m", "optype.infer", "--format", "compat", "lambda x: x + 1"), _ADD1_COMPAT),
    # a bare `-1` is a parameter position, not a flag; this is why the trailing
    # arguments are parsed as `argparse.REMAINDER`
    (
        ("-m", "optype.infer", "lambda x, y: x + y", "-1"),
        "[T, R](y: T) -> R\n[T, R](y: CanRAdd[T, R]) -> R",
    ),
    # a trailing def/class is inferred directly, without a closing name reference
    (
        ("-m", "optype", "infer", "def f(x, y): return x @ y"),
        "[T, R](x: CanMatmul[T, R], y: T) -> R\n[T, R](x: T, y: CanRMatmul[T, R]) -> R",
    ),
    (
        ("-m", "optype", "infer", "def f(x=0): return x"),
        "[T = Literal[0]](x: T = 0) -> T",
    ),
    (
        ("-m", "optype", "infer", "lambda *args: args"),
        "[*Ts](*args: *Ts) -> tuple[*Ts]",
    ),
    (
        ("-m", "optype", "infer", "lambda x: lambda y: (x, y)"),
        "[T, U](x: T) -> (y: U) -> tuple[T, U]",
    ),
    # a signatureless builtin is recovered by the arity probe instead of erroring out
    (("-m", "optype", "infer", "type"), "[T](T) -> type[T]"),
]


@pytest.mark.parametrize(
    ("args", "expected"),
    CLI_CASES,
    ids=[" ".join(args) for args, _ in CLI_CASES],
)
def test_cli(args: tuple[str, ...], expected: str) -> None:
    out = _run_cli(*args)
    assert out.returncode == 0
    assert out.stdout.strip() == expected


@pytest.mark.parametrize(
    ("expr", "error"),
    [
        # infer's own limitations exit cleanly, instead of with a traceback
        ("lambda x, y: getattr(x, str(y))", "InferError: no protocol"),
        # gh-778: a spy in the message renders as nothing, so the type name stands in
        ("lambda x: {}[x]", "InferError: KeyError\n"),
        # gh-778: a cause is shown once, and by type alone when it has no message
        (
            "def f(): raise RuntimeError('ran')",
            "InferError: the function never ran to completion (RuntimeError: ran)\n",
        ),
        (
            "def f(): raise AssertionError",
            "InferError: the function never ran to completion (AssertionError)\n",
        ),
        (
            "bytearray.__init__",
            "InferError: ran out of `*args` placeholders (bytearray",
        ),
        # a callable that raises SystemExit when probed errors cleanly, not silently
        ("exit", "InferError:"),
        ("quit", "InferError:"),
    ],
)
def test_cli_infer_error(expr: str, error: str) -> None:
    out = _run_cli("-m", "optype", "infer", expr)
    assert out.returncode == 1
    assert out.stderr.startswith(error)


def test_cli_infer_error_is_cut() -> None:
    # gh-778: the 1024 `*args` placeholders repr as nothing, so the message is cut
    out = _run_cli("-m", "optype", "infer", "lambda *a: int(str(a))")
    assert out.returncode == 1
    assert out.stderr.startswith("InferError: ran out of `*args` placeholders")
    assert out.stderr.rstrip().endswith("[...])")
    assert len(out.stderr) < 300


def test_cli_infer_error_unpicklable_cause_is_cut_once() -> None:
    # gh-778: a cause holding spies takes the pickling fallback; no second cut there
    out = _run_cli(
        "-m",
        "optype",
        "infer",
        "def f(*args): raise ValueError('x' * 300, args)",
    )
    assert out.returncode == 1
    assert out.stderr.rstrip().endswith("[...])")


@pytest.mark.parametrize(
    ("flag", "choices"),
    [("--format", ["terse", "compat"]), ("--color", ["auto", "always", "never"])],
)
def test_cli_invalid_choice(flag: str, choices: list[str]) -> None:
    out = _run_cli("-m", "optype.infer", flag, "json", "lambda x: x")
    assert out.returncode != 0
    # only the last line, since argparse wraps the usage above it to the terminal
    error = out.stderr.strip().splitlines()[-1]
    # py3.12 renders the choices bare, py3.13+ quotes them
    assert "invalid choice: 'json'" in error
    assert all(choice in error for choice in choices)


def test_cli_usage() -> None:
    out = _run_cli("-m", "optype")
    assert out.returncode == 1
    assert "usage" in out.stderr.lower()


def test_cli_warns_on_stderr() -> None:
    # an incomplete exploration keeps the signature on stdout and the warning on stderr
    out = _run_cli("-m", "optype", "infer", "lambda x: [bool(x[i]) for i in range(10)]")
    assert out.returncode == 0
    assert out.stdout.startswith("(x: CanGetitem")
    assert "warning:" in out.stderr
    assert "incomplete exploration" in out.stderr


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# (arguments, environment, whether the output is colored, the output without color)
COLOR_CASES: list[tuple[tuple[str, ...], dict[str, str], bool, str]] = [
    # a pipe is not a tty, so the default auto mode stays plain
    (("lambda x: x + 1",), {}, False, _ADD1),
    (("lambda x: x + 1",), {"FORCE_COLOR": "1"}, True, _ADD1),
    (("--color", "always", "lambda x: x + 1"), {}, True, _ADD1),
    (("--color", "never", "lambda x: x + 1"), {"FORCE_COLOR": "1"}, False, _ADD1),
    (("lambda x: x",), {"NO_COLOR": "1", "FORCE_COLOR": "1"}, False, "[T](x: T) -> T"),
    (
        ("--format", "compat", "lambda x: x + 1"),
        {"FORCE_COLOR": "1"},
        True,
        _ADD1_COMPAT,
    ),
]


@pytest.mark.parametrize(
    ("args", "env", "colored", "expected"),
    COLOR_CASES,
    ids=[" ".join([*args, *env]) for args, env, _, _ in COLOR_CASES],
)
def test_cli_color(
    args: tuple[str, ...],
    env: dict[str, str],
    colored: bool,
    expected: str,
) -> None:
    out = _run_cli("-m", "optype.infer", *args, **env)
    assert out.returncode == 0
    assert ("\x1b[" in out.stdout) is colored
    assert _ANSI_RE.sub("", out.stdout).strip() == expected


class _Tty(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


def test_want_color_mode_overrides() -> None:
    assert _color.want_color(io.StringIO(), "always")
    assert not _color.want_color(_Tty(), "never")


def test_want_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert _color.want_color(_Tty())
    assert not _color.want_color(io.StringIO())
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _color.want_color(io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not _color.want_color(_Tty())


def test_highlight_roundtrip() -> None:
    samples = [
        "[R](x: CanAdd[Literal[1], R]) -> R",
        "[*Ts](*args: *Ts) -> tuple[*Ts]",
        (
            "from typing import Protocol, overload\n"
            "@overload\n"
            'def f[R](x: CanAdd[Literal[1], R], msg: str = "x") -> R: ...\n'
            "class _H[T](Protocol):\n"
            "    def __getattr__(self, k: str, /) -> int: ..."
        ),
    ]
    for sig in samples:
        assert _ANSI_RE.sub("", _color.highlight(sig)) == sig


def _wrapped(token: str) -> bool:
    out = _color.highlight(token)
    return out != token and _ANSI_RE.sub("", out) == token


def test_highlight_targets() -> None:
    assert _wrapped("def")  # keyword
    assert _wrapped("1")  # number
    assert _wrapped('"x"')  # string literal
    # names and soft keywords stay plain
    assert _color.highlight("CanAdd") == "CanAdd"
    assert _color.highlight("type") == "type"
