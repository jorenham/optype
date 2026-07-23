"""The `optype infer` command-line logic."""

import ast
import sys
import warnings

from . import _color
from ._backends import BackendName
from ._color import ColorMode
from optype.infer import InferError, InferWarning, infer


def _flag_value(rest: list[str]) -> tuple[str, list[str]]:
    """Split the value off a leading `--flag VALUE` or `--flag=VALUE`."""
    _, eq, value = rest[0].partition("=")
    if eq:
        return value, rest[1:]
    return (rest[1] if len(rest) > 1 else ""), rest[2:]


def _format(rest: list[str]) -> tuple[BackendName, list[str]]:
    """Split off a leading `--format {terse,compat}`."""
    name, rest = _flag_value(rest)
    # return the literal, not `name`: not every type checker narrows `str` to the name
    if name == "terse":
        return "terse", rest
    if name == "compat":
        return "compat", rest
    sys.exit("--format must be one of: terse, compat")


def _color_flag(rest: list[str]) -> tuple[ColorMode, list[str]]:
    """Split off a leading `--color {auto,always,never}`."""
    name, rest = _flag_value(rest)
    if name == "auto":
        return "auto", rest
    if name == "always":
        return "always", rest
    if name == "never":
        return "never", rest
    sys.exit("--color must be one of: auto, always, never")


def _flags(args: tuple[str, ...]) -> tuple[BackendName, ColorMode, list[str]]:
    """Strip leading `--format`/`--color` flags in any order."""
    backend: BackendName = "terse"
    color: ColorMode = "auto"
    rest = list(args)
    while rest:
        match rest[0].partition("=")[0]:
            case "--format":
                backend, rest = _format(rest)
            case "--color":
                color, rest = _color_flag(rest)
            case _:
                break  # unrecognized flag or the expression: fall through
    return backend, color, rest


def run(*args: str) -> None:
    backend, color, rest = _flags(args)
    if not rest:
        sys.exit(
            "usage: optype infer [--format {terse,compat}] "
            "[--color {auto,always,never}] EXPR [PARAM ...]",
        )

    source, *selectors = rest
    selectors = [int(s) if s.removeprefix("-").isdigit() else s for s in selectors]

    body = ast.parse(source).body
    last = body[-1] if body else None
    if isinstance(last, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # append a reference to the definition so it becomes the final expression
        body = ast.parse(f"{source}\n{last.name}").body
        last = body[-1]
    if not isinstance(last, ast.Expr):
        sys.exit("the final statement must be an expression or a definition")

    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body[:-1], []), "<expr>", "exec"), namespace)
    code = compile(ast.Expression(last.value), "<expr>", "eval")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InferWarning)
            signature = infer(eval(code, namespace), *selectors, backend=backend)
    except (InferError, ValueError) as exc:
        cause = exc.__cause__
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        notes = "".join(f"\n  {note}" for note in getattr(exc, "__notes__", ()))
        sys.exit(f"{type(exc).__name__}: {exc}{detail}{notes}")

    if _color.want_color(sys.stdout, color):
        signature = _color.highlight(signature)
    print(signature)
    for entry in caught:
        if issubclass(entry.category, InferWarning):
            print(f"warning: {entry.message}", file=sys.stderr)
