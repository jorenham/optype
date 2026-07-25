"""The `optype infer` command-line logic."""

import argparse
import ast
import sys
import warnings
from typing import Final

from . import _color
from ._backends import BackendName
from ._color import ColorMode
from optype.infer import InferError, InferWarning, infer

_FORMATS: Final[tuple[BackendName, ...]] = "terse", "compat"
_COLORS: Final[tuple[ColorMode, ...]] = "auto", "always", "never"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optype infer",
        description="Infer the `optype` protocols required of a callable.",
    )
    parser.add_argument("--format", choices=_FORMATS, default=_FORMATS[0])
    parser.add_argument("--color", choices=_COLORS, default=_COLORS[0])
    # REMAINDER stops flag parsing at the expression, so a `-1` parameter position
    # (and an expression containing `--`) reaches us intact
    parser.add_argument("rest", nargs=argparse.REMAINDER, metavar="EXPR [PARAM ...]")
    return parser


def run(*args: str) -> None:
    parser = _parser()
    ns = parser.parse_args(args)
    rest: list[str] = ns.rest
    if not rest:
        parser.error("the EXPR argument is required")

    backend: BackendName = ns.format
    color: ColorMode = ns.color
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
            rendered = infer(eval(code, namespace), *selectors, backend=backend)
    except (InferError, ValueError) as exc:
        cause = exc.__cause__
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        notes = "".join(f"\n  {note}" for note in getattr(exc, "__notes__", ()))
        sys.exit(f"{type(exc).__name__}: {exc}{detail}{notes}")

    if _color.want_color(sys.stdout, color):
        rendered = _color.highlight(rendered)
    print(rendered)
    for entry in caught:
        if issubclass(entry.category, InferWarning):
            print(f"warning: {entry.message}", file=sys.stderr)
