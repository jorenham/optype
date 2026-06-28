"""The `optype infer` command-line logic."""

import ast
import sys
import warnings

from ._backends import BackendName
from optype.infer import InferError, InferWarning, infer


def _format(args: tuple[str, ...]) -> tuple[BackendName, list[str]]:
    """Split off a leading `--format {terse,compat}`, defaulting to terse."""
    rest = list(args)
    name = "terse"
    if rest and rest[0] == "--format":
        name, rest = (rest[1] if len(rest) > 1 else ""), rest[2:]
    elif rest and rest[0].startswith("--format="):
        name, rest = rest[0].removeprefix("--format="), rest[1:]
    # return the literal, not `name`: not every type checker narrows `str` to the name
    if name == "terse":
        return "terse", rest
    if name == "compat":
        return "compat", rest
    sys.exit("--format must be one of: terse, compat")


def run(*args: str) -> None:
    backend, rest = _format(args)
    if not rest:
        sys.exit("usage: optype infer [--format {terse,compat}] EXPR [PARAM ...]")

    source, *selectors = rest
    params = [int(s) if s.removeprefix("-").isdigit() else s for s in selectors]

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
            signature = infer(eval(code, namespace), *params, backend=backend)
    except (InferError, ValueError) as exc:
        cause = exc.__cause__
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        sys.exit(f"{type(exc).__name__}: {exc}{detail}")

    print(signature)
    for entry in caught:
        if issubclass(entry.category, InferWarning):
            print(f"warning: {entry.message}", file=sys.stderr)
