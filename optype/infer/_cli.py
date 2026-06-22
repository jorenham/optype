"""The `optype infer` command-line logic."""

import ast
import sys
import warnings

from optype.infer import InferError, InferWarning, infer


def run(*args: str) -> None:
    if not args:
        sys.exit("usage: optype infer EXPR [PARAM ...]")

    source, *selectors = args
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
            signature = infer(eval(code, namespace), *params)
    except (InferError, ValueError) as exc:
        cause = exc.__cause__
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        sys.exit(f"{type(exc).__name__}: {exc}{detail}")

    print(signature)
    for entry in caught:
        if issubclass(entry.category, InferWarning):
            print(f"warning: {entry.message}", file=sys.stderr)
