"""The `optype infer` command-line logic."""

import ast
import sys

from optype.infer import InferError, infer


def run(*args: str) -> None:
    if not args:
        sys.exit("usage: optype infer EXPR [PARAM ...]")

    source, *selectors = args
    params = [int(s) if s.removeprefix("-").isdigit() else s for s in selectors]

    body = ast.parse(source).body
    last = body[-1] if body else None
    if isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        # append a reference to the definition so it becomes the final expression
        body = ast.parse(f"{source}\n{last.name}").body
        last = body[-1]
    if not isinstance(last, ast.Expr):
        sys.exit("the final statement must be an expression or a definition")

    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body[:-1], []), "<expr>", "exec"), namespace)
    code = compile(ast.Expression(last.value), "<expr>", "eval")
    try:
        print(infer(eval(code, namespace), *params))
    except (InferError, ValueError) as exc:
        sys.exit(f"{type(exc).__name__}: {exc}")
