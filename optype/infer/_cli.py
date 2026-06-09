"""The ``optype infer`` command-line logic."""

import ast
import sys

from optype.infer import infer


def run(*args: str) -> None:
    if not args:
        sys.exit("usage: optype infer EXPR [PARAM ...]")

    source, *selectors = args
    params = [int(s) if s.lstrip("-").isdigit() else s for s in selectors]

    body = ast.parse(source).body
    last = body[-1] if body else None
    if isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        # append a reference to the definition so it becomes the final expression
        body = ast.parse(f"{source}\n{last.name}").body
        last = body[-1]
    if not isinstance(last, ast.Expr):
        sys.exit("the final statement must be an expression or a definition")

    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body[:-1], []), "<expr>", "exec"), namespace)  # noqa: S102
    code = compile(ast.Expression(last.value), "<expr>", "eval")
    try:
        print(infer(eval(code, namespace), *params))  # noqa: S307, T201
    except (NotImplementedError, ValueError, TypeError) as exc:
        sys.exit(f"{type(exc).__name__}: {exc}")
