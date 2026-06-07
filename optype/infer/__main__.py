"""Command-line entry point: ``python -m optype.infer EXPR [PARAM ...]``."""

import ast
import sys

from optype.infer import infer


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m optype.infer EXPR [PARAM ...]")

    source, *selectors = sys.argv[1:]
    params = [int(s) if s.lstrip("-").isdigit() else s for s in selectors]

    body = ast.parse(source).body
    last = body[-1] if body else None
    if not isinstance(last, ast.Expr):
        sys.exit("the final statement must be an expression")

    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body[:-1], []), "<expr>", "exec"), namespace)  # noqa: S102
    code = compile(ast.Expression(last.value), "<expr>", "eval")
    try:
        print(infer(eval(code, namespace), *params))  # noqa: S307, T201
    except (NotImplementedError, ValueError, TypeError) as exc:
        sys.exit(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
