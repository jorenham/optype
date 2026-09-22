"""The compat backend end to end: from source text to a type-checked `.pyi` stub."""

import ast
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from optype.infer import InferError, infer

type _Check = Callable[[Path], subprocess.CompletedProcess[str]]


def _compat(source: str) -> str:
    """Infer `source`'s final expression (or definition) with the compat backend."""
    body = ast.parse(source).body
    last = body[-1]
    if isinstance(last, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = ast.parse(f"{source}\n{last.name}").body
        last = body[-1]
    assert isinstance(last, ast.Expr)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body[:-1], []), "<expr>", "exec"), namespace)  # ruff: ignore[exec-builtin]
    func = eval(compile(ast.Expression(last.value), "<expr>", "eval"), namespace)  # ruff: ignore[suspicious-eval-usage]
    return infer(func, backend="compat")


# one representative input per construct, paired with its valid-Python `.pyi` rendering
COMPAT_CASES: list[tuple[str, str]] = [
    (
        # a requirement lifted into a typevar's bound joins with the bound's own
        "def f(make): a = make(0); -make(1); return a, -a",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanNeg\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanNeg[R2]])"
            " -> tuple[CanNeg[R2], R2]: ..."
        ),
    ),
    (
        # the same, one application deep
        "def f(make): a = make(0); abs(-make(1)); return a, abs(-a)",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanAbs, CanNeg\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanNeg[CanAbs[R2]]])"
            " -> tuple[CanNeg[CanAbs[R2]], R2]: ..."
        ),
    ),
    (
        # the same, with the narrower argument known only through its typevar's bound
        "def f(make): a = make(0); abs(-make(1)); b = -a; return a, b, abs(b)",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanAbs, CanNeg\n\n"
            "def f[R3](make: Callable[[Literal[0, 1]], CanNeg[CanAbs[R3]]])"
            " -> tuple[CanNeg[CanAbs[R3]], CanAbs[R3], R3]: ..."
        ),
    ),
    (
        # the same, inside a callable's return type
        "def f(make): a = make(0); abs((-make(1))()); return a, abs((-a)())",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanAbs, CanNeg\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanNeg[Callable[[],"
            " CanAbs[R2]]]]) -> tuple[CanNeg[Callable[[], CanAbs[R2]]], R2]: ..."
        ),
    ),
    (
        # the same, between reads of an attribute
        "def f(make): a = make(0); (-make(1)).x; return a, (-a).x",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal, Protocol\n"
            "from optype import CanNeg\n\n"
            "class HasX[T](Protocol):\n"
            "    @property\n"
            "    def x(self) -> T: ...\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanNeg[HasX[R2]]])"
            " -> tuple[CanNeg[HasX[R2]], R2]: ..."
        ),
    ),
    (
        # the same, through a bound that is an intersection
        ("def f(make): a = make(0); abs(-make(1)); b = -a; return a, b, abs(b), +b"),
        (
            "from collections.abc import Callable\n"
            "from typing import Literal, Protocol\n"
            "from optype import CanAbs, CanNeg, CanPos\n\n"
            "class CanAbsPos[T, U](CanAbs[T], CanPos[U], Protocol): ...\n\n"
            "def f[R3, R4](make: Callable[[Literal[0, 1]], CanNeg[CanAbsPos[R3, R4]]])"
            " -> tuple[CanNeg[CanAbsPos[R3, R4]], CanAbsPos[R3, R4], R3, R4]: ..."
        ),
    ),
    (
        # the same, through a method's return
        "def f(make): a = make(0); abs((-make(1)).x()); return a, abs((-a).x())",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal, Protocol\n"
            "from optype import CanAbs, CanNeg\n\n"
            "class HasX[T](Protocol):\n"
            "    def x(self) -> CanAbs[T]: ...\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanNeg[HasX[R2]]])"
            " -> tuple[CanNeg[HasX[R2]], R2]: ..."
        ),
    ),
    (
        # the same, between two intersections
        (
            "def f(make): a = make(0); abs(-make(1)); +(-make(1));"
            " return a, abs(-a), +(-a)"
        ),
        (
            "from collections.abc import Callable\n"
            "from typing import Literal, Protocol\n"
            "from optype import CanAbs, CanNeg, CanPos\n\n"
            "class CanAbsPos[T, U](CanAbs[T], CanPos[U], Protocol): ...\n\n"
            "def f[R2, R3](make: Callable[[Literal[0, 1]], CanNeg[CanAbsPos[R2,"
            " R3]]]) -> tuple[CanNeg[CanAbsPos[R2, R3]], R2, R3]: ..."
        ),
    ),
    (
        # a requirement lifted while lowering a bound is kept as a constraint
        (
            "def f(make, take): a = make(0); b = make(1); take(-a); abs(-b);"
            " abs(-make(2)); return a, b"
        ),
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanAbs, CanNeg\n\n"
            "def f[T: CanAbs[object]](make: Callable[[Literal[0, 1, 2]],"
            " CanNeg[T]], take: Callable[[T], object])"
            " -> tuple[CanNeg[T], CanNeg[T]]: ..."
        ),
    ),
    (
        # a bound that adds a shipped protocol's bound to a typevar is lowered once
        "lambda x: (x, iter(x))",
        (
            "from optype import CanIter, CanNext\n\n"
            "def f[R: CanNext[object]](x: CanIter[R]) -> tuple[CanIter[R], R]: ..."
        ),
    ),
    (
        # a helper an earlier lowering of a bound registered is not emitted
        "def f(make): a = make(0); next(iter(make(1))); return a, iter(a)",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanIter, CanNext\n\n"
            "def f[R2: CanNext[object]](make: Callable[[Literal[0, 1]],"
            " CanIter[R2]]) -> tuple[CanIter[R2], R2]: ..."
        ),
    ),
    (
        # a bound and a lifted requirement that are one union distribute once
        "def f(make): a = make(0); int(make(1)); return a, int(a)",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanIndex, CanInt\n\n"
            "def f[R: CanInt | CanIndex](make: Callable[[int], R])"
            " -> tuple[R, Literal[1]] | tuple[R, Literal[0]]: ..."
        ),
    ),
    (
        # an arity form of `round` joins by the shipped protocol's variance
        "def f(make): a = make(0); round(make(1)); return a, round(a)",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanRound1\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanRound1[R2]])"
            " -> tuple[CanRound1[R2], R2]: ..."
        ),
    ),
    (
        # the same for `pow`
        "def f(make): a = make(0); pow(make(1), 2, 3); return a, pow(a, 2, 3)",
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanPow3\n\n"
            "def f[R2](make: Callable[[Literal[0, 1]], CanPow3[Literal[2],"
            " Literal[3], R2]]) -> tuple[CanPow3[Literal[2], Literal[3], R2], R2]: ..."
        ),
    ),
    (
        # a builtin argument joins by the IR's variance
        (
            "def f(make): a = make(0); frozenset((1, 2)) in make(1);"
            " return a, frozenset((1,)) in a"
        ),
        (
            "from collections.abc import Callable\n"
            "from typing import Literal\n"
            "from optype import CanContains\n\n"
            "def f[R: CanContains[frozenset[Literal[1, 2]]]](make: Callable[[int],"
            " R]) -> tuple[R, Literal[True]] | tuple[R, Literal[False]]: ..."
        ),
    ),
    (
        # the two arities of `round` fold into the shipped three-parameter protocol
        "lambda x: (round(x), round(x, 2))",
        (
            "from typing import Literal\n"
            "from optype import CanRound\n\n"
            "def f[R, R2](x: CanRound[Literal[2], R, R2]) -> tuple[R, R2]: ..."
        ),
    ),
    (
        # the same for `pow`
        "lambda x: (x ** 2, pow(x, 2, 3))",
        (
            "from typing import Literal\n"
            "from optype import CanPow\n\n"
            "def f[R, R2](x: CanPow[Literal[2], Literal[3], R, R2])"
            " -> tuple[R, R2]: ..."
        ),
    ),
    (
        # both operations fold in one intersection
        "lambda x: (round(x), round(x, 2), x ** 2, pow(x, 2, 3))",
        (
            "from typing import Literal, Protocol\n"
            "from optype import CanPow, CanRound\n\n"
            "class CanRoundPow[T, U, V, W](CanRound[Literal[2], T, U],"
            " CanPow[Literal[2], Literal[3], V, W], Protocol): ...\n\n"
            "def f[R, R2, R3, R4](x: CanRoundPow[R, R2, R3, R4])"
            " -> tuple[R, R2, R3, R4]: ..."
        ),
    ),
    (
        # the shared exponent type compares up to union order
        "lambda x: (x ** 2, x ** 3, pow(x, 3, 4), pow(x, 2, 4))",
        (
            "from typing import Literal\n"
            "from optype import CanPow\n\n"
            "def f[R, R2](x: CanPow[Literal[2, 3], Literal[4], R, R2])"
            " -> tuple[R, R, R2, R2]: ..."
        ),
    ),
    (
        # two `pow` forms at different exponents become `__pow__` overloads
        "lambda x: (x ** 2, pow(x, 3, 5))",
        (
            "from typing import Literal, Protocol, overload\n\n"
            "class CanPow2Pow3[T, U](Protocol):\n"
            "    @overload\n"
            "    def __pow__(self, _0: Literal[2], /) -> T: ...\n"
            "    @overload\n"
            "    def __pow__(self, _0: Literal[3], _1: Literal[5], /) -> U: ...\n\n"
            "def f[R, R2](x: CanPow2Pow3[R, R2]) -> tuple[R, R2]: ..."
        ),
    ),
    (
        # an exponent type compares up to union order inside an invariant one
        "lambda x: (x ** [2, 3], pow(x, [3, 2], 4))",
        (
            "from typing import Literal\n"
            "from optype import CanPow\n\n"
            "def f[R, R2](x: CanPow[list[Literal[2, 3]], Literal[4], R, R2])"
            " -> tuple[R, R2]: ..."
        ),
    ),
    (
        "lambda x: x + 1",
        (
            "from typing import Literal\n"
            "from optype import CanAdd\n\n"
            "def f[R](x: CanAdd[Literal[1], R]) -> R: ..."
        ),
    ),
    (
        "lambda x, y: x * y",
        (
            "from typing import overload\n"
            "from optype import CanMul, CanRMul\n\n"
            "@overload\n"
            "def f[T, R](x: CanMul[T, R], y: T) -> R: ...\n"
            "@overload\n"
            "def f[T, R](x: T, y: CanRMul[T, R]) -> R: ..."
        ),
    ),
    (
        # an intersection becomes a combined protocol, used as a substituted bound
        "lambda x: x if x > 0 else -x",
        (
            "from typing import Literal, Protocol\n"
            "from optype import CanBool, CanGt, CanNeg\n\n"
            "class CanGtNeg[T]"
            "(CanGt[Literal[0], CanBool], CanNeg[T], Protocol): ...\n\n"
            "def f[R](x: CanGtNeg[R]) -> CanGtNeg[R] | R: ..."
        ),
    ),
    (
        # two typevars substituted by one type appear once in a union
        "def f(x, y, z): x(z); y(z); return [x, y]",
        (
            "from collections.abc import Callable\n\n"
            "def f[V](x: Callable[[V], object], y: Callable[[V], object], z: V)"
            " -> list[Callable[[V], object]]: ..."
        ),
    ),
    (
        # a self-referential bound becomes a self-referential protocol
        "lambda xs: sorted(xs)",
        (
            "from typing import Protocol\n"
            "from optype import CanBool, CanIter, CanLt, CanNext\n\n"
            "class CanLt2(CanLt[CanLt2, CanBool], Protocol): ...\n\n"
            "def f(xs: CanIter[CanNext[CanLt2]]) -> list[CanLt2]: ..."
        ),
    ),
    (
        # the inline `Has['spam', +R]` read becomes a `@property` protocol
        "lambda x: x.spam",
        (
            "from typing import Protocol\n\n"
            "class HasSpam[T](Protocol):\n"
            "    @property\n"
            "    def spam(self) -> T: ...\n\n"
            "def f[R](x: HasSpam[R]) -> R: ..."
        ),
    ),
    (
        # a write needs a settable attribute that accepts the value; any getter will do
        "def f(x): x.spam = 1",
        (
            "from typing import Literal, Protocol\n\n"
            "class HasSpam(Protocol):\n"
            "    @property\n"
            "    def spam(self) -> object: ...\n"
            "    @spam.setter\n"
            "    def spam(self, value: Literal[1], /) -> None: ...\n\n"
            "def f(x: HasSpam) -> None: ..."
        ),
    ),
    (
        # the getter binds the name, so the setter's type is qualified past it
        "def f(x): x.list = []",
        (
            "import builtins\n"
            "from typing import Never, Protocol\n\n"
            "class HasList(Protocol):\n"
            "    @property\n"
            "    def list(self) -> object: ...\n"
            "    @list.setter\n"
            "    def list(self, value: builtins.list[Never], /) -> None: ...\n\n"
            "def f(x: HasList) -> None: ..."
        ),
    ),
    (
        # a module the property name shadows is imported under another name
        "import enum\ndef f(x): x.enum = enum.FlagBoundary.STRICT",
        (
            "import enum as _enum\n"
            "from typing import Literal, Protocol\n\n"
            "class HasEnum(Protocol):\n"
            "    @property\n"
            "    def enum(self) -> object: ...\n"
            "    @enum.setter\n"
            "    def enum(self, value: Literal[_enum.FlagBoundary.STRICT], /)"
            " -> None: ...\n\n"
            "def f(x: HasEnum) -> None: ..."
        ),
    ),
    (
        # the module stays imported as itself for its other uses
        (
            "import enum\ndef f(x): x.enum = enum.FlagBoundary.STRICT;"
            " return enum.FlagBoundary.CONFORM"
        ),
        (
            "import enum\n"
            "import enum as _enum\n"
            "from typing import Literal, Protocol\n\n"
            "class HasEnum(Protocol):\n"
            "    @property\n"
            "    def enum(self) -> object: ...\n"
            "    @enum.setter\n"
            "    def enum(self, value: Literal[_enum.FlagBoundary.STRICT], /)"
            " -> None: ...\n\n"
            "def f(x: HasEnum) -> enum.FlagBoundary: ..."
        ),
    ),
    (
        # a dunder has a declared type, which a property would override incompatibly
        "def f(x, y): x.__module__ = str(y)",
        (
            "from typing import Protocol\n"
            "from optype import CanStr\n\n"
            "class Has__module__(Protocol):\n"
            "    __module__: str\n\n"
            "def f(x: Has__module__, y: CanStr) -> None: ..."
        ),
    ),
    (
        # a dunder read alone is still a read-only property
        "def f(x): return x.__array_interface__",
        (
            "from typing import Protocol\n\n"
            "class Has__array_interface__[T](Protocol):\n"
            "    @property\n"
            "    def __array_interface__(self) -> T: ...\n\n"
            "def f[R](x: Has__array_interface__[R]) -> R: ..."
        ),
    ),
    (
        # a shipped read beside a dunder write keeps the annotation form as well
        "def f(x): x.__code__ = (lambda: None).__code__; x.__code__",
        (
            "from types import CodeType\n"
            "from typing import Protocol\n"
            "from optype import HasCode\n\n"
            "class Has__code__(Protocol):\n"
            "    __code__: CodeType\n"
            "class Has__code__Code(Has__code__, HasCode, Protocol): ...\n\n"
            "def f(x: Has__code__Code) -> None: ..."
        ),
    ),
    (
        # a read and a write of one attribute share one member
        "def f(x): x.spam = 1; return x.spam",
        (
            "from typing import Literal, Protocol\n\n"
            "class HasSpam[T](Protocol):\n"
            "    @property\n"
            "    def spam(self) -> T: ...\n"
            "    @spam.setter\n"
            "    def spam(self, value: Literal[1], /) -> None: ...\n\n"
            "def f[R](x: HasSpam[R]) -> R: ..."
        ),
    ),
    (
        "def f(x): x.spam = x.spam",
        (
            "from typing import Protocol\n\n"
            "class HasSpam[T](Protocol):\n"
            "    @property\n"
            "    def spam(self) -> T: ...\n"
            "    @spam.setter\n"
            "    def spam(self, value: T, /) -> None: ...\n\n"
            "def f[T](x: HasSpam[T]) -> None: ..."
        ),
    ),
    (
        # bare presence accepts a read-only property too
        "def f(x): del x.spam",
        (
            "from typing import Protocol\n\n"
            "class HasSpam(Protocol):\n"
            "    @property\n"
            "    def spam(self) -> object: ...\n\n"
            "def f(x: HasSpam) -> None: ..."
        ),
    ),
    (
        "lambda x: x.spam()",
        (
            "from typing import Protocol\n\n"
            "class HasSpam[T](Protocol):\n"
            "    def spam(self) -> T: ...\n\n"
            "def f[R](x: HasSpam[R]) -> R: ..."
        ),
    ),
    (
        # a positional callable parameter renders as `Callable`
        "lambda f, x: map(f, x)",
        (
            "from collections.abc import Callable\n"
            "from optype import CanIter, CanNext\n\n"
            "def f[T, R](f: Callable[[T], R], x: CanIter[CanNext[T]]) -> map[R]: ..."
        ),
    ),
    (
        # a keyword callable parameter renders as a `__call__` protocol
        "lambda f: f(1, b=2)",
        (
            "from typing import Literal, Protocol\n\n"
            "class CanCallP[T](Protocol):\n"
            "    def __call__(self, _0: Literal[1], /, b: Literal[2]) -> T: ...\n\n"
            "def f[R](f: CanCallP[R]) -> R: ..."
        ),
    ),
    (
        # a positional-only default of a returned function is kept
        "lambda: (lambda x=0, /: x)",
        (
            "from typing import Protocol\n\n"
            "class CanCallP[T](Protocol):\n"
            "    def __call__(self, _0: T = 0, /) -> T: ...\n\n"
            "def f[T]() -> CanCallP[T]: ..."
        ),
    ),
    (
        # a required keyword without a `*` before it drops the default before it
        "lambda: (lambda x=0, /, *, y: (x, y))",
        (
            "from typing import Protocol\n\n"
            "class CanCallP[T, U](Protocol):\n"
            "    def __call__(self, _0: T, /, y: U) -> tuple[T, U]: ...\n\n"
            "def f[T, U]() -> CanCallP[T, U]: ..."
        ),
    ),
    (
        # a callable intersected with a protocol lifts into a `__call__`, not a base
        "lambda f, g, x: f(x) if g else g(x)",
        (
            "from collections.abc import Callable\n"
            "from typing import Protocol\n"
            "from optype import CanBool\n\n"
            "class CanBool2[T, U](CanBool, Protocol):\n"
            "    def __call__(self, _0: T, /) -> U: ...\n\n"
            "def f[T, R, R2](f: Callable[[T], R], g: CanBool2[T, R2], x: T)"
            " -> R | R2: ..."
        ),
    ),
    (
        # the `~None` complement is dropped: overloads are matched in order
        "def f(x=None): return [] if x is None else x",
        (
            "from typing import Never, overload\n\n"
            "@overload\n"
            "def f(x: None = None) -> list[Never]: ...\n"
            "@overload\n"
            "def f[T](x: T) -> T: ..."
        ),
    ),
    (
        "def f(x=0): return x",
        "from typing import Literal\n\ndef f[T = Literal[0]](x: T = 0) -> T: ...",
    ),
    ("lambda *args: args", "def f[*Ts](*args: *Ts) -> tuple[*Ts]: ..."),
    (
        # PEP 696: a typevar tuple moves behind a defaulted type parameter
        "lambda *args, x=0: (args, x)",
        (
            "from typing import Literal\n\n"
            "def f[T = Literal[0], *Ts = *tuple[()]](*args: *Ts, x: T = 0)"
            " -> tuple[tuple[*Ts], T]: ..."
        ),
    ),
    ("str.upper", "def f(_0: str, /) -> str: ..."),
    (
        # a *args run forwarded into a method renders as a star parameter (#776)
        "lambda x, f, *args: x.m(f, *args)",
        (
            "from typing import Protocol\n\n"
            "class HasM[T, U, V](Protocol):\n"
            "    def m(self, _0: T, /, *_1: U) -> V: ...\n\n"
            "def f[T, U, R](x: HasM[T, U, R], f: T, *args: U) -> R: ..."
        ),
    ),
    (
        # a leading star parameter takes no `/`; a keyword after it is keyword-only
        "lambda f, *args: f(*args, k=1)",
        (
            "from typing import Literal, Protocol\n\n"
            "class CanCallP[T, U](Protocol):\n"
            "    def __call__(self, *_0: T, k: Literal[1]) -> U: ...\n\n"
            "def f[T, R](f: CanCallP[T, R], *args: T) -> R: ..."
        ),
    ),
    (
        # an enum member renders as its imported member path, not its repr (#776)
        "import http\nlambda x: x == http.HTTPStatus.NOT_FOUND",
        (
            "import http\n"
            "from typing import Literal\n"
            "from optype import CanEq\n\n"
            "def f[R](x: CanEq[Literal[http.HTTPStatus.NOT_FOUND], R]) -> R: ..."
        ),
    ),
    (
        "import http\ndef f(x=http.HTTPStatus.NOT_FOUND): return x",
        (
            "import http\n"
            "from typing import Literal\n\n"
            "def f[T = Literal[http.HTTPStatus.NOT_FOUND]]"
            "(x: T = http.HTTPStatus.NOT_FOUND) -> T: ..."
        ),
    ),
    (
        # a stdlib class renders module-qualified, with its import (#775)
        "import abc\nabc.ABC",
        "import abc\n\ndef f() -> abc.ABC: ...",
    ),
    (
        "import decimal\ndecimal.Decimal",
        (
            "import decimal\n\n"
            "def f(value: str = '0', context: None = None) -> decimal.Decimal: ..."
        ),
    ),
    (
        # a private extension module defers to its public face (`_io` -> `io`)
        "import io\nlambda: io.TextIOWrapper(io.BytesIO())",
        "import io\n\ndef f() -> io.TextIOWrapper: ...",
    ),
    (
        # a generic container base is qualified too, not the deprecated typing alias
        "import collections\nlambda: collections.OrderedDict(a=1)",
        (
            "import collections\n"
            "from typing import Literal\n\n"
            "def f() -> collections.OrderedDict[Literal['a'], Literal[1]]: ..."
        ),
    ),
    (
        "import collections\nlambda: collections.Counter(a=1)",
        (
            "import collections\n"
            "from typing import Literal\n\n"
            "def f() -> collections.Counter[Literal['a']]: ..."
        ),
    ),
    (
        # a bounded protocol parameter hands its bound to the inferred typevar
        "lambda x: iter(x)",
        (
            "from optype import CanIter, CanNext\n\n"
            "def f[R: CanNext[object]](x: CanIter[R]) -> R: ..."
        ),
    ),
    (
        "lambda x: x.__dir__()",
        (
            "import typing\n"
            "from collections.abc import Iterable\n"
            "from optype import CanDir\n\n"
            "def f[R: Iterable[typing.Any]](x: CanDir[R]) -> R: ..."
        ),
    ),
]


@pytest.mark.parametrize(
    ("source", "expected"),
    COMPAT_CASES,
    ids=[src for src, _ in COMPAT_CASES],
)
def test_compat(source: str, expected: str) -> None:
    assert _compat(source) == expected


@pytest.mark.parametrize("attr", ["a-b", "with space", "class"])
def test_compat_non_identifier_attr(attr: str) -> None:
    # a non-identifier attribute cannot name a protocol member, so compat rejects it
    # rather than emit unparsable Python; the terse form still renders the fiction
    func = eval(f"lambda x: getattr(x, {attr!r})")  # ruff: ignore[suspicious-eval-usage]
    assert infer(func) == f"[R](x: Has[{attr!r}, +R]) -> R"
    with pytest.raises(InferError):
        infer(func, backend="compat")


def test_compat_write_matches_wider_attributes(tmp_path: Path) -> None:
    # basedpyright does not check a setter's value type against a plain attribute,
    # so the matching is verified with mypy
    if shutil.which("mypy") is None:
        pytest.skip("mypy is not installed")
    source = f"""{_compat("def f(x): x.spam = 1")}

class Wider:
    spam: int

class Exact:
    spam: Literal[1]

class Other:
    spam: str

class ReadOnly:
    @property
    def spam(self) -> int:
        return 1

f(Wider())
f(Exact())
f(Other())  # error
f(ReadOnly())  # error
"""
    (tmp_path / "write.py").write_text(source)
    out = subprocess.run(
        ["mypy", "--no-error-summary", "--no-color-output", "write.py"],  # ruff: ignore[start-process-with-partial-path]
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    errors = sorted(
        int(n) for n in re.findall(r"^write\.py:(\d+): error", out.stdout, re.MULTILINE)
    )
    lines = source.splitlines()
    assert errors == [
        i for i, line in enumerate(lines, 1) if line.endswith("# error")
    ], out.stdout


def test_compat_typechecks(tmp_path: Path, basedpyright: _Check) -> None:
    # each rendered stub must be valid, self-contained, type-checkable Python
    for i, (source, _) in enumerate(COMPAT_CASES):
        (tmp_path / f"case_{i}.pyi").write_text(f"{_compat(source)}\n")
    out = basedpyright(tmp_path)
    assert out.returncode == 0, out.stdout


_TEMPLATE_COMPAT_CASES: list[tuple[str, str]] = [
    (
        "lambda: t''",
        "import string.templatelib\n\ndef f() -> string.templatelib.Template: ...",
    ),
    (
        "lambda x: t'{x}'.interpolations[0]",
        (
            "import string.templatelib\n\n"
            "def f[T](x: T) -> string.templatelib.Interpolation[T]: ..."
        ),
    ),
    (
        "lambda: t'{1}'.interpolations[0]",
        (
            "import string.templatelib\n\n"
            "def f() -> string.templatelib.Interpolation[int]: ..."
        ),
    ),
]


@pytest.mark.skipif(sys.version_info < (3, 14), reason="requires Python 3.14+")
def test_compat_template_strings(tmp_path: Path, basedpyright: _Check) -> None:
    for i, (source, expected) in enumerate(_TEMPLATE_COMPAT_CASES):
        assert _compat(source) == expected
        (tmp_path / f"case_{i}.pyi").write_text(f"{_compat(source)}\n")
    out = basedpyright(tmp_path)
    assert out.returncode == 0, out.stdout
