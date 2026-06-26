# ruff: noqa: FURB118, PLW0108
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnusedParameter=false

import builtins
import functools
import gc
import itertools
import math
import operator
import random
import secrets
import subprocess  # noqa: S404
import sys
import warnings
import weakref
from collections.abc import Callable
from inspect import currentframe, signature
from types import MappingProxyType
from typing import Any

import pytest

from optype.infer import InferError, InferWarning, infer
from optype.infer._api import _Gap
from optype.infer._ir import (
    App,
    Arg,
    Dots,
    Fn,
    Lit,
    Name,
    Node,
    Type,
    names,
    render,
    subtype,
    union,
)
from optype.infer._numpy import array_function_node
from optype.infer._values import GapKind

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


def _type_of[T](x: T) -> type[T]:
    return type(x)


def _set_attr(x: Any) -> object:
    x.spam = 1
    return x


def _set_attr_twice(x: Any) -> None:
    x.spam = 1
    x.spam = 1.5


def _del_attr(x: Any) -> None:
    del x.spam


def _get_attr(x: Any) -> None:
    x.spam  # noqa: B018


def _call_attr(x: Any) -> None:
    x.spam()


def _iadd_attr(x: Any) -> None:
    x.spam += 1


def _set_dunder(x: Any) -> None:
    x.__name__ = 123


def _del_dunder(x: Any) -> None:
    del x.__name__


def _set_class_attr(x: Any) -> None:
    type(x).spam = 1


def _del_class_attr(x: Any) -> None:
    del type(x).spam


def _get_class_attr(x: Any) -> None:
    type(x).spam  # noqa: B018


UNARY_CASES: list[tuple[Callable[[Any], Any], str]] = [
    (lambda x: x + 1, "[R](x: CanAdd[Literal[1], R]) -> R"),
    (
        lambda x: x + x,
        "[T: CanAdd[T, R], R](x: T) -> R\n[T: CanRAdd[T, R], R](x: T) -> R",
    ),
    (lambda x: x - 1, "[R](x: CanSub[Literal[1], R]) -> R"),
    (lambda x: x * 2, "[R](x: CanMul[Literal[2], R]) -> R"),
    (
        lambda x: x / x,
        "[T: CanTruediv[T, R], R](x: T) -> R\n[T: CanRTruediv[T, R], R](x: T) -> R",
    ),
    (lambda x: x % 2, "[R](x: CanMod[Literal[2], R]) -> R"),
    (lambda x: x**2, "[R](x: CanPow[Literal[2], R]) -> R"),
    (lambda x: x | 1, "[R](x: CanOr[Literal[1], R]) -> R"),
    (lambda x: x < 1, "[R](x: CanLt[Literal[1], R]) -> R"),
    (lambda x: x <= 1, "[R](x: CanLe[Literal[1], R]) -> R"),
    (lambda x: x > 1, "[R](x: CanGt[Literal[1], R]) -> R"),
    (lambda x: x >= 1, "[R](x: CanGe[Literal[1], R]) -> R"),
    (lambda x: -x, "[R](x: CanNeg[R]) -> R"),
    (lambda x: ~x, "[R](x: CanInvert[R]) -> R"),
    (lambda x: abs(x), "[R](x: CanAbs[R]) -> R"),
    # a positional-only parameter cannot be passed by keyword, so no name is shown
    (lambda x, /: x, "[T](T) -> T"),
    (len, "(CanLen) -> int"),
    (
        list,
        (
            "(tuple[()] = ...) -> list[Never]\n"
            "[R](CanIter[CanNext[R]] & ~tuple[()]) -> list[R]"
        ),
    ),
    # `list()` probes `__len__` optionally via `length_hint`, so it is no requirement
    (lambda x: list(x), "[R](x: CanIter[CanNext[R]]) -> list[R]"),
    (math.sqrt, "(CanFloat | CanIndex) -> float"),
    (lambda x: int(x), "(x: CanInt | CanIndex) -> int"),
    (lambda x: complex(x), "(x: CanComplex | CanFloat | CanIndex) -> complex"),
    (operator.index, "(CanIndex) -> int"),
    (lambda x: x(), "[R](x: () -> R) -> R"),
    (lambda x: x(1, 2), "[R](x: (Literal[1], Literal[2]) -> R) -> R"),
    (lambda x: x(a=1), "[R](x: (a: Literal[1]) -> R) -> R"),
    (lambda x: x(1, b=2), "[R](x: (Literal[1], b: Literal[2]) -> R) -> R"),
    (lambda x: format(x, ">10"), "(x: CanFormat[Literal['>10']]) -> str"),
    (lambda x: f"{x}", "(x: CanFormat[Literal['']]) -> str"),
    (lambda x: x[0], "[R](x: CanGetitem[Literal[0], R]) -> R"),
    (lambda x: x["a"], "[R](x: CanGetitem[Literal['a'], R]) -> R"),
    (lambda x: x[b"a"], "[R](x: CanGetitem[Literal[b'a'], R]) -> R"),
    (lambda x: x[True], "[R](x: CanGetitem[Literal[True], R]) -> R"),
    (lambda x: x[None], "[R](x: CanGetitem[None, R]) -> R"),
    (lambda x: x[1.0], "[R](x: CanGetitem[float, R]) -> R"),
    (
        lambda x: +x * -x,
        (
            "[T, R](x: CanPos[CanMul[T, R]] & CanNeg[T]) -> R\n"
            "[T, R](x: CanPos[T] & CanNeg[CanRMul[T, R]]) -> R"
        ),
    ),
    (
        lambda x: -(x + x),
        (
            "[T: CanAdd[T, CanNeg[R]], R](x: T) -> R\n"
            "[T: CanRAdd[T, CanNeg[R]], R](x: T) -> R"
        ),
    ),
    (lambda x: (x + 1) * 2, "[R](x: CanAdd[Literal[1], CanMul[Literal[2], R]]) -> R"),
    (
        lambda x: x if x > 0 else -x,
        "[T: CanGt[Literal[0], CanBool] & CanNeg[R], R](x: T) -> T | R",
    ),
    (lambda x: -x if x else x, "[T: CanBool & CanNeg[R], R](x: T) -> R | T"),
    # `and` returns an operand: falsy `x` yields `x`, truthy `x` yields `not x`
    (lambda x: x and not x, "[T: CanBool](x: T) -> bool | T"),  # noqa: SIM220
    # `bool` is stable per run, so the `and` is always falsy and `foo` is unreachable
    (lambda x: x.foo() if (x and not x) else x, "[T: CanBool](x: T) -> T"),  # noqa: SIM220
    (
        lambda x: (x + 1) if x else (x - 1),
        (
            "[R, R2](x: CanBool & CanAdd[Literal[1], R] & "
            "CanSub[Literal[1], R2]) -> R | R2"
        ),
    ),
    # `CanGetitem & CanLen` combines as `CanSequence`
    (
        lambda x: x[0] if len(x) else None,
        "[R](x: CanSequence[Literal[0], R]) -> R | None",
    ),
    (
        lambda x: -x if int(x) else +x,
        "[R, R2](x: (CanInt | CanIndex) & CanNeg[R] & CanPos[R2]) -> R | R2",
    ),
    (
        lambda x: -x if operator.index(x) else +x,
        "[R, R2](x: CanIndex & CanNeg[R] & CanPos[R2]) -> R | R2",
    ),
    (
        lambda x: "empty" if len(x) == 0 else x[0],
        "[R](x: CanSequence[Literal[0], R]) -> R | str",
    ),
    (lambda x: len(x) + int(x), "(x: CanLen & (CanInt | CanIndex)) -> int"),
    (lambda x: True if x else 1, "(x: CanBool) -> int"),
    (lambda x: 1 if x else 1.5, "(x: CanBool) -> int | float"),
    (
        lambda x: (OSError(), "a") if x else (FileNotFoundError(), "a"),
        "(x: CanBool) -> tuple[OSError, Literal['a']]",
    ),
    (
        lambda x: [FileNotFoundError()] if x else [OSError()],  # list is invariant
        "(x: CanBool) -> list[FileNotFoundError] | list[OSError]",
    ),
    (
        lambda x: x[0] if len(x) else (-x if int(x) else None),
        (
            "[R, R2](x: CanSequence[Literal[0], R] & "
            "(CanInt | CanIndex) & CanNeg[R2]) -> R | R2 | None"
        ),
    ),
    (
        lambda x: x[0] + x[1],
        (
            "[T, R](x: CanGetitem[Literal[0, 1], T & CanAdd[T, R]]) -> R\n"
            "[T, R](x: CanGetitem[Literal[0, 1], T & CanRAdd[T, R]]) -> R"
        ),
    ),
    (lambda x: [i for i in x], "[R](x: CanIter[CanNext[R]]) -> list[R]"),  # noqa: C416
    (lambda x: [[i] for i in x], "[R](x: CanIter[CanNext[R]]) -> list[list[R]]"),
    (lambda x: [str(i) for i in x], "(x: CanIter[CanNext[CanStr]]) -> list[str]"),
    (lambda x: next(iter(x)), "[R](x: CanIter[CanNext[R]]) -> R"),
    (lambda x: {*x}, "[R: CanHash](x: CanIter[CanNext[R]]) -> set[R]"),
    (
        lambda x: {i: str(i) for i in x},
        "[R: CanStr & CanHash](x: CanIter[CanNext[R]]) -> dict[R, str]",
    ),
    # `0 + a + b` reflects two ways: the total keeps adding (`T` recurs in its bound)
    # or is absorbed right (`T` is used once, so it inlines)
    (
        lambda x: sum(x),
        (
            "[T: CanRAdd[Literal[0], CanAdd[T, R]], R](x: CanIter[CanNext[T]]) -> R\n"
            "[R](x: CanIter[CanNext[CanRAdd[Literal[0] | R, R]]]) -> R"
        ),
    ),
    # comparing reducers reach `__lt__`/`__gt__` only once the iterator yields a pair
    (lambda x: sorted(x), "[R: CanLt[R, CanBool]](x: CanIter[CanNext[R]]) -> list[R]"),
    (lambda x: min(x), "[R: CanLt[R, CanBool]](x: CanIter[CanNext[R]]) -> R"),
    (lambda x: max(x), "[R: CanGt[R, CanBool]](x: CanIter[CanNext[R]]) -> R"),
    (lambda x: (x + 1, x + 1), "[R](x: CanAdd[Literal[1], R]) -> tuple[R, R]"),
    (lambda x: (x + 1, x + 2), "[R](x: CanAdd[Literal[1, 2], R]) -> tuple[R, R]"),
    (
        lambda x: (x + 1, x + 2, x + 3),
        "[R](x: CanAdd[Literal[1, 2, 3], R]) -> tuple[R, R, R]",
    ),
    # PEP 484's numeric tower is a static-typing fiction, so `int` is no `float`
    (
        lambda x: (x + 1, x + 1.0),
        "[R](x: CanAdd[Literal[1] | float, R]) -> tuple[R, R]",
    ),
    (
        lambda x: (x + 1, x + 1j),
        "[R](x: CanAdd[Literal[1] | complex, R]) -> tuple[R, R]",
    ),
    (
        lambda x: (x + 1.0, x + 1j),
        "[R](x: CanAdd[float | complex, R]) -> tuple[R, R]",
    ),
    (
        lambda x: (x + 1, x + 1.0, x + "a"),
        "[R](x: CanAdd[Literal[1, 'a'] | float, R]) -> tuple[R, R, R]",
    ),
    (
        lambda x: (x[True], x[1.5]),
        "[R](x: CanGetitem[Literal[True] | float, R]) -> tuple[R, R]",
    ),
    (
        lambda x: (x[True], x[1]),  # Literal[True] is not Literal[1]
        "[R](x: CanGetitem[Literal[True, 1], R]) -> tuple[R, R]",
    ),
    (lambda x: (x[1.0], x[None]), "[R](x: CanGetitem[float | None, R]) -> tuple[R, R]"),
    (lambda x: (-x, -x), "[R](x: CanNeg[R]) -> tuple[R, R]"),
    (lambda x: (x + 1, "a"), "[R](x: CanAdd[Literal[1], R]) -> tuple[R, Literal['a']]"),
    (lambda x: x + (1, 2), "[R](x: CanAdd[tuple[Literal[1], Literal[2]], R]) -> R"),  # noqa: RUF005
    (lambda x: [x], "[T](x: T) -> list[T]"),
    (lambda x: (x, 1), "[T](x: T) -> tuple[T, Literal[1]]"),
    # every spy has a unique class, so a `type(x)` result renders generically
    (_type_of, "[T](x: T) -> type[T]"),
    (lambda x: x.__class__, "[T](x: T) -> type[T]"),
    (lambda x: (x, type(x)), "[T](x: T) -> tuple[T, type[T]]"),
    (lambda x: type(next(x)), "[R](x: CanNext[R]) -> type[R]"),
    (lambda x: type(str(x)), "(x: CanStr) -> type[str]"),
    # an instance of a spy's class collapses onto the spy itself, and an operation
    # on such an instance is required of the spy's type
    (lambda x: type(x)(), "[T](x: T) -> T"),
    (lambda x: type(x)() + 1, "[R](x: CanAdd[Literal[1], R]) -> R"),
    (lambda x: str(type(x)()), "(x: CanStr) -> str"),
    # a nameable class renders parameterized, and `type` is covariant
    (lambda x: int if x else bool, "(x: CanBool) -> type[int]"),
    (lambda x: bool if x else type(None), "(x: CanBool) -> type[bool] | type[None]"),
    (lambda x: x.__name__, "[R](x: HasName[R]) -> R"),
    (lambda x: x.__qualname__, "[R](x: HasQualname[R]) -> R"),
    (lambda x: x.__match_args__, "[R](x: HasMatchArgs[R]) -> R"),
    (lambda x: x.__type_params__, "[R](x: HasTypeParams[R]) -> R"),
    (lambda x: x.__self__, "[R](x: HasSelf[R]) -> R"),
    # an attribute without a shipped `Has*` protocol synthesizes the fictional
    # inline `Has['name', T]` form, which is not valid Python, like `&` and `~`;
    # a read requires only the covariant (read-only) `+T`, i.e. a property; on
    # a callable the sigil sinks into the return type, i.e. a method
    (lambda x: x.spam, "[R](x: Has['spam', +R]) -> R"),
    (lambda x: x.spam(), "[R](x: Has['spam', () -> +R]) -> R"),
    (lambda x: x.spam(1), "[R](x: Has['spam', (Literal[1]) -> +R]) -> R"),
    (_call_attr, "(x: Has['spam', () -> object]) -> None"),
    (lambda x: x.__wibble__, "[R](x: Has['__wibble__', +R]) -> R"),
    (lambda x: getattr(x, "spam"), "[R](x: Has['spam', +R]) -> R"),  # noqa: B009
    (lambda x: x.spam.ham, "[R](x: Has['spam', +Has['ham', +R]]) -> R"),
    # distinct attributes get distinct typevars; repeated reads share one
    (
        lambda x: (x.spam, x.ham),
        "[R, R2](x: Has['spam', +R] & Has['ham', +R2]) -> tuple[R, R2]",
    ),
    (lambda x: (x.spam, x.spam), "[R](x: Has['spam', +R]) -> tuple[R, R]"),
    (
        lambda x: (x.__name__, x.__qualname__),
        "[R, R2](x: HasName[R] & HasQualname[R2]) -> tuple[R, R2]",
    ),
    # an assignment binds the contravariant (write-only) `-T`, which the attribute
    # only has to accept, so merged writes union; a deletion or an unused read
    # requires bare existence
    (_set_attr, "[T: Has['spam', -Literal[1]]](x: T) -> T"),
    (_set_attr_twice, "(x: Has['spam', -(Literal[1] | float)]) -> None"),
    (_del_attr, "(x: Has['spam']) -> None"),
    (_get_attr, "(x: Has['spam']) -> None"),
    # an augmented assignment is both a read and a write
    (
        _iadd_attr,
        "[T](x: Has['spam', +CanIAdd[Literal[1], T]] & Has['spam', -T]) -> None",
    ),
    # a write to an attribute with a shipped protocol synthesizes too: `HasName`
    # declares `__name__: str`, which the assigned value need not satisfy
    (_set_dunder, "(x: Has['__name__', -Literal[123]]) -> None"),
    (_del_dunder, "(x: Has['__name__']) -> None"),
    # an attribute on the class itself mirrors a `ClassVar` protocol member
    (lambda x: type(x).spam, "[R](x: Has['spam', ClassVar[+R]]) -> R"),
    (lambda x: x.__class__.spam, "[R](x: Has['spam', ClassVar[+R]]) -> R"),
    (lambda x: type(x).spam(), "[R](x: Has['spam', ClassVar[() -> +R]]) -> R"),
    (
        lambda x: type(x).spam(1),
        "[R](x: Has['spam', ClassVar[(Literal[1]) -> +R]]) -> R",
    ),
    (lambda x: type(x).spam.ham, "[R](x: Has['spam', ClassVar[+Has['ham', +R]]]) -> R"),
    (
        lambda x: (type(x).spam, type(x).spam),
        "[R](x: Has['spam', ClassVar[+R]]) -> tuple[R, R]",
    ),
    (
        lambda x: (x.spam, type(x).spam),
        "[R, R2](x: Has['spam', +R] & Has['spam', ClassVar[+R2]]) -> tuple[R, R2]",
    ),
    (_set_class_attr, "(x: Has['spam', ClassVar[-Literal[1]]]) -> None"),
    (_del_class_attr, "(x: Has['spam', ClassVar]) -> None"),
    (_get_class_attr, "(x: Has['spam', ClassVar]) -> None"),
]

BINARY_CASES: list[tuple[Callable[[Any, Any], Any], str]] = [
    # a positional-only parameter renders bare, a keyword-passable one by name
    (lambda x, /, y: (x, y), "[T, U](T, y: U) -> tuple[T, U]"),
    (
        lambda x, y: x * y,
        "[T, R](x: CanMul[T, R], y: T) -> R\n[T, R](x: T, y: CanRMul[T, R]) -> R",
    ),
    (
        lambda x, y: x**y,
        "[T, R](x: CanPow[T, R], y: T) -> R\n[T, R](x: T, y: CanRPow[T, R]) -> R",
    ),
    (
        lambda x, y: x + y,
        "[T, R](x: CanAdd[T, R], y: T) -> R\n[T, R](x: T, y: CanRAdd[T, R]) -> R",
    ),
    (
        lambda x, y: y + x,
        "[T, R](x: T, y: CanAdd[T, R]) -> R\n[T, R](x: CanRAdd[T, R], y: T) -> R",
    ),
    (
        lambda x, y: x + y * y,
        (
            "[T: CanMul[T, U], U, R](x: CanAdd[U, R], y: T) -> R\n"
            "[T, U: CanRMul[U, CanRAdd[T, R]], R](x: T, y: U) -> R"
        ),
    ),
    (lambda x, y: x[y], "[T, R](x: CanGetitem[T, R], y: T) -> R"),
    (lambda x, y: y[str(x)], "[R](x: CanStr, y: CanGetitem[str, R]) -> R"),
    # `sorted` compares the `key` results, so they must be mutually `<`-comparable
    (
        lambda xs, key: sorted(xs, key=key),
        "[T, R](xs: CanIter[CanNext[R]], key: (R) -> T & CanLt[T, CanBool]) -> list[R]",
    ),
    (lambda x, y: x, "[T](x: T, y: object) -> T"),  # noqa: ARG005
    (lambda x, y: (type(x), type(y)), "[T, U](x: T, y: U) -> tuple[type[T], type[U]]"),
    (
        lambda x, y: y + type(x)(),
        "[T, R](x: T, y: CanAdd[T, R]) -> R\n[T, R](x: CanRAdd[T, R], y: T) -> R",
    ),
    (lambda x, y: x or y, "[T: CanBool, U](x: T, y: U) -> T | U"),
    (lambda x, y: x if x in y else y, "[T, U: CanContains[T]](x: T, y: U) -> T | U"),
    (lambda x, y: x and y, "[T: CanBool, U](x: T, y: U) -> U | T"),
    # a predicate is stable per run, so a self-contradicting guard never traces `foo`
    (
        lambda x, y: y.foo() if (y in x and y not in x) else x,
        "[T: CanContains[U], U](x: T, y: U) -> T",
    ),
    (
        lambda x, y: y.foo() if (isinstance(y, x) and not isinstance(y, x)) else x,
        "[T: CanInstancecheck](x: T, y: object) -> T",
    ),
    (
        lambda x, y: y.foo() if (issubclass(y, x) and not issubclass(y, x)) else x,
        "[T: CanSubclasscheck](x: T, y: object) -> T",
    ),
    # distinct operands stay independent, so this guard is satisfiable and traces `foo`
    (
        lambda x, y: y.foo() if (y in x and 0 not in x) else x,
        (
            "[T: CanContains[Literal[0] | U], U: Has['foo', () -> +R], R]"
            "(x: T, y: U) -> T | R"
        ),
    ),
    (
        lambda x, y: -x if x else -y if y else x,
        "[T: CanBool & CanNeg[R], R, R2](x: T, y: CanBool & CanNeg[R2]) -> R | R2 | T",
    ),
    (
        lambda x, y: (x + y) if x else y,
        (
            "[T, R](x: CanBool & CanAdd[T, R], y: T) -> R | T\n"
            "[T: CanBool, U: CanRAdd[T, R], R](x: T, y: U) -> R | U"
        ),
    ),
    (
        lambda x, y: x[0] if len(x) else y,
        "[T, R](x: CanSequence[Literal[0], R], y: T) -> R | T",
    ),
    (
        lambda x, y: x[y] if len(x) else None,
        "[T, R](x: CanSequence[T, R], y: T) -> R | None",
    ),
    (
        lambda x, y: (x + y) if len(x) else y,
        (
            "[T, R](x: CanLen & CanAdd[T, R], y: T) -> R | T\n"
            "[T: CanLen, U: CanRAdd[T, R], R](x: T, y: U) -> R | U"
        ),
    ),
    # a coerced int flowing as *data* into another spy op pollutes the key
    (
        lambda x, y: x[int(y)],
        "[R](x: CanGetitem[Literal[1, 0], R], y: CanInt | CanIndex) -> R",
    ),
    (
        lambda x, y: x * 2 + y,
        (
            "[T, R](x: CanMul[Literal[2], CanAdd[T, R]], y: T) -> R\n"
            "[T, R](x: CanMul[Literal[2], T], y: CanRAdd[T, R]) -> R"
        ),
    ),
    (
        lambda x, y: (x + y, y * 2),
        (
            "[T: CanMul[Literal[2], R2], R, R2](x: CanAdd[T, R], y: T)"
            " -> tuple[R, R2]\n"
            "[T, R, R2](x: T, y: CanMul[Literal[2], R2] & CanRAdd[T, R])"
            " -> tuple[R, R2]"
        ),
    ),
    (
        lambda x, y: (x + y, y + x),
        (
            "[T: CanAdd[U, R], U: CanAdd[T, R2], R, R2](x: T, y: U) -> tuple[R, R2]\n"
            "[T: CanRAdd[U, R2], U: CanRAdd[T, R], R, R2](x: T, y: U) -> tuple[R, R2]"
        ),
    ),
    # a synthesized attribute assignment binds the assigned value's type
    (lambda x, y: setattr(x, "spam", y), "[T](x: Has['spam', -T], y: T) -> None"),
]


def _unused_args(x: Any, *_args: Any) -> Any:
    return x + 1


def _call_packed(x: Any, *args: Any) -> Any:
    return x(args)


def _prepend(x: Any, *args: Any) -> Any:
    return (x, *args)


def _unpack3(*args: Any) -> Any:
    _a, b, _c = args
    return b


def _takes3(a: Any, _b: Any, _c: Any, /) -> Any:
    return a


def _takes9(
    a: Any,
    _b: Any,
    _c: Any,
    _d: Any,
    _e: Any,
    _f: Any,
    _g: Any,
    _h: Any,
    _i: Any,
    /,
) -> Any:
    # exactly 9 positional parameters; its arity falls in a former yield-budget gap
    return a


def _spread(*args: Any) -> Any:
    return _takes3(*args)


def _call_spread(f: Any, *args: Any) -> Any:
    return f(*args)


def _kwargs_key(**kwargs: Any) -> Any:
    return kwargs["key"]


def _var_kwargs(**kwargs: Any) -> Any:
    return kwargs


def _sum_kwargs(**kwargs: Any) -> Any:
    return sum(kwargs.values())


VARIADIC_CASES: list[tuple[Callable[..., Any], str]] = [
    # a variadic parameter whose every use is packed becomes a TypeVarTuple
    (lambda *args: args, "[*Ts](*args: *Ts) -> tuple[*Ts]"),
    (lambda *args: (args, 1), "[*Ts](*args: *Ts) -> tuple[tuple[*Ts], Literal[1]]"),
    (_call_packed, "[*Ts, R](x: (tuple[*Ts]) -> R, *args: *Ts) -> R"),
    # a star-unpack splices the TypeVarTuple into the surrounding tuple
    (lambda *args: (1, *args), "[*Ts](*args: *Ts) -> tuple[Literal[1], *Ts]"),
    (
        lambda *args: (1, *args, 2),
        "[*Ts](*args: *Ts) -> tuple[Literal[1], *Ts, Literal[2]]",
    ),
    (_prepend, "[T, *Ts](x: T, *args: *Ts) -> tuple[T, *Ts]"),
    # a bare element use is inexpressible with a TypeVarTuple, so all elements
    # share a single element type instead
    (lambda *args: list(args), "[T](*args: T) -> list[T]"),
    (lambda *args: args[0], "[T](*args: T) -> T"),
    # too few placeholders are retried with more, until the call succeeds
    (lambda *args: args[2], "[T](*args: T) -> T"),
    (lambda *args: args[100], "[T](*args: T) -> T"),
    (_unpack3, "[T](*args: T) -> T"),
    (_spread, "[T](*args: T) -> T"),
    # a missing `**kwargs` key is injected and retried
    (_kwargs_key, "[T](**kwargs: T) -> T"),
    (lambda *args: (args[0], *args), "[T](*args: T) -> tuple[T, ...]"),
    (
        lambda *args: args[0] + args[1],
        "[T: CanAdd[T, R], R](*args: T) -> R\n[T: CanRAdd[T, R], R](*args: T) -> R",
    ),
    (
        lambda *args: [a + 1 for a in args],
        "[R](*args: CanAdd[Literal[1], R]) -> list[R]",
    ),
    # an unused variadic parameter is dropped, like an unused default
    (_unused_args, "[R](x: CanAdd[Literal[1], R]) -> R"),
    # the fixed placeholder count leaks through `len`
    (lambda *args: (args[0], len(args)), "[T](*args: T) -> tuple[T, Literal[2]]"),
    (_var_kwargs, "[T](**kwargs: T) -> dict[str, T]"),
    (_sum_kwargs, "[R](**kwargs: CanRAdd[Literal[0], R]) -> R"),
    # a variadic spread into a callable collapses to `*tuple[T, ...]` (gh-687)
    (_call_spread, "[T, R](f: (*tuple[T, ...]) -> R, *args: T) -> R"),
]


def _int_default(x: Any = 0) -> Any:
    return x


def _none_default(x: Any = None) -> Any:
    return x


def _float_default(x: Any = 1.5) -> Any:
    return x


def _add_one_default(x: Any = 0) -> Any:
    return x + 1


def _mul_defaults(x: Any = 1, y: Any = 2) -> Any:
    return x * y


def _add_default(x: Any, y: Any = 0) -> Any:
    return x + y


def _getitem_default(x: Any, *, y: Any = 1) -> Any:
    return x[y]


def _yield_default(x: Any = 0) -> Any:
    yield x


def _set_attr_default(x: Any, y: Any = 1) -> None:
    x.spam = y


def _if_none(x: Any = None) -> Any:
    return [] if x is None else x


def _or_default(x: Any = None) -> None:
    _: Any = x or []


def _str_default(x: Any = 0) -> Any:
    return str(x)


def _type_default(x: object = 0) -> type[object]:
    return type(x)


def _unused_default(x: Any, _y: Any = 0) -> Any:
    return x + 1


DEFAULT_CASES: list[tuple[Callable[..., Any], str]] = [
    # the typevar of a defaulted parameter gets a PEP 696 default when omitting
    # the argument behaves like substituting its value
    (_int_default, "[T = Literal[0]](x: T = 0) -> T"),
    (_none_default, "[T = None](x: T = None) -> T"),
    (_float_default, "[T = float](x: T = 1.5) -> T"),
    (
        _add_default,
        (
            "[R, T = Literal[0]](x: CanAdd[T, R], y: T = 0) -> R\n"
            "[T, R](x: T, y: CanRAdd[T, R] = 0) -> R"
        ),
    ),
    (_getitem_default, "[R, T = Literal[1]](x: CanGetitem[T, R], y: T = 1) -> R"),
    (_set_attr_default, "[T = Literal[1]](x: Has['spam', -T], y: T = 1) -> None"),
    (_yield_default, "[T = Literal[0]](x: T = 0) -> Generator[T]"),
    (_type_default, "[T = Literal[0]](x: T = 0) -> type[T]"),
    # a parameter whose typevar would only appear once shows the default inline
    (_or_default, "(x: CanBool = None) -> None"),
    (_str_default, "(x: CanStr = 0) -> str"),
    # when omission behaves differently, the omitted call is its own overload,
    # which also covers passing the default explicitly; a single default's type
    # is excluded from the generic overload, so that the overloads are disjoint
    (_if_none, "(x: None = None) -> list[Never]\n[T: ~None](x: T) -> T"),
    (
        _add_one_default,
        "(x: Literal[0] = 0) -> int\n[R](x: CanAdd[Literal[1], R] & ~Literal[0]) -> R",
    ),
    # with several defaults, each one also gets an overload of its own
    (
        _mul_defaults,
        (
            "(x: Literal[1] = 1, y: Literal[2] = 2) -> int\n"
            "[R](x: Literal[1] = 1, y: CanRMul[Literal[1], R]) -> R\n"
            "[R](x: CanMul[Literal[2], R], y: Literal[2] = 2) -> R\n"
            "[T, R](x: CanMul[T, R], y: T) -> R\n"
            "[T, R](x: T, y: CanRMul[T, R]) -> R"
        ),
    ),
    # an unused default is dropped from the signature either way
    (_unused_default, "[R](x: CanAdd[Literal[1], R]) -> R"),
]


def _self_return(x: Any) -> Any:  # noqa: ARG001
    return _self_return


def _fn_factory(n: Any) -> Any:
    return lambda: _fn_factory(n)


def _fn_default(x: Any = 0) -> Any:
    return lambda: x


def _add2(x: Any, y: Any) -> Any:
    return x + y


def _counter(start: Any) -> Any:
    return (lambda: start), (lambda by: start + by)


def _yield_fn(x: Any) -> Any:
    yield lambda: x


FUNCTION_CASES: list[tuple[Callable[..., Any], str]] = [
    # a returned function is lazy, so it is explored with placeholders of its own
    # and renders in signature syntax; its parameter spies are named like parameters
    (lambda x: lambda y: (x, y), "[T, U](x: T) -> (y: U) -> tuple[T, U]"),
    (lambda x: lambda: x, "[T](x: T) -> () -> T"),
    (lambda x: lambda y: y, "[T](x: object) -> (y: T) -> T"),  # noqa: ARG005
    (lambda x: lambda *, y: (x, y), "[T, U](x: T) -> (y: U) -> tuple[T, U]"),
    (lambda x: lambda y=1: (x, y), "[T, U](x: T) -> (y: U = 1) -> tuple[T, U]"),
    # ...except for a positional-only parameter, which renders without its name
    (lambda x: lambda y, /: (x, y), "[T, U](x: T) -> (U) -> tuple[T, U]"),
    (lambda x: lambda y=1, /: (x, y), "[T, U](x: T) -> (U = 1) -> tuple[T, U]"),
    (
        lambda x: lambda y: lambda z: (x, y, z),
        "[T, U, V](x: T) -> (y: U) -> (z: V) -> tuple[T, U, V]",
    ),
    # an operation inside the returned function is required of the closed-over
    # parameter, and is reflected onto its right-hand side like any other
    (
        lambda x: lambda y: x + y,
        (
            "[T, R](x: CanAdd[T, R]) -> (y: T) -> R\n"
            "[T, R](x: T) -> (y: CanRAdd[T, R]) -> R"
        ),
    ),
    (lambda x: lambda f: f(x), "[T, R](x: T) -> (f: (T) -> R) -> R"),
    (
        lambda x: lambda y: y.foo,  # noqa: ARG005
        "[R](x: object) -> (y: Has['foo', +R]) -> R",
    ),
    # a failed inner run rolls back its traces on the closed-over parameter, so
    # an optionally probed `__len__` is no requirement, just like in a direct call
    (lambda x: lambda: len(x), "(x: CanLen) -> () -> int"),
    (lambda x: lambda: list(x), "[R](x: CanIter[CanNext[R]]) -> () -> list[R]"),
    # a returned builtin or method descriptor explores like a direct `infer`,
    # including the lenient fallback that pins a rejected defaulted parameter
    (lambda: len, "() -> (CanLen) -> int"),
    (lambda: math.sqrt, "() -> (CanFloat | CanIndex) -> float"),
    (lambda: str.upper, "() -> (str) -> str"),
    (
        lambda: str.split,
        "() -> (str, sep: None = None, maxsplit: CanIndex = -1) -> list[Never]",
    ),
    (
        lambda: dict.get,
        "[T]() -> (dict, CanHash, T = None) -> T",
    ),
    # a `functools.partial` explores with its bound arguments in place
    (
        lambda: functools.partial(_add2, 1),
        "[R]() -> (y: CanRAdd[Literal[1], R]) -> R",
    ),
    (
        lambda x: functools.partial(_add2, x),
        (
            "[T, R](x: CanAdd[T, R]) -> (y: T) -> R\n"
            "[T, R](x: T) -> (y: CanRAdd[T, R]) -> R"
        ),
    ),
    # generic `functools` wrappers, parameterized by the wrapped return type (#724)
    (
        lambda f: functools.partial(f, 1),
        "[R](f: (Literal[1]) -> R) -> functools.partial[R]",
    ),
    (
        lambda f, x: functools.partial(f, x),
        "[T, R](f: (T) -> R, x: T) -> functools.partial[R]",
    ),
    (
        lambda f: functools.partialmethod(f, 1),
        "[R](f: (Literal[1]) -> R) -> functools.partialmethod[R]",
    ),
    (
        lambda f: functools.cached_property(f),
        "[R](f: (object) -> R) -> functools.cached_property[R]",
    ),
    # the wrapper classes themselves: `func` is a callable, not `object`. `partial` is
    # a C type whose signature is only introspectable on 3.13+; older versions probe
    # arities and miss `**keywords`
    (
        functools.partial,
        (
            "[T, U, R]((*tuple[T, ...], U) -> R, *args: T, **keywords: U) "
            "-> functools.partial[R]"
            if sys.version_info >= (3, 13)
            else "[T, R]((*tuple[T, ...]) -> R, *args: T) -> functools.partial[R]"
        ),
    ),
    (
        functools.partialmethod,
        (
            "[T, U, R]((*tuple[T, ...], U) -> R, *args: T, **keywords: U) "
            "-> functools.partialmethod[R]"
        ),
    ),
    (
        functools.cached_property,
        "[R](func: (object) -> R) -> functools.cached_property[R]",
    ),
    # a non-spy callable can't be called for its return, so the bare wrapper renders
    (lambda: functools.partial(print, "a"), "() -> functools.partial"),
    # but a `partial` of an explorable real function keeps its reduced call signature,
    # rendered as a callable rather than as `functools.partial[R]`
    (
        lambda: functools.partial(divmod, 10),
        "[R]() -> (CanRDivmod[Literal[10], R]) -> R",
    ),
    # a function within a returned container is explored as well, but a set member
    # must stay hashable, so it is left as-is
    (
        lambda x: (lambda y: y, 1),  # noqa: ARG005
        "[T](x: object) -> tuple[(y: T) -> T, Literal[1]]",
    ),
    (lambda x: [lambda: x], "[T](x: T) -> list[() -> T]"),
    (lambda x: {"get": lambda: x}, "[T](x: T) -> dict[Literal['get'], () -> T]"),
    (lambda x: {lambda: x}, "(x: object) -> set[FunctionType]"),
    # ...and so is a function within the yields of a generator
    (_yield_fn, "[T](x: T) -> Generator[() -> T]"),
    (
        _counter,
        (
            "[T: CanAdd[U, R], U, R](start: T) -> tuple[() -> T, (by: U) -> R]\n"
            "[T, R](start: T) -> tuple[() -> T, (by: CanRAdd[T, R]) -> R]"
        ),
    ),
    # a function type in a union is parenthesized; a covariant return type lets
    # one union member absorb the other
    (lambda x: (lambda: 1) if x else None, "(x: CanBool) -> (() -> int) | None"),
    (lambda x: (lambda: True) if x else (lambda: 1), "(x: CanBool) -> () -> int"),
    # a returned function returning a generator or a packed `*args` traces through
    (
        lambda x: lambda: (i for i in x),
        "[R](x: CanIter[CanNext[R]]) -> () -> Generator[R]",
    ),
    (lambda *args: lambda: args, "[*Ts](*args: *Ts) -> () -> tuple[*Ts]"),
    # a recursive function (factory) has an inexpressible type, so it stays opaque,
    # as do variadic parameters
    (_self_return, "(x: object) -> FunctionType"),
    (_fn_factory, "(n: object) -> () -> FunctionType"),
    (lambda x: lambda *args: x, "(x: object) -> FunctionType"),  # noqa: ARG005
]


def _unpack_pair(x: Any) -> Any:
    a, b = x
    return a, b


def _unpack_star(x: Any) -> Any:
    a, *b = x
    return a, b


def _unpack_iter(x: Any) -> Any:
    a, b = iter(x)
    return a, b


def _divmod_unpack(x: Any, y: Any) -> Any:
    div, mod = divmod(x, y)
    return div, mod


ITERATOR_CASES: list[tuple[Callable[..., Any], str]] = [
    # lazy builtin iterators are iterated like generators
    (lambda x: map(str, x), "(x: CanIter[CanNext[CanStr]]) -> map[str]"),
    (
        lambda f, x: map(f, x),
        "[T, R](f: (T) -> R, x: CanIter[CanNext[T]]) -> map[R]",
    ),
    # the variadic iterables collapse into a `*tuple[T, ...]` mapper (gh-687)
    (
        lambda f, *iterables: map(f, *iterables),
        "[T, R](f: (*tuple[T, ...]) -> R, *iterables: CanIter[CanNext[T]]) -> map[R]",
    ),
    (
        lambda x: map(lambda v: lambda: v, x),
        "[R](x: CanIter[CanNext[R]]) -> map[() -> R]",
    ),
    (lambda x: filter(None, x), "[R: CanBool](x: CanIter[CanNext[R]]) -> filter[R]"),
    (
        lambda x, y: zip(x, y),  # noqa: B905
        "[R, R2](x: CanIter[CanNext[R]], y: CanIter[CanNext[R2]]) -> zip[tuple[R, R2]]",
    ),
    # a variadic `*iterables` makes the zipped tuple homogeneous and variadic (gh-688)
    (
        lambda *iterables: zip(*iterables),  # noqa: B905
        "[R](*iterables: CanIter[CanNext[R]]) -> zip[tuple[R, ...]]",
    ),
    # known limitation: a fixed tuple of count-many variadic elements is
    # indistinguishable from a `zip(*a)` spread, so it collapses to `tuple[R, ...]`
    (
        lambda *a: (next(iter(a[0])), next(iter(a[0]))),
        "[R](*a: CanIter[CanNext[R]]) -> tuple[R, ...]",
    ),
    # `enumerate[R]` is parameterized by the element type, not the yielded pair
    (lambda x: enumerate(x), "[R](x: CanIter[CanNext[R]]) -> enumerate[R]"),
    # only `zip` is covariant in typeshed, so an `enumerate` union does not absorb
    (
        lambda x: zip([FileNotFoundError()]) if x else zip([OSError()]),
        "(x: CanBool) -> zip[tuple[OSError]]",
    ),
    (
        lambda x: enumerate([True]) if x else enumerate([1]),
        "(x: CanBool) -> enumerate[bool] | enumerate[int]",
    ),
    (lambda: map(str, [1, 2]), "() -> map[str]"),
    (lambda: zip((), ()), "() -> zip[Never]"),  # noqa: B905
    # generic `itertools` iterators (#722)
    (
        lambda xs: itertools.chain(xs),
        "[R](xs: CanIter[CanNext[R]]) -> itertools.chain[R]",
    ),
    (
        lambda *xss: itertools.chain(*xss),
        "[R](*xss: CanIter[CanNext[R]]) -> itertools.chain[R]",
    ),
    (
        lambda xs: itertools.cycle(xs),
        "[R](xs: CanIter[CanNext[R]]) -> itertools.cycle[R]",
    ),
    (
        lambda xs, sel: itertools.compress(xs, sel),
        (
            "[R](xs: CanIter[CanNext[R]], sel: CanIter[CanNext[CanBool]]) "
            "-> itertools.compress[R]"
        ),
    ),
    (
        lambda f, xs: itertools.takewhile(f, xs),
        "[R](f: (R) -> CanBool, xs: CanIter[CanNext[R]]) -> itertools.takewhile[R]",
    ),
    (
        lambda xs: itertools.pairwise(xs),
        "[R](xs: CanIter[CanNext[R]]) -> itertools.pairwise[tuple[R, R]]",
    ),
    (
        lambda *xss: itertools.zip_longest(*xss),
        "[R](*xss: CanIter[CanNext[R]]) -> itertools.zip_longest[tuple[R, ...]]",
    ),
    # `tee` yields `Iterator`, not `_tee` (#722); the list avoids `tee`'s `__copy__`
    # shortcut, which varies across CPython 3.12 patches
    (
        lambda x: itertools.tee([x]),
        "[T](x: T) -> tuple[Iterator[T], Iterator[T]]",
    ),
    # an unrecoverable element type renders bare (#722)
    (
        lambda f, xs: itertools.dropwhile(f, xs),
        "[T](f: (T) -> CanBool, xs: CanIter[CanNext[T]]) -> itertools.dropwhile",
    ),
    (
        lambda f, xs: itertools.filterfalse(f, xs),
        "[T](f: (T) -> CanBool, xs: CanIter[CanNext[T]]) -> itertools.filterfalse",
    ),
    (
        lambda x: ((i for i in x), 1),
        "[R](x: CanIter[CanNext[R]]) -> tuple[Generator[R], Literal[1]]",
    ),
    # fixed-size unpacking (#683)
    (_unpack_pair, "[R](x: CanIter[CanNext[R]]) -> tuple[R, R]"),
    (_unpack_star, "[R](x: CanIter[CanNext[R]]) -> tuple[R, list[R]]"),
    (_unpack_iter, "[R](x: CanIter[CanNext[R]]) -> tuple[R, R]"),
    (
        lambda x: {k: v for k, v in x},  # noqa: C416
        "[R: CanHash](x: CanIter[CanNext[CanIter[CanNext[R]]]]) -> dict[R, R]",
    ),
    # a star-unpack into a fixed-arity call (#683)
    (
        lambda x, y: divmod(*divmod(x, y)),  # type: ignore[misc]
        (
            "[T, U: CanDivmod[U, R], R]"
            "(x: CanDivmod[T, CanIter[CanNext[U]]], y: T) -> R\n"
            "[T, U: CanRDivmod[U, R], R]"
            "(x: T, y: CanRDivmod[T, CanIter[CanNext[U]]]) -> R"
        ),
    ),
    (
        _divmod_unpack,
        (
            "[T, R](x: CanDivmod[T, CanIter[CanNext[R]]], y: T) -> tuple[R, R]\n"
            "[T, R](x: T, y: CanRDivmod[T, CanIter[CanNext[R]]]) -> tuple[R, R]"
        ),
    ),
]


INFER_CASES = [
    *UNARY_CASES,
    *BINARY_CASES,
    *VARIADIC_CASES,
    *DEFAULT_CASES,
    *FUNCTION_CASES,
    *ITERATOR_CASES,
]


@pytest.mark.parametrize(
    ("func", "expected"),
    INFER_CASES,
    ids=[f"{i}:{e.splitlines()[0]}" for i, (_, e) in enumerate(INFER_CASES)],
)
def test_infer(func: Callable[..., Any], expected: str) -> None:
    assert infer(func) == expected


SELECT_CASES: list[tuple[Callable[[Any, Any], Any], tuple[str | int, ...], str]] = [
    (lambda x, y: x[y], ("x",), "[T, R](x: CanGetitem[T, R]) -> R"),
    (lambda x, y: x[y], ("y",), "[T, R](y: T) -> R"),
    (lambda x, y: x[y], (0,), "[T, R](x: CanGetitem[T, R]) -> R"),
    (lambda x, y: x[y], (-1,), "[T, R](y: T) -> R"),
    (lambda x, y: x[y], ("x", "y"), "[T, R](x: CanGetitem[T, R], y: T) -> R"),
    (lambda x, y: x[y], ("y", "x"), "[T, R](y: T, x: CanGetitem[T, R]) -> R"),
    (lambda x, y: x[y], (1, 0), "[T, R](y: T, x: CanGetitem[T, R]) -> R"),
    (lambda x, y: x[y], (), "[T, R](x: CanGetitem[T, R], y: T) -> R"),
]


@pytest.mark.parametrize(("func", "params", "expected"), SELECT_CASES)
def test_infer_select(
    func: Callable[[Any, Any], Any],
    params: tuple[str | int, ...],
    expected: str,
) -> None:
    assert infer(func, *params) == expected


def _str(x: Any) -> Any:
    return str(x)


def _repr(x: Any) -> Any:
    return repr(x)


def _bytes(x: Any) -> Any:
    return bytes(x)


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        (_str, "(x: CanStr) -> str"),
        (_repr, "(x: CanRepr) -> str"),
        (_bytes, "(x: CanBytes) -> bytes"),
    ],
)
def test_stringify(func: Callable[[Any], Any], expected: str) -> None:
    assert infer(func) == expected


SUBTYPE_CASES: list[tuple[Any, Any, bool]] = [
    (Type(bool), Type(int), True),
    (Type(int), Type(bool), False),
    (Type(int), Type(float), False),  # PEP 484's numeric tower is not real
    (Type(FileNotFoundError), Type(OSError), True),
    (Type(OSError), Type(FileNotFoundError), False),
    (Name("Never"), Type(str), True),
    (Type(str), Name("object"), True),
    (Type(str), Type(object), True),
    (Lit((True, 1)), Type(int), True),
    (Lit((1, "a")), Type(int), False),
    (Lit((1,)), Lit((1, 2)), True),
    (Lit((True,)), Lit((1,)), False),  # Literal[True] is not Literal[1]
    # the yield type is covariant
    (App("Generator", (Type(bool),)), App("Generator", (Type(int),)), True),
    (App("Generator", (Type(int),)), App("Generator", (Type(bool),)), False),
    # the send type is contravariant
    (
        App("Generator", (Type(int), Type(int), Type(int))),
        App("Generator", (Type(int), Type(bool), Type(int))),
        True,
    ),
    (
        App("Generator", (Type(int), Type(bool), Type(int))),
        App("Generator", (Type(int), Type(int), Type(int))),
        False,
    ),
    # tuple is covariant in every element; list is invariant
    (App("tuple", (Type(bool), Lit((1,)))), App("tuple", (Type(int),) * 2), True),
    (App("tuple", (Type(bool),)), App("tuple", (Type(int),) * 2), False),
    (App("list", (Type(bool),)), App("list", (Type(int),)), False),
    # type is covariant
    (App("type", (Type(bool),)), App("type", (Type(int),)), True),
    (App("type", (Type(int),)), App("type", (Type(bool),)), False),
    # a function's parameters are contravariant, its return type is covariant
    (Fn((), Type(bool)), Fn((), Type(int)), True),
    (Fn((), Type(int)), Fn((), Type(bool)), False),
    (
        Fn((Arg("x", Type(int)),), Type(int)),
        Fn((Arg("x", Type(bool)),), Type(int)),
        True,
    ),
    (
        Fn((Arg("x", Type(bool)),), Type(int)),
        Fn((Arg("x", Type(int)),), Type(int)),
        False,
    ),
    (Fn((), Type(int)), Fn((Arg("x", Type(int)),), Type(int)), False),
    # `zip` is covariant in typeshed; `map`, `filter`, and `enumerate` are invariant
    (App("zip", (Type(bool),)), App("zip", (Type(int),)), True),
    (App("map", (Type(bool),)), App("map", (Type(int),)), False),
    (App("enumerate", (Type(bool),)), App("enumerate", (Type(int),)), False),
]


@pytest.mark.parametrize(("sub", "sup", "expected"), SUBTYPE_CASES)
def test_subtype(sub: Any, sup: Any, expected: bool) -> None:
    assert subtype(sub, sup) is expected


def test_return_union_parenthesized() -> None:
    # a union intersected with a typevar is parenthesized, also in return position
    def f(x: Any) -> Any:
        return [x[0], int(x[1])]

    assert infer(f) == (
        "[R](x: CanGetitem[Literal[0, 1], R & (CanInt | CanIndex)])"
        " -> list[Literal[1] | R] | list[Literal[0] | R]"
    )


def test_keyword_only() -> None:
    def f(x: Any, *, y: Any) -> Any:
        return x[y]

    assert infer(f) == "[T, R](x: CanGetitem[T, R], y: T) -> R"


def test_callable_instance() -> None:
    class Add1:
        def __call__(self, x: Any) -> Any:
            return x + 1

    assert infer(Add1()) == "[R](x: CanAdd[Literal[1], R]) -> R"


def test_method_descriptor() -> None:
    # an unbound method descriptor's `self` requires a real `__objclass__` instance
    assert infer(str.upper) == "(str) -> str"
    assert infer(int.bit_length) == "(int) -> int"
    assert infer(float.hex) == "(float) -> str"
    assert infer(list[Any].append) == "(list, object) -> None"
    assert infer(object.__str__) == "(object) -> str"
    assert infer(dict[Any, Any].get) == "[T = None](dict, CanHash, T = None) -> T"
    # `memoryview()` is constructed from a spy through its `__buffer__`
    assert infer(memoryview.tobytes) == "(memoryview, order: str = 'C') -> bytes"


def test_buffer_spy_finalizes_without_cycle() -> None:
    # A cycle that holds a buffer-exporting spy until cyclic GC makes its C-level
    # `__release_buffer__` lookup fail as an unraisable. Guards the weak typer->renderer
    # link; a strong ref would regress this silently (pytest ignores unraisables).
    captured: list[object] = []
    hook = sys.unraisablehook
    sys.unraisablehook = lambda args: captured.append(args.err_msg)
    enabled = gc.isenabled()
    gc.disable()
    try:
        infer(memoryview.tobytes)
        gc.collect()
    finally:
        if enabled:
            gc.enable()
        sys.unraisablehook = hook
    assert not captured


def test_method_descriptor_fixed_defaults() -> None:
    # a defaulted parameter whose spy the function rejects passes its default
    # instead, while the accepting parameters stay structural
    assert infer(str.split) == (
        "(str, sep: None = None, maxsplit: CanIndex = -1) -> list[Never]"
    )
    assert infer(bytes.decode) == (
        "(bytes, encoding: str = 'utf-8', errors: str = 'strict') -> str"
    )
    # a rejected default widens to its type, both for the parameter and where the
    # value flows on (the `format_spec` reaches `value.__format__`)
    assert infer(format) == "(CanFormat[str], str = '') -> str"


def test_method_descriptor_unsupported() -> None:
    # generators cannot be constructed, spy-rejecting parameters without a default
    # have nothing to fall back on, and an empty `self` cannot always run
    send = type(x for x in range(0)).send
    # before 3.13, `generator.send` has no `inspect.signature` to explore at all
    with pytest.raises(InferError, match=r"instantiate 'generator'|no signature"):
        infer(send)
    with pytest.raises(InferError, match="expected str instance"):
        infer(str.join)
    with pytest.raises(InferError, match="pop from empty list"):
        infer(list[Any].pop)
    with pytest.raises(InferError, match="dictionary is empty"):
        infer(dict.popitem)


def test_ternary_pow() -> None:
    # the optional modulo is used forward but dropped from the reflected overload
    def f(x: Any, y: Any, z: Any = None) -> Any:
        return x.__pow__(y, z)  # noqa: PLC2801

    assert infer(f) == (
        "[T, R, U = None](x: CanPow[T, U, R], y: T, z: U = None) -> R\n"
        "[T, R](x: T, y: CanRPow[T, R]) -> R"
    )


def test_deprecated() -> None:
    # a `@deprecated` callable's `DeprecationWarning` becomes an `@deprecated` marker
    @deprecated("Use bar instead")
    def foo(x: Any) -> Any:
        return x + 1

    assert infer(foo) == (  # pyright: ignore[reportDeprecated]
        "@deprecated('Use bar instead')\n[R](x: CanAdd[Literal[1], R]) -> R"
    )


def test_deprecated_warn() -> None:
    # a plain `warnings.warn(..., DeprecationWarning)` counts too
    def foo(x: Any) -> Any:
        warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
        return x + 1

    assert infer(foo) == (
        "@deprecated('deprecated')\n[R](x: CanAdd[Literal[1], R]) -> R"
    )


def test_deprecated_overload() -> None:
    # only the call forms that raise the warning are marked: omitting `y` stays quiet
    def foo(x: Any, y: Any = None) -> Any:
        if y is None:
            return x + 1
        warnings.warn("y is deprecated", DeprecationWarning, stacklevel=2)
        return x + y

    assert infer(foo) == (
        "[R](x: CanAdd[Literal[1], R], y: None = None) -> R\n"
        "@deprecated('y is deprecated')\n"
        "[T: ~None, R](x: CanAdd[T, R], y: T) -> R\n"
        "@deprecated('y is deprecated')\n"
        "[T, R](x: T, y: CanRAdd[T, R] & ~None) -> R"
    )


def test_deprecated_operator() -> None:
    # both the forward and reflected overloads are marked
    @deprecated("old op")
    def foo(x: Any, y: Any) -> Any:
        return x * y

    assert infer(foo) == (  # pyright: ignore[reportDeprecated]
        "@deprecated('old op')\n"
        "[T, R](x: CanMul[T, R], y: T) -> R\n"
        "@deprecated('old op')\n"
        "[T, R](x: T, y: CanRMul[T, R]) -> R"
    )


def test_infer_with() -> None:
    # a `with` statement requires `__enter__` and `__exit__` together, which is the
    # combined `CanWith`; its unused `__exit__` result is unconstrained
    def f(x: Any) -> Any:
        with x as y:
            return y

    assert infer(f) == "[R](x: CanWith[R, object]) -> R"

    def g(x: Any) -> Any:
        with x as y:
            return y + 1

    assert infer(g) == "[R](x: CanWith[CanAdd[Literal[1], R], object]) -> R"

    # a lone `__enter__` or `__exit__` does not imply the other, so it stays as-is
    def enter_only(x: Any) -> Any:
        x.__enter__()  # noqa: PLC2801
        return x

    assert infer(enter_only) == "[T: CanEnter[object]](x: T) -> T"

    def exit_only(x: Any) -> Any:
        return x.__exit__(None, None, None)

    assert infer(exit_only) == "(x: CanExit[None, None, None]) -> None"


def test_infer_async() -> None:
    # coroutine functions are driven to completion; await/async with/async for trace
    async def aw(x: Any) -> Any:
        return (await x) + 1

    assert infer(aw) == "[R](x: CanAwait[CanAdd[Literal[1], R]]) -> R"

    async def async_with(x: Any) -> Any:
        async with x as y:
            return y

    # like `CanWith`, but its declared parameters are the awaited results
    assert infer(async_with) == "[R](x: CanAsyncWith[R, object]) -> R"

    async def aenter_only(x: Any) -> Any:
        return await x.__aenter__()  # noqa: PLC2801

    assert infer(aenter_only) == "[R](x: CanAEnter[CanAwait[R]]) -> R"

    async def aexit_only(x: Any) -> Any:
        return await x.__aexit__(None, None, None)

    assert infer(aexit_only) == "[R](x: CanAExit[None, None, None, CanAwait[R]]) -> R"

    async def both(x: Any) -> Any:
        with x:
            async with x as y:
                return y

    assert infer(both) == (
        "[R](x: CanWith[object, object] & CanAsyncWith[R, object]) -> R"
    )

    async def async_for(x: Any) -> Any:
        async for item in x:
            return item
        return None

    assert infer(async_for) == "[R](x: CanAIter[CanANext[CanAwait[R]]]) -> R"


def test_infer_anext() -> None:
    # 2-arg `anext` returns a coroutine resolving to the value or the default
    assert infer(anext) == (
        "[R](CanANext[R]) -> R\n"
        "[T, R](CanANext[CanAwait[R]], T) -> Coroutine[object, None, R | T]"
    )


def test_infer_next_default() -> None:
    # 2-arg `next` returns the value or the default on exhaustion
    assert infer(next) == ("[R](CanNext[R]) -> R\n[T, R](CanNext[R], T) -> R | T")


def test_infer_generator() -> None:
    # generator expressions are lazy, so they are iterated to trace what they yield
    def gen(xs: Any) -> Any:
        return (x for x in xs)

    assert infer(gen) == "[R](xs: CanIter[CanNext[R]]) -> Generator[R]"

    def gen_add(xs: Any) -> Any:
        return (x + 1 for x in xs)

    assert infer(gen_add) == (
        "[R](xs: CanIter[CanNext[CanAdd[Literal[1], R]]]) -> Generator[R]"
    )

    async def async_gen(xs: Any) -> Any:
        return (x async for x in xs)

    assert infer(async_gen) == (
        "[R](xs: CanAIter[CanANext[CanAwait[R]]]) -> AsyncGenerator[R]"
    )


def test_inline_renumbers_survivors() -> None:
    # `str`'s element is single-use and inlines away; the surviving `map(g, y)`
    # typevar, first named `U`, must renumber to `T` to keep the parameters gapless
    assert infer(lambda x, g, y: (map(str, x), map(g, y))) == (
        "[T, R](x: CanIter[CanNext[CanStr]], g: (T) -> R, y: CanIter[CanNext[T]])"
        " -> tuple[map[str], map[R]]"
    )


def test_infer_generator_yields() -> None:
    # heterogeneous yields are sampled and unioned; an empty generator yields Never
    def hetero() -> Any:
        yield None
        yield 1

    assert infer(hetero) == "() -> Generator[None | int]"

    def empty() -> Any:
        yield from ()

    assert infer(empty) == "() -> Generator[Never]"

    # a yield whose type is a (nominal) subtype of another is absorbed into it
    def nominal() -> Any:
        yield True
        yield 1

    assert infer(nominal) == "() -> Generator[int]"

    def raisable() -> Any:
        yield FileNotFoundError()
        yield OSError()

    assert infer(raisable) == "() -> Generator[OSError]"

    # user-defined subclass relations are recognized as well
    class Base: ...

    class Child(Base): ...

    def custom() -> Any:
        yield Child()
        yield Base()

    assert infer(custom) == "() -> Generator[Base]"

    # an infinite generator terminates once a yield shape repeats, without exploding R
    def loop(x: Any) -> Any:
        while True:
            yield x + 1

    assert infer(loop) == "[R](x: CanAdd[Literal[1], R]) -> Generator[R]"


def test_returned_function_default() -> None:
    # a typevar default reaches through the returned function's body
    assert infer(_fn_default) == "[T = Literal[0]](x: T = 0) -> () -> T"


def test_returned_function_async() -> None:
    # an inner coroutine function is driven to completion, like the outer one
    async def make(x: Any) -> Any:  # noqa: RUF029
        async def inner(y: Any) -> Any:  # noqa: RUF029
            return (x, y)

        return inner

    assert infer(make) == "[T, U](x: T) -> (y: U) -> tuple[T, U]"


def test_returned_function_mutual_recursion() -> None:
    # mutually recursive functions terminate; the cycle stays opaque
    def ping(x: Any) -> Any:  # noqa: ARG001
        return pong

    def pong(x: Any) -> Any:  # noqa: ARG001
        return ping

    assert infer(ping) == "(x: object) -> (x: object) -> FunctionType"


def test_infer_empty_container() -> None:
    # empty containers parametrize with `Never`; the empty tuple is `tuple[()]`
    def returns(value: Any) -> Callable[[], Any]:
        return lambda: value

    assert infer(returns([])) == "() -> list[Never]"
    assert infer(returns({})) == "() -> dict[Never, Never]"
    assert infer(returns(())) == "() -> tuple[()]"
    assert infer(returns(set())) == "() -> set[Never]"
    assert infer(returns(frozenset())) == "() -> frozenset[Never]"
    assert infer(returns([[]])) == "() -> list[list[Never]]"


def test_infer_ellipsis() -> None:
    # the `...` value is `EllipsisType`, never its unusable `ellipsis` `__name__`
    assert infer(lambda: ...) == "() -> EllipsisType"
    assert infer(lambda x: (x, ...)) == "[T](x: T) -> tuple[T, EllipsisType]"
    assert infer(lambda: [..., ...]) == "() -> list[EllipsisType]"


def test_infer_types_aliases() -> None:
    # render by the importable `types` name, not the cpython-internal `__name__`
    assert infer(lambda: math) == "() -> ModuleType"
    assert infer(lambda: (lambda: 0).__code__) == "() -> CodeType"
    assert infer(lambda: currentframe()) == "() -> FrameType"
    assert infer(lambda: type.__dict__["__dict__"]) == "() -> GetSetDescriptorType"
    assert infer(lambda: MappingProxyType({"k": 1})) == (
        "() -> MappingProxyType[Literal['k'], Literal[1]]"
    )


class _MyGeneric[T]: ...  # module-level, so its name resolves


def test_infer_generic_alias() -> None:
    # a subscripted generic denotes the type it spells, not its `GenericAlias` runtime
    assert infer(lambda: list[int]) == "() -> type[list[int]]"
    assert infer(lambda: dict[str, int]) == "() -> type[dict[str, int]]"
    assert infer(lambda: tuple[int, ...]) == "() -> type[tuple[int, ...]]"
    assert infer(lambda: list[int | None]) == "() -> type[list[int | None]]"
    assert infer(lambda: list[dict[str, int]]) == "() -> type[list[dict[str, int]]]"
    assert infer(lambda x: (x, list[int])) == "[T](x: T) -> tuple[T, type[list[int]]]"
    # a user-defined generic (`typing._GenericAlias`) unwraps like a builtin one
    assert infer(lambda: _MyGeneric[int]) == "() -> type[_MyGeneric[int]]"
    assert infer(lambda: list[_MyGeneric[int]]) == "() -> type[list[_MyGeneric[int]]]"


def test_infer_generic_alias_union() -> None:
    # a union has no `type[...]` form; `type[int | str]` means `type[int] | type[str]`
    assert infer(lambda: int | str) == "() -> TypeForm[int | str]"
    assert infer(lambda: int | None) == "() -> TypeForm[int | None]"


def test_infer_generic_alias_callable() -> None:
    # a `Callable` has no `type[...]` form either; `TypeForm` over the arrow form
    assert infer(lambda: Callable[[int], str]) == "() -> TypeForm[(int) -> str]"
    assert infer(lambda: Callable[..., str]) == "() -> TypeForm[(...) -> str]"
    assert infer(lambda: list[Callable[[int], str]]) == (
        "() -> type[list[(int) -> str]]"
    )


def test_infer_generic_alias_unnameable() -> None:
    # an unnameable origin or argument keeps the unhelpful but honest `GenericAlias`
    def make_builtin() -> Callable[[], Any]:
        class Local: ...

        return lambda: list[Local]

    def make_callable() -> Callable[[], Any]:
        class Local: ...

        return lambda: Callable[[int], Local]

    def make_user() -> Callable[[], Any]:
        class Local: ...

        return lambda: _MyGeneric[Local]

    assert infer(make_builtin()) == "() -> GenericAlias"
    assert infer(make_callable()) == "() -> GenericAlias"
    assert infer(make_user()) == "() -> GenericAlias"


@pytest.mark.skipif(sys.version_info < (3, 15), reason="requires Python 3.15+")
def test_infer_frozendict() -> None:
    frozendict: Any = getattr(builtins, "frozendict", None)

    # a `frozendict` result parametrizes like `dict`, also through a typevar default
    assert infer(lambda x: frozendict({"k": x + 1})) == (
        "[R](x: CanAdd[Literal[1], R]) -> frozendict[Literal['k'], R]"
    )

    empty = frozendict()
    assert infer(lambda: empty) == "() -> frozendict[Never, Never]"

    def f(x: Any = 0) -> Any:
        return frozendict({"k": x})

    assert infer(f) == "[T = Literal[0]](x: T = 0) -> frozendict[Literal['k'], T]"


@pytest.mark.skipif(sys.version_info < (3, 15), reason="requires Python 3.15+")
def test_infer_sentinel() -> None:
    sentinel: Any = getattr(builtins, "sentinel", None)

    # a sentinel is its own (PEP 661) type, spelled as its declared name
    missing = sentinel("MISSING")

    assert infer(lambda x: x + missing) == "[R](x: CanAdd[MISSING, R]) -> R"

    def f(x: Any = missing) -> Any:
        return x

    assert infer(f) == "[T = MISSING](x: T = MISSING) -> T"

    def g(x: Any = missing) -> Any:
        return [] if x is missing else x

    assert infer(g) == "(x: MISSING = MISSING) -> list[Never]\n[T: ~MISSING](x: T) -> T"


def test_infer_ufunc() -> None:
    np = pytest.importorskip("numpy")
    assert infer(np.sin) == "[R](x: CanArrayUFunc[np.ufunc, R] | ToComplexND) -> R"
    assert infer(np.add) == (
        "[R](x1: CanArrayUFunc[np.ufunc, R] | ToComplexND, x2: ToComplexND) -> R"
    )
    assert infer(np.hypot) == (
        "[R](x1: CanArrayUFunc[np.ufunc, R] | ToFloatND, x2: ToFloatND) -> R"
    )
    assert infer(np.gcd) == (
        "[R](x1: CanArrayUFunc[np.ufunc, R] | ToIntND, x2: ToIntND) -> R"
    )
    # ldexp(mantissa: float, exponent: int) — different widest dtype per input
    assert infer(np.ldexp) == (
        "[R](x1: CanArrayUFunc[np.ufunc, R] | ToFloatND, x2: ToIntND) -> R"
    )


def test_infer_ufunc_in_function() -> None:
    np = pytest.importorskip("numpy")

    # a ufunc inside a traced function → override path only (spy's `__array_ufunc__`)
    def f(x: Any) -> Any:
        return np.sin(x)

    assert infer(f) == "[R](x: CanArrayUFunc[np.ufunc, R]) -> R"


def test_infer_array_function() -> None:
    np = pytest.importorskip("numpy")
    # NEP-18 functions (np.mean, np.strings.upper, ...) dispatch via __array_function__
    sig = "[R](a: CanArrayFunction[(Any) -> R, R]) -> R"
    assert infer(np.mean) == sig
    assert infer(np.sum) == sig
    # the func type's arity tracks the required positional params (a, b)
    assert infer(np.outer) == (
        "[R](a: CanArrayFunction[(Any, Any) -> R, R], b: object) -> R"
    )


def test_array_function_node() -> None:
    # a structured `App`, not a string: one `Any` per required positional parameter
    ret = Name("R")

    def f(a: object, b: object) -> object: ...

    node = array_function_node(f, ret)
    assert node == App("CanArrayFunction", (Fn((Name("Any"), Name("Any")), ret), ret))
    assert render(node) == "CanArrayFunction[(Any, Any) -> R, R]"
    # being structured, `names` reaches the inner typevar a bare string would hide
    assert list(names(node)) == ["Any", "Any", "R", "R"]

    # a variadic dispatched function has no fixed arity, rendering as `(...)`
    def g(*args: object) -> object: ...

    assert render(array_function_node(g, ret)) == "CanArrayFunction[(...) -> R, R]"


@pytest.mark.parametrize("selector", ["nope", 9, -9])
def test_unknown_param(selector: str | int) -> None:
    with pytest.raises(ValueError, match="parameter"):
        infer(abs, selector)


def test_variadic_exhausted() -> None:
    # placeholder growth is bounded; running out reports cleanly
    with pytest.raises(InferError, match="placeholder"):
        infer(lambda *args: args[10_000])


def test_fixed_unpack_beyond_default_unsupported() -> None:
    # an iterator yields two by default, so a 3+ target unpack can't be satisfied
    def f(x: Any) -> Any:
        a, b, c = x
        return a, b, c

    with pytest.raises(InferError):
        infer(f)


def test_mixed_star_unpack() -> None:
    # two star unpackings of different fixed arities can't share one budget
    def f(a: Any, b: Any) -> Any:
        return divmod(*a), _takes3(*b)  # type: ignore[misc]

    with pytest.raises(InferError):
        infer(f)


def test_star_unpack_no_budget_leak() -> None:
    # star-unpack's grown budget (here 3) must not leak to `sum(z)`; `z` alone keeps
    # the default yield of 2, so its `CanAdd` chain stays one level deep (#683, #686)
    assert infer(lambda x, z: (_takes3(*x), sum(z))) == (
        "[T: CanRAdd[Literal[0], CanAdd[T, R2]], R, R2]"
        "(x: CanIter[CanNext[R]], z: CanIter[CanNext[T]]) -> tuple[R, R2]\n"
        "[R, R2]"
        "(x: CanIter[CanNext[R]], z: CanIter[CanNext[CanRAdd[Literal[0] | R2, R2]]])"
        " -> tuple[R, R2]"
    )


def test_star_unpack_gap_arity() -> None:
    # a star-unpack into a 9-ary call hits a budget the old sparse range skipped (#683)
    assert infer(lambda x: _takes9(*x)) == "[R](x: CanIter[CanNext[R]]) -> R"


def test_args_and_star_unpack() -> None:
    # `*args` and a growable star-unpack coexist: each grows its own budget instead of
    # the `*args` retry starving the star-unpack of yields (#683)
    def f(*args: Any) -> Any:
        return divmod(*divmod(args[0], args[1]))  # type: ignore[misc]

    sig = infer(f)
    assert "CanDivmod" in sig
    assert "CanRDivmod" in sig


def test_star_unpack_error_not_masked() -> None:
    # a non-arity error from a star-unpack target must surface as-is, not buried under
    # a bogus "got N args" after needlessly climbing the whole yield budget (#683)
    def picky(_a: Any) -> Any:
        raise ValueError("domain error")

    with pytest.raises(InferError, match="domain error"):
        infer(lambda x: picky(*x))


def test_variadic_mixed() -> None:
    def f(x: Any, *args: Any, **kwargs: Any) -> Any:
        return (x, args, kwargs)

    assert infer(f) == (
        "[T, *Ts, U](x: T, *args: *Ts, **kwargs: U)"
        " -> tuple[T, tuple[*Ts], dict[str, U]]"
    )


def test_fork_explosion() -> None:
    # exploration is budgeted: a function that forks on every run raises (timely)
    # instead of exhaustively walking all 2**100 decision paths. Distinct `x[i]` spies
    # each fork independently, since one spy's `bool` is stable per run.
    def f(x: Any) -> Any:
        for i in range(100):
            bool(x[i])
        return x

    with pytest.raises(InferError, match="completion"):
        infer(f)


@pytest.mark.parametrize(
    "choose",
    [random.choice, secrets.choice],  # noqa: S311
    ids=["random", "secrets"],
)
def test_infer_choice_does_not_hang(choose: Callable[..., Any]) -> None:
    # issue #667: a consistent `len()` keeps random's empty-range loop unreachable
    with pytest.raises(InferError):
        infer(choose)


def _budget_exhausting(x: Any) -> Any:
    return [bool(x[i]) for i in range(10)]


def test_fork_truncation_warning() -> None:
    # when the budget runs out before every branch was explored, the warning categorizes
    # the gap and names the affected form. Distinct `x[i]` spies each fork on `bool`.
    with pytest.warns(InferWarning, match=r"run budget exhausted in \(x\)"):
        infer(_budget_exhausting)


def test_strict_raises_on_gap() -> None:
    # the same incompleteness that warns by default fails closed under `strict`
    with pytest.warns(InferWarning):
        assert isinstance(infer(_budget_exhausting), str)
    with pytest.raises(InferError, match="incomplete exploration"):
        infer(_budget_exhausting, strict=True)


def test_strict_silent_when_complete() -> None:
    # an exhaustively explored function neither warns nor raises under `strict`
    with warnings.catch_warnings():
        warnings.simplefilter("error", InferWarning)
        result = infer(lambda x: x + 1, strict=True)
    assert result == "[R](x: CanAdd[Literal[1], R]) -> R"


def test_gap_message_and_dedup() -> None:
    gap = _Gap(GapKind.RUN_BUDGET, "(x)")
    assert gap.message() == "run budget exhausted in (x)"
    # frozen and slotted, so equal gaps collapse in a set
    assert len({gap, _Gap(GapKind.RUN_BUDGET, "(x)")}) == 1


def test_dispatch_hasattr_collapses_to_object() -> None:
    # `hasattr` tolerates absence, so the attribute is not a requirement
    assert infer(lambda x: hasattr(x, "a")) == "(x: object) -> bool"


def test_dispatch_try_except_bool_collapses_to_object() -> None:
    # a `try`/`except AttributeError` that returns a bool is also a presence predicate
    def f(x: Any) -> Any:
        try:
            x.value  # noqa: B018
        except AttributeError:
            return False
        return True

    assert infer(f) == "(x: object) -> bool"


def test_dispatch_negated_predicate_collapses_to_object() -> None:
    # the inverted polarity (`False` present, `True` absent) is still a bool predicate
    assert infer(lambda x: not hasattr(x, "a")) == "(x: object) -> bool"


def test_dispatch_presence_independent_return_collapses() -> None:
    # the return ignores the attribute's value, so one overload with the unioned return
    # of both branches covers it, rather than an `object` fallback
    assert infer(lambda x: 1 if hasattr(x, "a") else 2) == "(x: object) -> int"


def test_dispatch_getattr_default_widens_fallback() -> None:
    # a value dispatch keeps two overloads, but the fallback's return widens to a sound
    # supertype of the present `R` (`object`), so the overloads no longer overlap
    def f(x: Any) -> Any:
        return getattr(x, "value", None)

    assert infer(f) == "[R](x: Has['value', +R]) -> R\n(x: object) -> object"


def test_dispatch_try_except_value_widens_fallback() -> None:
    def f(x: Any) -> Any:
        try:
            return x.value
        except AttributeError:
            return 0

    assert infer(f) == "[R](x: Has['value', +R]) -> R\n(x: object) -> object"


def test_dispatch_required_attribute_unchanged() -> None:
    # a directly-used attribute has no fallback, so its absent variant raises and is
    # dropped; the attribute stays a hard requirement
    assert infer(lambda x: x.a) == "[R](x: Has['a', +R]) -> R"


def test_dispatch_conditional_use_no_overload() -> None:
    # `y.foo` is used in one branch of an unrelated fork, not dispatched on; forcing it
    # absent completes via the other branch but leaves no tolerated-absence marker
    def f(x: Any, y: Any) -> Any:
        return x if 0 in x else y.foo()

    assert infer(f) == (
        "[T: CanContains[Literal[0]], R](x: T, y: Has['foo', () -> +R]) -> T | R"
    )


def test_dispatch_independent_dispatches_keep_baseline() -> None:
    # two presence-tests on one parameter don't compose soundly, so the strict baseline
    # (requiring both) is kept rather than overloads that resolve wrongly for mixed x
    def f(x: Any) -> Any:
        return hasattr(x, "a"), hasattr(x, "b")

    assert infer(f) == (
        "(x: Has['a'] & Has['b']) -> tuple[Literal[True], Literal[True]]"
    )


def test_dispatch_multi_parameter_keeps_baseline() -> None:
    # a dispatch on a multi-parameter function isn't collapsed or widened (the fallback
    # can't be rendered soundly without losing the other parameters), so the baseline
    # stays; `y`'s own `a` is unaffected by forcing `x`'s `a` absent
    def f(x: Any, y: Any) -> Any:
        return hasattr(x, "a"), y.a

    assert infer(f) == ("[R](x: Has['a'], y: Has['a', +R]) -> tuple[Literal[True], R]")


def test_dispatch_value_from_parameter_keeps_baseline() -> None:
    # the absent branch derives its value from the parameter (`x.b`), so a widened
    # fallback would orphan that typevar; the strict baseline is kept instead
    def f(x: Any) -> Any:
        return hasattr(x, "a") or x.b

    assert infer(f) == "(x: Has['a']) -> bool"


def test_dispatch_fallback_keeps_unconditional_requirement() -> None:
    # the `value` absence marker must not suppress the absent branch's `fallback` read
    def f(x: Any) -> Any:
        try:
            return x.value
        except AttributeError:
            x.fallback  # noqa: B018
            return 0

    assert infer(f) == "[R](x: Has['value', +R]) -> R\n(x: Has['fallback']) -> object"


def test_dispatch_value_capability_keeps_baseline() -> None:
    # the present branch constrains the value (`+ 1`), so an `object` fallback would
    # admit a `value` that cannot add; the sound baseline is kept
    def f(x: Any) -> Any:
        return getattr(x, "value", 0) + 1

    assert infer(f) == "[R](x: Has['value', +CanAdd[Literal[1], R]]) -> R"


def test_dispatch_present_extra_requirement_keeps_baseline() -> None:
    # the present branch also requires `b`, so widening to `object` would be unsound
    def f(x: Any) -> Any:
        if hasattr(x, "a"):
            _ = x.b
            return True
        return False

    assert infer(f) == "(x: Has['a'] & Has['b']) -> bool"


def test_dispatch_no_budget_warning_for_plain_reads() -> None:
    # reading many attributes is not a dispatch, so it must not trip the dispatch budget
    def f(x: Any) -> Any:
        return x.a, x.b, x.c, x.d, x.e, x.f, x.g, x.h, x.i

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(infer(f), str)
    assert isinstance(infer(f, strict=True), str)


def test_dispatch_multi_parameter_no_budget_warning() -> None:
    # a multi-parameter function never dispatches, so attribute reads cannot truncate
    def f(x: Any, y: Any) -> Any:
        return x.a, x.b, x.c, x.d, x.e, x.f, x.g, x.h, x.i, y

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(infer(f), str)


def test_target_exception_skipped() -> None:
    # a non-protocol error from the target marks a failed run; it never escapes
    def f(x: Any) -> None:  # noqa: ARG001
        raise AssertionError

    with pytest.raises(InferError, match="completion") as excinfo:
        infer(f)
    assert isinstance(excinfo.value.__cause__, AssertionError)


def test_target_exception_cause() -> None:
    # when no run completes, the target's last exception is chained as the cause
    with pytest.raises(InferError, match="completion") as excinfo:
        infer(lambda: 0 / 0)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_target_exception_partial() -> None:
    # only the branch that raises is dropped; the surviving branch still infers
    def f(x: Any) -> int:
        if not x:
            raise ZeroDivisionError
        return 1

    assert infer(f) == "(x: CanBool) -> int"


def test_large_tuple_widens() -> None:
    # a long concrete tuple (like `random.getstate`) widens instead of listing literals
    def f() -> tuple[int, ...]:
        return tuple(range(50))

    assert infer(f) == "() -> tuple[int, ...]"


def test_union_tuple_collapse() -> None:
    # a wide union of same-arity tuples (like `colorsys.hls_to_rgb`) collapses per
    # position; a small one keeps its correlation
    def pair(i: int) -> App:
        return App("tuple", (Name(f"A{i}"), Name(f"B{i}")))

    def triple(i: int) -> App:
        return App("tuple", (Name(f"C{i}"), Name(f"D{i}"), Name(f"E{i}")))

    def rendered(nodes: list[Node], *, tuples: bool) -> str:
        node = union(nodes, tuples=tuples)
        assert node is not None
        return render(node)

    cols2 = " | ".join(f"A{i}" for i in range(9)), " | ".join(f"B{i}" for i in range(9))
    wide2 = f"tuple[{cols2[0]}, {cols2[1]}]"

    small: list[Node] = [pair(0), pair(1)]
    assert rendered(small, tuples=True) == "tuple[A0, B0] | tuple[A1, B1]"

    wide: list[Node] = [pair(i) for i in range(9)]
    assert rendered(wide, tuples=True) == wide2
    assert rendered(wide, tuples=False).count("tuple[") == 9

    # each arity collapses on its own; a wider triple group folds independently
    cols3 = tuple(" | ".join(f"{p}{i}" for i in range(9)) for p in "CDE")
    mixed = wide + [triple(i) for i in range(9)]
    assert rendered(mixed, tuples=True) == f"{wide2} | tuple[{', '.join(cols3)}]"

    # a non-tuple member and a variadic tuple stay untouched beside the collapse
    assert rendered([*wide, Name("X")], tuples=True) == f"{wide2} | X"
    variadic = App("tuple", (Name("V"), Dots()))
    assert rendered([*wide, variadic], tuples=True) == f"{wide2} | tuple[V, ...]"


def test_self_referential_result() -> None:
    # a cyclic result is a recursive type, tied off with a self-bounded typevar
    def f() -> list[object]:
        a: list[object] = []
        a.append(a)
        return a

    assert infer(f) == "[R: list[R]]() -> R"


def test_self_referential_result_nested() -> None:
    # the cycle is detected by identity through any depth of intermediate containers
    def f() -> list[object]:
        a: list[object] = []
        b: list[object] = [a]
        a.append(b)
        return a

    assert infer(f) == "[R: list[list[R]]]() -> R"


def test_self_referential_result_mixed() -> None:
    # a recursive container alongside a parameter and a result typevar
    def f(x: Any) -> list[object]:
        a: list[object] = []
        a.extend((a, x + 1))
        return a

    assert infer(f) == "[R, R2: list[R2 | R]](x: CanAdd[Literal[1], R]) -> R2"


def test_self_referential_result_defaulted() -> None:
    # PEP 696: the defaultless recursive typevar must precede the defaulted one
    def f(x: Any = 1) -> list[object]:
        a: list[object] = []
        a.extend((a, x))
        return a

    assert infer(f) == "[R: list[R | T], T = Literal[1]](x: T = 1) -> R"


def test_deeply_nested_result() -> None:
    # a finite result too deep for the call stack is reported, not crashed on
    def f() -> list[object]:
        x: list[object] = []
        for _ in range(sys.getrecursionlimit() * 2):
            x = [x]
        return x

    with pytest.raises(InferError, match="deeply"):
        infer(f)


def test_structural_dedup() -> None:
    # every forked run re-derives `a = x * 2` and then `a + 1` / `a + 2`; the fresh
    # placeholders for these repeated subexpressions collapse onto one type parameter
    # each, not one per run (which is what made e.g. `colorsys.hls_to_rgb` explode)
    def f(x: Any) -> object:
        a = x * 2
        return (a + 1) if x else (a + 2) if a else a

    assert infer(f) == (
        "[R, R2: CanBool](x: CanMul[Literal[2], R2 & CanAdd[Literal[1, 2], R]"
        " & CanBool] & CanBool) -> R | R2"
    )


def test_dedup_different_operands() -> None:
    # `x[y]` and `x[z]` share an op-shape, so they collapse onto one return typevar
    def f(x: Any, y: Any, z: Any) -> object:
        return x[y] if x else x[z]

    assert infer(f) == "[T, U, R](x: CanBool & CanGetitem[T | U, R], y: T, z: U) -> R"

    # merged results may be traced (`.foo`), which the old untraced-only guard blocked
    def g(x: Any, y: Any, z: Any) -> object:
        p = x[y] if x else x[z]
        return p.foo

    assert infer(g) == (
        "[T, U, R](x: CanBool & CanGetitem[T | U, Has['foo', +R]], y: T, z: U) -> R"
    )


def test_not_callable() -> None:
    not_callable: Any = 42
    with pytest.raises(InferError, match="not a callable"):
        infer(not_callable)


def test_iter() -> None:
    # the callable_iterator enrichment, independent of `iter`'s own docstring
    assert infer(lambda f, s: iter(f, s)) == "[R](f: () -> R, s: object) -> Iterator[R]"
    # the callable is the first argument, not the sentinel: a non-spy callable can't
    # be probed, so the element type is left opaque rather than blamed on the sentinel
    assert infer(lambda s: iter(int, s)) == "(s: object) -> callable_iterator"


def test_builtin_without_signature() -> None:
    try:
        signature(iter)
    except ValueError:
        pass
    else:
        pytest.skip("this build exposes an `inspect.signature` for `iter`")
    # the arity probe recovers a signatureless builtin instead of raising, exploring
    # each accepted arity as a separate overload
    assert infer(iter) == "[R](CanIter[R]) -> R\n[R](() -> R, object) -> Iterator[R]"


def test_builtin_type() -> None:
    # the metaclass `type` has no `inspect.signature`; the probe recovers its 1-argument
    # form (the 3-argument metaclass call needs typed placeholders, out of scope)
    assert infer(type) == "[T](T) -> type[T]"


def test_builtin_range() -> None:
    assert infer(range) == (
        "(CanIndex) -> range\n"
        "(CanIndex, CanIndex) -> range\n"
        "(CanIndex, CanIndex, CanIndex) -> range"
    )


def test_builtin_slice() -> None:
    assert infer(slice) == (
        "[T](T) -> slice[None, T, None]\n"
        "[T, U](T, U) -> slice[T, U, None]\n"
        "[T, U, V](T, U, V) -> slice[T, U, V]"
    )


def test_builtin_variadic() -> None:
    # an arity accepted all the way to the probe cap is rendered as `*args`
    assert infer(min) == "[T, U: CanLt[T | U, CanBool]](T, *args: U) -> U | T"


def test_functools_reduce() -> None:
    # the iterable must yield a pair so `reduce` actually calls the function (#723)
    if sys.version_info >= (3, 15):
        # `reduce` gained a signature with an `initial` sentinel default
        assert infer(functools.reduce) == (
            "[T, R]((T, T) -> R, CanIter[CanNext[T]],"
            " initial: _initial_missing = _initial_missing) -> R\n"
            "[T: ~_initial_missing, U, R]"
            "((T | R, U) -> R, CanIter[CanNext[U]], initial: T) -> R"
        )
    else:
        assert infer(functools.reduce) == (
            "[T, R]((T, T) -> R, CanIter[CanNext[T]]) -> R\n"
            "[T, U, R]((T | R, U) -> R, CanIter[CanNext[U]], T) -> R"
        )


def test_builtin_sorted() -> None:
    # the iterable yields a pair, so the elements reach `__lt__` and the `key` result
    # carries its own comparison constraint (#723)
    assert infer(sorted) == (
        "[R: CanLt[R, CanBool]]"
        "(CanIter[CanNext[R]], key: None = None, reverse: Literal[False] = False)"
        " -> list[R]\n"
        "[R: CanLt[R, CanBool]]"
        "(CanIter[CanNext[R]], key: None = None, reverse: CanBool) -> list[R]\n"
        "[T, R]"
        "(CanIter[CanNext[R]], key: (R) -> T & CanLt[T, CanBool],"
        " reverse: Literal[False] = False) -> list[R]\n"
        "[T, R]"
        "(CanIter[CanNext[R]], key: (R) -> T & CanLt[T, CanBool], reverse: CanBool)"
        " -> list[R]"
    )


def test_builtin_typed_argument() -> None:
    # `getattr` needs a real `str` name that an object spy cannot supply, so no arity
    # explores and it raises rather than inferring a bogus signature
    with pytest.raises(InferError):
        infer(getattr)


def test_dynamic_attr_name() -> None:
    # a spy-derived attribute name is not statically known
    with pytest.raises(InferError, match="no protocol"):
        infer(lambda x, y: getattr(x, str(y)))


def test_instance_subclass_check() -> None:
    # isinstance/issubclass recurse into a tuple classinfo, so the param widens (#716)
    assert infer(lambda x: isinstance(0, x)) == (
        "(x: CanInstancecheck | tuple[CanInstancecheck, ...]) -> bool"
    )
    assert infer(lambda x: issubclass(int, x)) == (
        "(x: CanSubclasscheck | tuple[CanSubclasscheck, ...]) -> bool"
    )
    assert infer(lambda x, y: isinstance(y, x)) == (
        "(x: CanInstancecheck | tuple[CanInstancecheck, ...], y: object) -> bool"
    )
    assert infer(lambda x, y: issubclass(y, x)) == (
        "(x: CanSubclasscheck | tuple[CanSubclasscheck, ...], y: object) -> bool"
    )


def test_instance_subclass_check_builtin() -> None:
    assert infer(isinstance) == (
        "(object, CanInstancecheck | tuple[CanInstancecheck, ...]) -> bool"
    )
    assert infer(issubclass) == (
        "(object, CanSubclasscheck | tuple[CanSubclasscheck, ...]) -> bool"
    )


def test_instance_check_no_overwiden() -> None:
    # a returned classinfo is a typevar (no widening); `in`/`len` do not distribute
    assert (
        infer(
            lambda x, y: y.foo() if (isinstance(y, x) and not isinstance(y, x)) else x,
        )
        == "[T: CanInstancecheck](x: T, y: object) -> T"
    )
    assert infer(lambda x: 0 in x) == "(x: CanContains[Literal[0]]) -> bool"
    assert infer(lambda x: len(x)) == "(x: CanLen) -> int"


def _set_name(x: Any) -> object:
    class C:
        attr: Any = x

    return C


def test_set_name() -> None:
    assert infer(_set_name) == "(x: CanSetName[type, Literal['attr']]) -> type"


def test_weakref_proxy() -> None:
    # a `weakref.proxy` forwards `__class__`, so it must not be mistaken for a spy
    assert infer(weakref.proxy) == "(object) -> CallableProxyType"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_module() -> None:
    out = _run_cli("-m", "optype.infer", "lambda x: x + 1")
    assert out.returncode == 0
    assert out.stdout.strip() == "[R](x: CanAdd[Literal[1], R]) -> R"


def test_cli_subcommand() -> None:
    out = _run_cli("-m", "optype", "infer", "lambda x: x * 2")
    assert out.returncode == 0
    assert out.stdout.strip() == "[R](x: CanMul[Literal[2], R]) -> R"


def test_cli_def() -> None:
    # a trailing def/class is inferred directly, without a closing name reference
    out = _run_cli("-m", "optype", "infer", "def f(x, y): return x @ y")
    assert out.returncode == 0
    assert out.stdout.strip() == (
        "[T, R](x: CanMatmul[T, R], y: T) -> R\n[T, R](x: T, y: CanRMatmul[T, R]) -> R"
    )


def test_cli_default() -> None:
    out = _run_cli("-m", "optype", "infer", "def f(x=0): return x")
    assert out.returncode == 0
    assert out.stdout.strip() == "[T = Literal[0]](x: T = 0) -> T"


def test_cli_variadic() -> None:
    out = _run_cli("-m", "optype", "infer", "lambda *args: args")
    assert out.returncode == 0
    assert out.stdout.strip() == "[*Ts](*args: *Ts) -> tuple[*Ts]"


def test_cli_returned_function() -> None:
    out = _run_cli("-m", "optype", "infer", "lambda x: lambda y: (x, y)")
    assert out.returncode == 0
    assert out.stdout.strip() == "[T, U](x: T) -> (y: U) -> tuple[T, U]"


def test_cli_builtin() -> None:
    # a signatureless builtin is recovered by the arity probe instead of erroring out
    out = _run_cli("-m", "optype", "infer", "type")
    assert out.returncode == 0
    assert out.stdout.strip() == "[T](T) -> type[T]"


def test_cli_usage() -> None:
    out = _run_cli("-m", "optype")
    assert out.returncode == 1
    assert "usage" in out.stderr.lower()


def test_cli_infer_error() -> None:
    # infer's own limitations exit cleanly, instead of with a traceback
    out = _run_cli("-m", "optype", "infer", "lambda x, y: getattr(x, str(y))")
    assert out.returncode == 1
    assert out.stderr.startswith("InferError: no protocol")


def test_cli_warns_on_stderr() -> None:
    # an incomplete exploration keeps the signature on stdout and the warning on stderr
    out = _run_cli("-m", "optype", "infer", "lambda x: [bool(x[i]) for i in range(10)]")
    assert out.returncode == 0
    assert out.stdout.startswith("(x: CanGetitem")
    assert "warning:" in out.stderr
    assert "incomplete exploration" in out.stderr
