# ruff: noqa: FURB118, PLW0108
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnusedParameter=false

import builtins
import functools
import math
import operator
import subprocess  # noqa: S404
import sys
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from optype.infer import InferError, InferWarning, infer
from optype.infer._explore import _doc_params
from optype.infer._ir import App, Arg, Fn, Lit, Name, Type, subtype


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
    (
        lambda x: sum(x),
        "[R](x: CanIter[CanNext[CanRAdd[Literal[0], R]]]) -> R",
    ),
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
    # `sorted` discards the `key` results, so they are unconstrained; the explicit
    # `object` keeps the last argument out of the return slot
    (
        lambda xs, key: sorted(xs, key=key),
        "[R](xs: CanIter[CanNext[R]], key: (R) -> object) -> list[R]",
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


def _spread(*args: Any) -> Any:
    return _takes3(*args)


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
    # a splat splices the TypeVarTuple into the surrounding tuple
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
    # a function within a returned container is explored as well, but a set member
    # must stay hashable, so it is left as-is
    (
        lambda x: (lambda y: y, 1),  # noqa: ARG005
        "[T](x: object) -> tuple[(y: T) -> T, Literal[1]]",
    ),
    (lambda x: [lambda: x], "[T](x: T) -> list[() -> T]"),
    (lambda x: {"get": lambda: x}, "[T](x: T) -> dict[Literal['get'], () -> T]"),
    (lambda x: {lambda: x}, "(x: object) -> set[function]"),
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
    # as do variadic parameters and a `partial` of a non-function
    (_self_return, "(x: object) -> function"),
    (_fn_factory, "(n: object) -> () -> function"),
    (lambda x: lambda *args: x, "(x: object) -> function"),  # noqa: ARG005
    (lambda f: functools.partial(f, 1), "(f: object) -> partial"),
    (lambda: functools.partial(print, "a"), "() -> partial"),
]

ITERATOR_CASES: list[tuple[Callable[..., Any], str]] = [
    # lazy builtin iterators are iterated like generators
    (lambda x: map(str, x), "(x: CanIter[CanNext[CanStr]]) -> map[str]"),
    (
        lambda f, x: map(f, x),
        "[T, R](f: (T) -> R, x: CanIter[CanNext[T]]) -> map[R]",
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
    (
        lambda x: ((i for i in x), 1),
        "[R](x: CanIter[CanNext[R]]) -> tuple[Generator[R], Literal[1]]",
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
    assert infer(memoryview.tobytes) == (
        "(memoryview, order: Literal['C'] = 'C') -> bytes"
    )


def test_method_descriptor_fixed_defaults() -> None:
    # a defaulted parameter whose spy the function rejects passes its default
    # instead, while the accepting parameters stay structural
    assert infer(str.split) == (
        "(str, sep: None = None, maxsplit: CanIndex = -1) -> list[Never]"
    )
    assert infer(bytes.decode) == (
        "(bytes, encoding: Literal['utf-8'] = 'utf-8', "
        "errors: Literal['strict'] = 'strict') -> str"
    )


def test_method_descriptor_unsupported() -> None:
    # generators cannot be constructed, spy-rejecting parameters without a default
    # have nothing to fall back on, and an empty `self` cannot always run
    send = type(x for x in range(0)).send
    with pytest.raises(InferError, match="cannot instantiate 'generator'"):
        infer(send)
    with pytest.raises(InferError, match="expected str instance"):
        infer(str.join)
    with pytest.raises(InferError, match="pop from empty list"):
        infer(list[Any].pop)


def test_ternary_pow() -> None:
    # the optional modulo is used forward but dropped from the reflected overload
    def f(x: Any, y: Any, z: Any = None) -> Any:
        return x.__pow__(y, z)  # noqa: PLC2801

    assert infer(f) == (
        "[T, R, U = None](x: CanPow[T, U, R], y: T, z: U = None) -> R\n"
        "[T, R](x: T, y: CanRPow[T, R]) -> R"
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

    assert infer(ping) == "(x: object) -> (x: object) -> function"


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


@pytest.mark.parametrize("selector", ["nope", 9, -9])
def test_unknown_param(selector: str | int) -> None:
    with pytest.raises(ValueError, match="parameter"):
        infer(abs, selector)


def test_variadic_exhausted() -> None:
    # placeholder growth is bounded; running out reports cleanly
    with pytest.raises(InferError, match="placeholder"):
        infer(lambda *args: args[10_000])


def test_variadic_mixed() -> None:
    def f(x: Any, *args: Any, **kwargs: Any) -> Any:
        return (x, args, kwargs)

    assert infer(f) == (
        "[T, *Ts, U](x: T, *args: *Ts, **kwargs: U)"
        " -> tuple[T, tuple[*Ts], dict[str, U]]"
    )


def test_fork_explosion() -> None:
    # exploration is budgeted: a function that forks on every run raises (timely)
    # instead of exhaustively walking all 2**100 decision paths
    def f(x: Any) -> Any:
        for _ in range(100):
            bool(x)
        return x

    with pytest.raises(InferError, match="completion"):
        infer(f)


def test_fork_truncation_warning() -> None:
    # when the budget runs out before every branch was explored, it warns
    def f(x: Any) -> Any:
        return [bool(x) for _ in range(10)]

    with pytest.warns(InferWarning, match="branch"):
        infer(f)


def test_not_callable() -> None:
    not_callable: Any = 42
    with pytest.raises(InferError, match="not a callable"):
        infer(not_callable)


def test_doc_params() -> None:
    def f() -> None: ...

    f.__doc__ = "f(a, b=1, [c]) -> z"
    assert _doc_params(f) == ["a", "b", "c"]

    f.__doc__ = "no signature line here"
    assert _doc_params(f) is None

    # a usage example, not a signature: its keyword arg is no parameter name
    f.__doc__ = "Apply f. For example, f(lambda x, y: x + y, [1, 2, 3])."
    assert _doc_params(f) is None


def test_doc_signature() -> None:
    # builtins like `int` have no signature; fall back to the docstring
    assert infer(int) == "(x: CanInt | CanIndex) -> int"


def test_type() -> None:
    assert infer(type) == "[T](object: T) -> type[T]"


def test_dynamic_attr_name() -> None:
    # a spy-derived attribute name is not statically known
    with pytest.raises(InferError, match="no protocol"):
        infer(lambda x, y: getattr(x, str(y)))


def test_instance_subclass_check() -> None:
    assert infer(lambda x: isinstance(0, x)) == "(x: CanInstancecheck) -> bool"
    assert infer(lambda x: issubclass(int, x)) == "(x: CanSubclasscheck) -> bool"
    assert infer(lambda x, y: isinstance(y, x)) == (
        "(x: CanInstancecheck, y: object) -> bool"
    )
    assert infer(lambda x, y: issubclass(y, x)) == (
        "(x: CanSubclasscheck, y: object) -> bool"
    )


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


def test_cli_usage() -> None:
    out = _run_cli("-m", "optype")
    assert out.returncode == 1
    assert "usage" in out.stderr.lower()


def test_cli_infer_error() -> None:
    # infer's own limitations exit cleanly, instead of with a traceback
    out = _run_cli("-m", "optype", "infer", "lambda x, y: getattr(x, str(y))")
    assert out.returncode == 1
    assert out.stderr.startswith("InferError: no protocol")
