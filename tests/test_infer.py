# ruff: noqa: FURB118

import math
import operator
import subprocess  # noqa: S404
import sys
from collections.abc import Callable
from typing import Any

import pytest

from optype.infer import InferError, InferWarning, _doc_params, infer
from optype.infer._ir import App, Lit, Name, Type, subtype

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
    (lambda x: abs(x), "[R](x: CanAbs[R]) -> R"),  # noqa: PLW0108
    (len, "(obj: CanLen) -> int"),
    (list, "[R](iterable: CanIter[CanNext[R]] & CanLen) -> list[R]"),
    (math.sqrt, "(x: CanFloat | CanIndex) -> float"),
    (lambda x: int(x), "(x: CanInt | CanIndex) -> int"),  # noqa: PLW0108
    (lambda x: complex(x), "(x: CanComplex | CanFloat | CanIndex) -> complex"),  # noqa: PLW0108
    (operator.index, "(a: CanIndex) -> int"),
    (lambda x: x(), "[R](x: CanCall[R]) -> R"),
    (lambda x: x(1, 2), "[R](x: CanCall[Literal[1], Literal[2], R]) -> R"),
    (lambda x: x(a=1), "[R](x: CanCall[a=Literal[1], R]) -> R"),
    (lambda x: x(1, b=2), "[R](x: CanCall[Literal[1], b=Literal[2], R]) -> R"),
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
    (
        lambda x: x[0] if len(x) else None,
        "[R](x: CanLen & CanGetitem[Literal[0], R]) -> R | None",
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
        "[R](x: CanLen & CanGetitem[Literal[0], R]) -> R | str",
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
            "[R, R2](x: CanLen & CanGetitem[Literal[0], R] & "
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
        lambda x: sum(x),  # noqa: PLW0108
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
    (lambda x: x.__name__, "[R](x: HasName[R]) -> R"),
    (lambda x: x.__qualname__, "[R](x: HasQualname[R]) -> R"),
    (lambda x: x.__match_args__, "[R](x: HasMatchArgs[R]) -> R"),
    (lambda x: x.__type_params__, "[R](x: HasTypeParams[R]) -> R"),
    (lambda x: x.__self__, "[R](x: HasSelf[R]) -> R"),
]

BINARY_CASES: list[tuple[Callable[[Any, Any], Any], str]] = [
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
    (lambda x, y: x, "[T](x: T, y: object) -> T"),  # noqa: ARG005
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
        "[T, R](x: CanLen & CanGetitem[Literal[0], R], y: T) -> R | T",
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
    (_call_packed, "[*Ts, R](x: CanCall[tuple[*Ts], R], *args: *Ts) -> R"),
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


INFER_CASES = [*UNARY_CASES, *BINARY_CASES, *VARIADIC_CASES]


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


def test_ternary_pow() -> None:
    # the optional modulo is used forward but dropped from the reflected overload
    def f(x: Any, y: Any, z: Any = None) -> Any:
        return x.__pow__(y, z)  # noqa: PLC2801

    assert infer(f) == (
        "[T, U, R](x: CanPow[T, U, R], y: T, z: U) -> R\n"
        "[T, R](x: T, y: CanRPow[T, R]) -> R"
    )


def test_infer_async() -> None:
    # coroutine functions are driven to completion; await/async with/async for trace
    async def aw(x: Any) -> Any:
        return (await x) + 1

    assert infer(aw) == "[R](x: CanAwait[CanAdd[Literal[1], R]]) -> R"

    async def async_with(x: Any) -> Any:
        async with x as y:
            return y

    assert infer(async_with) == (
        "[R](x: CanAEnter[CanAwait[R]] & CanAExit[None, None, None, CanAwait]) -> R"
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
    sig = "[R](a: CanArrayFunction[CanCall[Any, R], R]) -> R"
    assert infer(np.mean) == sig
    assert infer(np.sum) == sig
    # the func type's arity tracks the required positional params (a, b)
    assert infer(np.outer) == (
        "[R](a: CanArrayFunction[CanCall[Any, Any, R], R], b: object) -> R"
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


def test_doc_signature() -> None:
    # builtins like `int` have no signature; fall back to the docstring
    assert infer(int) == "(x: CanInt | CanIndex) -> int"


def _set_attr(x: Any) -> object:
    x.spam = 1
    return x


def _set_name(x: Any) -> object:
    class C:
        attr: Any = x

    return C


def test_set_name() -> None:
    assert infer(_set_name) == "(x: CanSetName[type, Literal['attr']]) -> type"


NO_PROTOCOL_CASES: list[Callable[[Any], Any]] = [
    lambda x: x.foo,
    _set_attr,
]


@pytest.mark.parametrize("func", NO_PROTOCOL_CASES, ids=["getattr", "setattr"])
def test_no_protocol(func: Callable[[Any], Any]) -> None:
    with pytest.raises(InferError, match="no protocol"):
        infer(func)


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


def test_cli_variadic() -> None:
    out = _run_cli("-m", "optype", "infer", "lambda *args: args")
    assert out.returncode == 0
    assert out.stdout.strip() == "[*Ts](*args: *Ts) -> tuple[*Ts]"


def test_cli_usage() -> None:
    out = _run_cli("-m", "optype")
    assert out.returncode == 1
    assert "usage" in out.stderr.lower()


def test_cli_infer_error() -> None:
    # infer's own limitations exit cleanly, instead of with a traceback
    out = _run_cli("-m", "optype", "infer", "lambda x: x.foo")
    assert out.returncode == 1
    assert out.stderr.startswith("InferError: no protocol")
