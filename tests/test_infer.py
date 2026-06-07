# ruff: noqa: FURB118, E501
import math
import operator
from collections.abc import Callable
from typing import Any

import pytest

from optype.infer import _doc_params, _infer, infer

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
    (abs, "[R](x: CanAbs[R]) -> R"),
    (len, "(obj: CanLen) -> int"),
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
        "[T, R](x: CanPos[CanMul[T, R]] & CanNeg[T]) -> R\n[T, R](x: CanPos[T] & CanNeg[CanRMul[T, R]]) -> R",
    ),
    (
        lambda x: -(x + x),
        "[T: CanAdd[T, CanNeg[R]], R](x: T) -> R\n[T: CanRAdd[T, CanNeg[R]], R](x: T) -> R",
    ),
    (lambda x: (x + 1) * 2, "[R](x: CanAdd[Literal[1], CanMul[Literal[2], R]]) -> R"),
    (
        lambda x: x[0] + x[1],
        "[T, R](x: CanGetitem[Literal[0, 1], T & CanAdd[T, R]]) -> R\n[T, R](x: CanGetitem[Literal[0, 1], T & CanRAdd[T, R]]) -> R",
    ),
    (lambda x: (x + 1, x + 1), "(x: CanAdd[Literal[1]]) -> tuple"),
    (lambda x: (x + 1, x + 2), "(x: CanAdd[Literal[1, 2]]) -> tuple"),
    (lambda x: (x + 1, x + 2, x + 3), "(x: CanAdd[Literal[1, 2, 3]]) -> tuple"),
    (lambda x: (x + 1, x + 1.0), "(x: CanAdd[float]) -> tuple"),
    (lambda x: (x + 1, x + 1j), "(x: CanAdd[complex]) -> tuple"),
    (lambda x: (x + 1.0, x + 1j), "(x: CanAdd[complex]) -> tuple"),
    (lambda x: (x + 1, x + 1.0, x + "a"), "(x: CanAdd[Literal['a'] | float]) -> tuple"),
    (lambda x: (x[1.0], x[None]), "(x: CanGetitem[float | None]) -> tuple"),
    (lambda x: (-x, -x), "(x: CanNeg) -> tuple"),
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
        "[T: CanMul[T, U], U, R](x: CanAdd[U, R], y: T) -> R\n[T, U: CanRMul[U, CanRAdd[T, R]], R](x: T, y: U) -> R",
    ),
    (lambda x, y: x[y], "[T, R](x: CanGetitem[T, R], y: T) -> R"),
    (lambda x, y: x, "[T](x: T, y: Unused) -> T"),  # noqa: ARG005
    (
        lambda x, y: x * 2 + y,
        "[T, R](x: CanMul[Literal[2], CanAdd[T, R]], y: T) -> R\n[T, R](x: CanMul[Literal[2], T], y: CanRAdd[T, R]) -> R",
    ),
    (
        lambda x, y: (x + y, y * 2),
        "[T: CanMul[Literal[2]]](x: CanAdd[T], y: T) -> tuple\n[T](x: T, y: CanMul[Literal[2]] & CanRAdd[T]) -> tuple",
    ),
    (
        lambda x, y: (x + y, y + x),
        "[T: CanAdd[U], U: CanAdd[T]](x: T, y: U) -> tuple\n[T: CanRAdd[U], U: CanRAdd[T]](x: T, y: U) -> tuple",
    ),
]


INFER_CASES = [*UNARY_CASES, *BINARY_CASES]


@pytest.mark.parametrize(
    ("func", "expected"),
    INFER_CASES,
    ids=[f"{i}:{e.splitlines()[0]}" for i, (_, e) in enumerate(INFER_CASES)],
)
def test_infer(func: Callable[..., Any], expected: str) -> None:
    assert _infer(func) == expected


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
    assert _infer(func, *params) == expected


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
    assert _infer(func) == expected


def test_keyword_only() -> None:
    def f(x: Any, *, y: Any) -> Any:
        return x[y]

    assert _infer(f) == "[T, R](x: CanGetitem[T, R], y: T) -> R"


def test_callable_instance() -> None:
    class Add1:
        def __call__(self, x: Any) -> Any:
            return x + 1

    assert _infer(Add1()) == "[R](x: CanAdd[Literal[1], R]) -> R"


def test_ternary_pow() -> None:
    # the optional modulo is used forward but dropped from the reflected overload
    def f(x: Any, y: Any, z: Any = None) -> Any:
        return x.__pow__(y, z)  # noqa: PLC2801

    assert (
        _infer(f)
        == "[T, U, R](x: CanPow[T, U, R], y: T, z: U) -> R\n[T, R](x: T, y: CanRPow[T, R]) -> R"
    )


@pytest.mark.parametrize("selector", ["nope", 9, -9])
def test_unknown_param(selector: str | int) -> None:
    with pytest.raises(ValueError, match="parameter"):
        _infer(abs, selector)


def _var_args(*args: Any) -> Any:
    return args


def _var_kwargs(**kwargs: Any) -> Any:
    return kwargs


@pytest.mark.parametrize("func", [_var_args, _var_kwargs], ids=["args", "kwargs"])
def test_variadic(func: Callable[..., Any]) -> None:
    with pytest.raises(NotImplementedError):
        _infer(func)


def test_doc_params() -> None:
    def f() -> None: ...

    f.__doc__ = "f(a, b=1, [c]) -> z"
    assert _doc_params(f) == ["a", "b", "c"]

    f.__doc__ = "no signature line here"
    assert _doc_params(f) is None


def test_doc_signature() -> None:
    # builtins like `int` have no signature; fall back to the docstring
    assert _infer(int) == "(x: CanInt | CanIndex) -> int"


def _set_attr(x: Any) -> object:
    x.spam = 1
    return x


def _set_name(x: Any) -> object:
    class C:
        attr: Any = x

    return C


NO_PROTOCOL_CASES: list[Callable[[Any], Any]] = [
    lambda x: x.foo,
    _set_attr,
    _set_name,
]


@pytest.mark.parametrize(
    "func",
    NO_PROTOCOL_CASES,
    ids=["getattr", "setattr", "set_name"],
)
def test_no_protocol(func: Callable[[Any], Any]) -> None:
    with pytest.raises(NotImplementedError):
        _infer(func)


def test_infer_prints(capsys: pytest.CaptureFixture[str]) -> None:
    infer(abs)
    assert capsys.readouterr().out == "[R](x: CanAbs[R]) -> R\n"
