import enum
import sys
from typing import Generic, Literal, final


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

import optype.typing as opt


_IntT_co = TypeVar("_IntT_co", bound=int, covariant=True)


# `Any*` type aliases


@final
class Index(Generic[_IntT_co]):
    def __init__(self, x: _IntT_co, /) -> None:
        self._x = x

    def __index__(self, /) -> _IntT_co:
        return self._x


@final
class IntLike(Generic[_IntT_co]):
    def __init__(self, x: _IntT_co, /) -> None:
        self._x = x

    def __int__(self, /) -> _IntT_co:
        return self._x


@final
class FloatLike:
    def __init__(self, x: float, /) -> None:
        self._x = x

    def __float__(self, /) -> float:
        return self._x


@final
class ComplexLike:
    def __init__(self, x: complex, /) -> None:
        self._x = x

    def __complex__(self, /) -> complex:
        return self._x


_V_co = TypeVar("_V_co", covariant=True)


@final
class SequenceLike(Generic[_V_co]):
    def __init__(self, /, *values: _V_co) -> None:
        self._xs = values

    def __getitem__(self, index: int, /) -> _V_co:
        return self._xs[index]


def test_any_int() -> None:
    p_bool: opt.AnyInt = True
    p_int: opt.AnyInt[Literal[2]] = 2
    p_index: opt.AnyInt[Literal[3]] = Index(3)
    p_int_like: opt.AnyInt[Literal[4]] = IntLike(4)

    n_complex: opt.AnyInt = 5j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.AnyInt = "6"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_float() -> None:
    p_bool: opt.AnyFloat = True
    p_int: opt.AnyFloat = 1
    p_float: opt.AnyFloat = 2.0
    p_index: opt.AnyFloat = Index(3)
    p_float_like: opt.AnyFloat = FloatLike(4.0)

    n_int_like: opt.AnyFloat = IntLike(5)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: opt.AnyInt = 6j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.AnyInt = "7"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_complex() -> None:
    p_bool: opt.AnyComplex = True
    p_int: opt.AnyComplex = 1
    p_float: opt.AnyComplex = 2.0
    p_complex: opt.AnyComplex = 3j
    p_index: opt.AnyComplex = Index(4)
    p_float_like: opt.AnyComplex = FloatLike(5.0)
    p_complex_like: opt.AnyComplex = ComplexLike(6.0)

    n_int_like: opt.AnyComplex = IntLike(7)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.AnyComplex = "8"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_iterable() -> None:
    p_tuple: opt.AnyIterable[bool] = (False,)
    p_list: opt.AnyIterable[int] = [1]
    p_set: opt.AnyIterable[float] = {2.0}
    p_dict: opt.AnyIterable[str] = {"3": 4}
    p_range: opt.AnyIterable[complex] = range(5)
    p_generator: opt.AnyIterable[complex] = (i for i in range(6))
    p_str: opt.AnyIterable[str] = "7"
    p_bytes: opt.AnyIterable[int] = b"8"
    p_sequence: opt.AnyIterable[bytes] = SequenceLike(b"9")


class ColorChannel(enum.Enum):
    R, G, B = 0, 1, 2


def test_any_literal() -> None:
    p_none: opt.AnyLiteral = None
    p_bool: opt.AnyLiteral = True
    p_int: opt.AnyLiteral = 1
    n_float: opt.AnyLiteral = 2.0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: opt.AnyLiteral = 3j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    p_str: opt.AnyLiteral = "4"
    p_bytes: opt.AnyLiteral = b"5"
    p_enum: opt.AnyLiteral = ColorChannel.R


# `Empty*` type aliases


def test_empty_string() -> None:
    empty: opt.EmptyString = ""
    not_empty: opt.EmptyString = "0"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_bytes() -> None:
    empty: opt.EmptyBytes = b""
    not_empty: opt.EmptyBytes = b"0"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_tuple() -> None:
    empty: opt.EmptyTuple = ()
    not_empty: opt.EmptyTuple = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_list() -> None:
    empty: opt.EmptyList = []
    not_empty: opt.EmptyList = [0]  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_set() -> None:
    empty: opt.EmptySet = set()
    not_empty: opt.EmptySet = {0}  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_dict() -> None:
    empty: opt.EmptyDict = {}
    not_empty: opt.EmptyDict = {0: 0}  # type: ignore[assignment,misc]  # pyright: ignore[reportAssignmentType]


def test_empty_iterable() -> None:
    empty: opt.EmptyIterable = ()
    not_empty: opt.EmptyIterable = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


# `Literal*` type aliases


def test_literal_bool() -> None:
    p_true: opt.LiteralBool = False
    p_false: opt.LiteralBool = True

    n_0: opt.LiteralBool = 0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_none: opt.LiteralBool = None  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(opt.LiteralBool.__args__) == 2  # type: ignore[attr-defined]


def test_literal_byte() -> None:
    p_0: opt.LiteralByte = 0
    p_255: opt.LiteralByte = 255

    n_256: opt.LiteralByte = 256  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_m1: opt.LiteralByte = -1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_bool: opt.LiteralByte = False  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_ix0: opt.LiteralByte = Index(0)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(opt.LiteralByte.__args__) == 256  # type: ignore[attr-defined]


def test_just() -> None:
    # positives

    class A: ...

    class B(A): ...

    class C(B): ...

    a, b, c = A(), B(), C()

    b_b: opt.Just[B] = b
    b_a: opt.Just[B] = a  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # TODO: `pyright: ignore` after https://github.com/python/typeshed/issues/12997
    b_c: opt.Just[B] = c  # type: ignore[assignment]


def test_just_int() -> None:
    # instance assignment: true negatives
    x_int: int = int("42")
    x_int_lit: Literal[42] = 42
    tn_int: opt.JustInt = x_int
    tn_lit: opt.JustInt = x_int_lit

    # instance assignment: true positives
    x_bool: bool = bool(x_int)
    x_true: Literal[True] = True
    tp_bool: opt.JustInt = x_bool  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_true: opt.JustInt = x_true  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # type assignment
    tn_int_type: type[opt.JustInt] = int
    tp_int_type: type[opt.JustInt] = bool  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
