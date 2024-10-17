import enum
from typing import Generic, Literal

from typing_extensions import TypeVar

import optype as opt


_IntT_co = TypeVar("_IntT_co", bound=int, covariant=True)


# `Any*` type aliases


class Index(Generic[_IntT_co]):
    def __init__(self, x: _IntT_co, /) -> None:
        self._x = x

    def __index__(self, /) -> _IntT_co:
        return self._x


class IntLike(Generic[_IntT_co]):
    def __init__(self, x: _IntT_co, /) -> None:
        self._x = x

    def __int__(self, /) -> _IntT_co:
        return self._x


class FloatLike:
    def __init__(self, x: float, /) -> None:
        self._x = x

    def __float__(self, /) -> float:
        return self._x


class ComplexLike:
    def __init__(self, x: complex, /) -> None:
        self._x = x

    def __complex__(self, /) -> complex:
        return self._x


_V_co = TypeVar("_V_co", covariant=True)


class SequenceLike(Generic[_V_co]):
    def __init__(self, /, *values: _V_co) -> None:
        self._xs = values

    def __getitem__(self, index: int, /) -> _V_co:
        return self._xs[index]


def test_any_int() -> None:
    p_bool: opt.typing.AnyInt = True
    p_int: opt.typing.AnyInt[Literal[2]] = 2
    p_index: opt.typing.AnyInt[Literal[3]] = Index(3)
    p_int_like: opt.typing.AnyInt[Literal[4]] = IntLike(4)

    n_complex: opt.typing.AnyInt = 5j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.typing.AnyInt = "6"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_float() -> None:
    p_bool: opt.typing.AnyFloat = True
    p_int: opt.typing.AnyFloat = 1
    p_float: opt.typing.AnyFloat = 2.0
    p_index: opt.typing.AnyFloat = Index(3)
    p_float_like: opt.typing.AnyFloat = FloatLike(4.0)

    n_int_like: opt.typing.AnyFloat = IntLike(5)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: opt.typing.AnyInt = 6j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.typing.AnyInt = "7"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_complex() -> None:
    p_bool: opt.typing.AnyComplex = True
    p_int: opt.typing.AnyComplex = 1
    p_float: opt.typing.AnyComplex = 2.0
    p_complex: opt.typing.AnyComplex = 3j
    p_index: opt.typing.AnyComplex = Index(4)
    p_float_like: opt.typing.AnyComplex = FloatLike(5.0)
    p_complex_like: opt.typing.AnyComplex = ComplexLike(6.0)

    n_int_like: opt.typing.AnyComplex = IntLike(7)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.typing.AnyComplex = "8"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_iterable() -> None:
    p_tuple: opt.typing.AnyIterable[bool] = (False,)
    p_list: opt.typing.AnyIterable[int] = [1]
    p_set: opt.typing.AnyIterable[float] = {2.0}
    p_dict: opt.typing.AnyIterable[str] = {"3": 4}
    p_range: opt.typing.AnyIterable[complex] = range(5)
    p_generator: opt.typing.AnyIterable[complex] = (i for i in range(6))
    p_str: opt.typing.AnyIterable[str] = "7"
    p_bytes: opt.typing.AnyIterable[int] = b"8"
    p_sequence: opt.typing.AnyIterable[bytes] = SequenceLike(b"9")


class ColorChannel(enum.Enum):
    R, G, B = 0, 1, 2


def test_any_literal() -> None:
    p_none: opt.typing.AnyLiteral = None
    p_bool: opt.typing.AnyLiteral = True
    p_int: opt.typing.AnyLiteral = 1
    n_float: opt.typing.AnyLiteral = 2.0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: opt.typing.AnyLiteral = 3j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    p_str: opt.typing.AnyLiteral = "4"
    p_bytes: opt.typing.AnyLiteral = b"5"
    p_enum: opt.typing.AnyLiteral = ColorChannel.R


# `Empty*` type aliases


def test_empty_string() -> None:
    empty: opt.typing.EmptyString = ""
    not_empty: opt.typing.EmptyString = "0"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_bytes() -> None:
    empty: opt.typing.EmptyBytes = b""
    not_empty: opt.typing.EmptyBytes = b"0"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_tuple() -> None:
    empty: opt.typing.EmptyTuple = ()
    not_empty: opt.typing.EmptyTuple = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_list() -> None:
    empty: opt.typing.EmptyList = []
    not_empty: opt.typing.EmptyList = [0]  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_set() -> None:
    empty: opt.typing.EmptySet = set()
    not_empty: opt.typing.EmptySet = {0}  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_dict() -> None:
    empty: opt.typing.EmptyDict = {}
    not_empty: opt.typing.EmptyDict = {0: 0}  # type: ignore[assignment,misc]  # pyright: ignore[reportAssignmentType]


def test_empty_iterable() -> None:
    empty: opt.typing.EmptyIterable = ()
    not_empty: opt.typing.EmptyIterable = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


# `Literal*` type aliases


def test_literal_bool() -> None:
    p_true: opt.typing.LiteralBool = False
    p_false: opt.typing.LiteralBool = True

    n_0: opt.typing.LiteralBool = 0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_none: opt.typing.LiteralBool = None  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(opt.typing.LiteralBool.__args__) == 2  # type: ignore[attr-defined]


def test_literal_byte() -> None:
    p_0: opt.typing.LiteralByte = 0
    p_255: opt.typing.LiteralByte = 255

    n_256: opt.typing.LiteralByte = 256  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_m1: opt.typing.LiteralByte = -1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_bool: opt.typing.LiteralByte = False  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_ix0: opt.typing.LiteralByte = Index(0)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(opt.typing.LiteralByte.__args__) == 256  # type: ignore[attr-defined]
