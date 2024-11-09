import enum
from typing import Generic, Literal, final

from typing_extensions import TypeVar

import optype as o


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
    p_bool: o.typing.AnyInt = True
    p_int: o.typing.AnyInt[Literal[2]] = 2
    p_index: o.typing.AnyInt[Literal[3]] = Index(3)
    p_int_like: o.typing.AnyInt[Literal[4]] = IntLike(4)

    n_complex: o.typing.AnyInt = 5j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: o.typing.AnyInt = "6"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_float() -> None:
    p_bool: o.typing.AnyFloat = True
    p_int: o.typing.AnyFloat = 1
    p_float: o.typing.AnyFloat = 2.0
    p_index: o.typing.AnyFloat = Index(3)
    p_float_like: o.typing.AnyFloat = FloatLike(4.0)

    n_int_like: o.typing.AnyFloat = IntLike(5)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: o.typing.AnyInt = 6j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: o.typing.AnyInt = "7"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_complex() -> None:
    p_bool: o.typing.AnyComplex = True
    p_int: o.typing.AnyComplex = 1
    p_float: o.typing.AnyComplex = 2.0
    p_complex: o.typing.AnyComplex = 3j
    p_index: o.typing.AnyComplex = Index(4)
    p_float_like: o.typing.AnyComplex = FloatLike(5.0)
    p_complex_like: o.typing.AnyComplex = ComplexLike(6.0)

    n_int_like: o.typing.AnyComplex = IntLike(7)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: o.typing.AnyComplex = "8"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_iterable() -> None:
    p_tuple: o.typing.AnyIterable[bool] = (False,)
    p_list: o.typing.AnyIterable[int] = [1]
    p_set: o.typing.AnyIterable[float] = {2.0}
    p_dict: o.typing.AnyIterable[str] = {"3": 4}
    p_range: o.typing.AnyIterable[complex] = range(5)
    p_generator: o.typing.AnyIterable[complex] = (i for i in range(6))
    p_str: o.typing.AnyIterable[str] = "7"
    p_bytes: o.typing.AnyIterable[int] = b"8"
    p_sequence: o.typing.AnyIterable[bytes] = SequenceLike(b"9")


class ColorChannel(enum.Enum):
    R, G, B = 0, 1, 2


def test_any_literal() -> None:
    p_none: o.typing.AnyLiteral = None
    p_bool: o.typing.AnyLiteral = True
    p_int: o.typing.AnyLiteral = 1
    n_float: o.typing.AnyLiteral = 2.0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: o.typing.AnyLiteral = 3j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    p_str: o.typing.AnyLiteral = "4"
    p_bytes: o.typing.AnyLiteral = b"5"
    p_enum: o.typing.AnyLiteral = ColorChannel.R


# `Empty*` type aliases


def test_empty_string() -> None:
    empty: o.typing.EmptyString = ""
    not_empty: o.typing.EmptyString = "0"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_bytes() -> None:
    empty: o.typing.EmptyBytes = b""
    not_empty: o.typing.EmptyBytes = b"0"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_tuple() -> None:
    empty: o.typing.EmptyTuple = ()
    not_empty: o.typing.EmptyTuple = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_list() -> None:
    empty: o.typing.EmptyList = []
    not_empty: o.typing.EmptyList = [0]  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_set() -> None:
    empty: o.typing.EmptySet = set()
    not_empty: o.typing.EmptySet = {0}  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_empty_dict() -> None:
    empty: o.typing.EmptyDict = {}
    not_empty: o.typing.EmptyDict = {0: 0}  # type: ignore[assignment,misc]  # pyright: ignore[reportAssignmentType]


def test_empty_iterable() -> None:
    empty: o.typing.EmptyIterable = ()
    not_empty: o.typing.EmptyIterable = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


# `Literal*` type aliases


def test_literal_bool() -> None:
    p_true: o.typing.LiteralBool = False
    p_false: o.typing.LiteralBool = True

    n_0: o.typing.LiteralBool = 0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_none: o.typing.LiteralBool = None  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(o.typing.LiteralBool.__args__) == 2  # type: ignore[attr-defined]


def test_literal_byte() -> None:
    p_0: o.typing.LiteralByte = 0
    p_255: o.typing.LiteralByte = 255

    n_256: o.typing.LiteralByte = 256  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_m1: o.typing.LiteralByte = -1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_bool: o.typing.LiteralByte = False  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_ix0: o.typing.LiteralByte = Index(0)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(o.typing.LiteralByte.__args__) == 256  # type: ignore[attr-defined]
