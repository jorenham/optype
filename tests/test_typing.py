import enum
from typing import final

import optype.typing as opt

# `Any*` type aliases


@final
class SimpleIndex[IntT: int]:
    __slots__ = ("_x",)

    def __init__(self, x: IntT, /) -> None:
        self._x = x

    def __index__(self, /) -> IntT:
        return self._x


@final
class SimpleInt[IntT: int]:
    __slots__ = ("_x",)

    def __init__(self, x: IntT, /) -> None:
        self._x = x

    def __int__(self, /) -> IntT:
        return self._x


@final
class SimpleFloat:
    __slots__ = ("_x",)

    def __init__(self, x: float, /) -> None:
        self._x = x

    def __float__(self, /) -> float:
        return self._x


@final
class SimpleComplex:
    __slots__ = ("_x",)

    def __init__(self, x: complex, /) -> None:
        self._x = x

    def __complex__(self, /) -> complex:
        return self._x


@final
class SequenceLike[V]:
    def __init__(self, /, *values: V) -> None:
        self._xs = values

    def __getitem__(self, index: int, /) -> V:
        return self._xs[index]


def test_any_int() -> None:
    p_bool: opt.AnyInt = True
    p_int: opt.AnyInt = 2
    p_index: opt.AnyInt = SimpleIndex(3)
    p_int_like: opt.AnyInt = SimpleInt(4)

    n_complex: opt.AnyInt = 5j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.AnyInt = "6"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_float() -> None:
    p_bool: opt.AnyFloat = True
    p_int: opt.AnyFloat = 1
    p_float: opt.AnyFloat = 2.0
    p_index: opt.AnyFloat = SimpleIndex(3)
    p_float_like: opt.AnyFloat = SimpleFloat(4.0)

    n_int_like: opt.AnyFloat = SimpleInt(5)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_complex: opt.AnyInt = 6j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_str: opt.AnyInt = "7"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_any_complex() -> None:
    p_bool: opt.AnyComplex = True
    p_int: opt.AnyComplex = 1
    p_float: opt.AnyComplex = 2.0
    p_complex: opt.AnyComplex = 3j
    p_index: opt.AnyComplex = SimpleIndex(4)
    p_float_like: opt.AnyComplex = SimpleFloat(5.0)
    p_complex_like: opt.AnyComplex = SimpleComplex(6.0)

    n_int_like: opt.AnyComplex = SimpleInt(7)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
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
    not_empty: opt.EmptyList = [0]  # type: ignore[list-item]  # pyright: ignore[reportAssignmentType]


def test_empty_set() -> None:
    empty: opt.EmptySet = set()
    not_empty: opt.EmptySet = {0}  # type: ignore[arg-type]  # pyright: ignore[reportAssignmentType]


def test_empty_dict() -> None:
    empty: opt.EmptyDict = {}
    not_empty: opt.EmptyDict = {0: 0}  # type: ignore[dict-item]  # pyright: ignore[reportAssignmentType]


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
    assert len(opt.LiteralBool.__value__.__args__) == 2


def test_literal_byte() -> None:
    p_0: opt.LiteralByte = 0
    p_255: opt.LiteralByte = 255

    n_256: opt.LiteralByte = 256  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_m1: opt.LiteralByte = -1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_bool: opt.LiteralByte = False  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    n_ix0: opt.LiteralByte = SimpleIndex(0)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # mypy doesn't understand `Literal` aliases...
    assert len(opt.LiteralByte.__value__.__args__) == 256
