from __future__ import annotations

import enum
import sys
from typing import Literal, NoReturn, Protocol, TypeAlias


if sys.version_info >= (3, 13):
    from typing import (
        LiteralString,
        Never,
        Self,
        TypeVar,
        TypedDict,
        Unpack,
        final,
        override,
    )
else:
    from typing_extensions import (
        LiteralString,
        Never,
        Self,
        TypeVar,
        TypedDict,
        Unpack,
        final,
        override,
    )

from ._core import _can as _c


__all__ = (
    "AnyComplex",
    "AnyFloat",
    "AnyInt",
    "AnyIterable",
    "AnyLiteral",
    "EmptyBytes",
    "EmptyDict",
    "EmptyIterable",
    "EmptyList",
    "EmptySet",
    "EmptyString",
    "EmptyTuple",
    "Just",
    "JustComplex",
    "JustFloat",
    "JustInt",
    "LiteralBool",
    "LiteralByte",
)


def __dir__() -> tuple[str, ...]:
    return __all__


_T = TypeVar("_T")
_IntT = TypeVar("_IntT", bound=int, default=int)
_ValueT = TypeVar("_ValueT", default=object)


# "Just" types


class Just(Protocol[_T]):
    """
    Experimental "invariant" wrapper type, so that `Invariant[int]` only accepts `int`
    but not `bool` (or any other `int` subtypes).

    Important:
        This requires `pyright>=1.390` to work.

    Warning:
        In mypy this doesn't work with the special-cased `float` and `complex`,
        caused by (at least) one of the >1200 confirmed mypy bugs.

    Note:
        The only reason that this worked on mypy `1.13.0` and below, is because
        of (yet another) bug, where mypy blindly aassumes that the setter of a property
        has the same parameter type as the return type of the getter, even though that's
        the main usecase of `@property`...
    """

    @property  # type: ignore[explicit-override]  # seriously..?
    @override
    def __class__(self, /) -> type[_T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @__class__.setter
    @override
    def __class__(self, t: type[_T], /) -> None: ...


class JustInt(Just[int], Protocol):
    """
    A `pyright<1.390` compatible version of `Just[int]`.

    Example:
        This example shows a situation you want to accept instances of just `int`,
        and reject and other instances, even they're subtypes of `int` such as `bool`,

        ```python
        def f(x: int, /) -> int:
            assert type(x) is int
            return x


        f(1337)  # accepted
        f(True)  # accepted :(
        ```

        But because `bool <: int`, booleans are also assignable to `int`, which we
        don't want to allow.

        Previously, it was considered impossible to achieve this using just python
        typing, and it was told that a PEP would be necessary to make it work.

        But this is not the case at all, and `optype` provides a clean counter-example
        that works with pyright and (even) mypy:

        ```python
        def f_fixed(x: JustInt, /) -> int:
            assert type(x) is int
            return x  # accepted


        f_fixed(1337)  # accepted
        f_fixed(True)  # rejected :)
        ```

    Note:
        On `mypy` this behaves in the same way as `Just[int]`, which accidentally
        already works because of a mypy bug.
    """

    def __new__(cls, x: str | bytes | bytearray, /, base: _c.CanIndex) -> Self: ...


class JustFloat(Just[float], Protocol):
    """
    Like `Just[float]`, but also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this DOES NOT WORK IN MYPY (due to a bug related to the
        special-cased "promotion rules" of `float`)
    """

    def hex(self, /) -> str: ...


class JustComplex(Just[complex], Protocol):
    """
    Like `Just[complex]`, but also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this DOES NOT WORK IN MYPY (due to a bug related to the
        special-cased "promotion rules" of `complex`).
    """

    def __new__(cls, /, real: complex | AnyFloat, imag: complex | AnyFloat) -> Self: ...


# Anything that can *always* be converted to an `int` / `float` / `complex`
AnyInt: TypeAlias = _IntT | _c.CanInt[_IntT] | _c.CanIndex[_IntT]

AnyFloat: TypeAlias = _c.CanFloat | _c.CanIndex
if sys.version_info >= (3, 11):
    AnyComplex: TypeAlias = _c.CanComplex | _c.CanFloat | _c.CanIndex
else:
    # `complex.__complex__` didn't exists before Python 3.11
    AnyComplex: TypeAlias = complex | _c.CanComplex | _c.CanFloat | _c.CanIndex

# Anything that can be iterated over, e.g. in a `for` loop,`builtins.iter`,
# `builtins.enumerate`, or `numpy.array`.
AnyIterable: TypeAlias = _c.CanIter[_c.CanNext[_ValueT]] | _c.CanGetitem[int, _ValueT]

# The closest supertype of a `Literal`, i.e. the allowed types that can be
# passed to `typing.Literal`.
AnyLiteral: TypeAlias = bool | JustInt | LiteralString | bytes | enum.Enum | None


# Empty collection type aliases


@final
class _EmptyTypedDict(TypedDict):
    pass


EmptyString: TypeAlias = Literal[""]
EmptyBytes: TypeAlias = Literal[b""]
EmptyTuple: TypeAlias = (
    # this should be the only that's needed here, but in practice we need 3
    # other variants, that are equivalent, but are somehow treated as
    # different types by pyright or mypy or both
    tuple[()]
    # both mypy and pyright don't simplify this to `tuple[()]`, even though
    # it's equivalent, and `tuple[()]` is much easier to read
    | tuple[Never, ...]
    # in pyright `NoReturn` and `Never` aren't identical, only equivalent
    | tuple[NoReturn, ...]
    # this is what infers the result of `tuple[Never, ...] + tuple[()]` as...
    | tuple[Unpack[tuple[Never, ...]]]
)
EmptyList: TypeAlias = list[Never]
EmptySet: TypeAlias = set[Never]
EmptyDict: TypeAlias = dict[object, Never] | _EmptyTypedDict
EmptyIterable: TypeAlias = AnyIterable[Never]


# Literal

LiteralBool: TypeAlias = Literal[False, True]  # noqa: RUF038
LiteralByte: TypeAlias = Literal[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
]  # fmt: skip
