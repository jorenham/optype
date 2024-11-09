# ruff: noqa: N801
"""
Runtime protocols for NumPy's numeric scalar types.
See https://github.com/jorenham/optype/issues/25#issuecomment-2453445552
"""

import sys
from types import GenericAlias, ModuleType
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, runtime_checkable


if sys.version_info >= (3, 13):
    from typing import Self
else:
    from typing_extensions import Self


_ItemT = TypeVar("_ItemT", bound=complex)
_NewArgsT_co = TypeVar("_NewArgsT_co", covariant=True, bound=tuple[float, ...])

_NewArgs1: TypeAlias = tuple[float]
_NewArgs2: TypeAlias = tuple[float, float]


__all__ = [
    "Bool",
    "Complex128",
    "ComplexFloating_co",
    "Float64",
    "Floating",
    "Floating_co",
    "Integer",
    "Integer_co",
    "Number",
    "RealNumber",
]


@runtime_checkable
class _CanArrayNamespace(Protocol):
    def __array_namespace__(self, /, *, api_version: None = None) -> ModuleType: ...


@runtime_checkable
class _CanGetNewArgs(Protocol[_NewArgsT_co]):
    def __getnewargs__(self, /) -> _NewArgsT_co: ...


@runtime_checkable
class _CanClassGetItem(Protocol):
    def __class_getitem__(cls, k: object, /) -> GenericAlias: ...


# NOTE: forcibly set to invariant to work around the incorrect int/float promotion rules
@runtime_checkable
class _CanItem(Protocol[_ItemT]):  # type:ignore[misc]  # pyright: ignore[reportInvalidTypeVarUse]
    def item(self, /) -> _ItemT: ...


@runtime_checkable
class Integer_co(_CanArrayNamespace, _CanItem[int], Protocol):
    """Booleans and (signed and unsigned) integers."""

    if not TYPE_CHECKING:
        # NOTE: this is a workaround for `numpy<2.2`
        def __index__(self, /) -> Self: ...


@runtime_checkable
class Floating_co(_CanArrayNamespace, _CanItem[float], Protocol):
    """Booleans, integers, and real floats."""


@runtime_checkable
class ComplexFloating_co(_CanArrayNamespace, _CanItem[complex], Protocol):
    """Booleans, integers, real- and complex floats."""


@runtime_checkable
class Number(_CanArrayNamespace, _CanClassGetItem, _CanItem[complex], Protocol):
    """Integers, real- and complex floats, i.e. `ComplexFloating_co` minus `Bool`."""

    if not TYPE_CHECKING:
        # NOTE: this is a workaround for `numpy<2.2`
        def __round__(self, ndigits: int | None = None, /) -> int | Self: ...


@runtime_checkable
class RealNumber(_CanArrayNamespace, _CanClassGetItem, _CanItem[float], Protocol):
    """Integers and real floats, i.e. `Number` minus `Bool`."""

    def is_integer(self, /) -> bool: ...

    if not TYPE_CHECKING:
        # NOTE: this is a workaround for `numpy<2.2`
        def __round__(self, ndigits: int | None = None, /) -> int | Self: ...


@runtime_checkable
class Integer(Integer_co, Protocol):
    """Signed and unsigned integers."""

    def bit_count(self, /) -> int: ...


@runtime_checkable
class Floating(Floating_co, Protocol):
    """Real floats."""

    def as_integer_ratio(self, /) -> tuple[int, int]: ...


@runtime_checkable
class ComplexFloating(Number, Protocol):
    """
    IMPORTANT: On `python<3.11` `complex.__complex__` does not exist, which makes
    distringuishing between float and complex types impossible in certain cases.
    IMPORTANT: On `numpy<2.2` this does not work as expected when type-checking!
    Use `Complex128` or `Number` instead.
    """

    if sys.version_info >= (3, 11):

        def __complex__(self, /) -> complex: ...


@runtime_checkable
class Bool(_CanArrayNamespace, _CanItem[bool], Protocol):
    if not TYPE_CHECKING:
        # NOTE: this is a workaround for `numpy<2.2`
        def __index__(self, /) -> int: ...


@runtime_checkable
class Float64(_CanGetNewArgs[_NewArgs1], Floating, Protocol): ...


@runtime_checkable
class Complex128(_CanGetNewArgs[_NewArgs2], ComplexFloating, Protocol): ...
