# ruff: noqa: ERA001
# mypy: disable-error-code="no-any-explicit, no-any-decorated"
from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from optype._core._can import CanIndex

from ._dtypes import Bool, DType


if sys.version_info >= (3, 13):
    from types import CapsuleType
    from typing import Never, Protocol, Self, TypeVar, runtime_checkable
else:
    from typing_extensions import (
        CapsuleType,
        Never,
        Protocol,
        Self,
        TypeVar,
        runtime_checkable,
    )


if TYPE_CHECKING:
    import enum

    from ._device import Device


__all__ = ["Array"]


_IndexSingle: TypeAlias = CanIndex | slice | types.EllipsisType | None
_IndexMulti: TypeAlias = _IndexSingle | tuple[_IndexSingle, ...]

_T_co = TypeVar("_T_co", covariant=True, bound=DType, default=DType)


@runtime_checkable
class BaseArray(Protocol[_T_co]):
    @property
    def dtype(self, /) -> _T_co: ...
    @property
    def device(self, /) -> Device: ...
    def to_device(self, device: Never, /, *, stream: int | None = None) -> Self: ...

    # TODO: Return a dedicated array-api namespace type.
    def __array_namespace__(
        self,
        /,
        *,
        api_version: Literal["2023.12"] | None = None,
    ) -> types.ModuleType: ...

    def __dlpack__(
        self,
        /,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, Literal[0]] | None = None,
        # NOTE: thee `copy: bool | None` is omitted because of a `numpy<2.2` typing bug
    ) -> CapsuleType: ...
    def __dlpack_device__(self, /) -> tuple[enum.Enum | int, int]: ...


@runtime_checkable
class Array(BaseArray[_T_co], Protocol[_T_co]):
    @property
    def shape(self, /) -> tuple[int, ...]: ...
    @property
    def ndim(self, /) -> int: ...
    @property
    def size(self, /) -> int: ...

    @property
    def T(self, /) -> Self: ...  # noqa: N802
    @property
    def mT(self, /) -> Self: ...  # noqa: N802

    # NOTE: for `numpy.ndarray` compatibility, we can't accept or return `Array` here.
    def __lt__(self, x: float, /) -> Any: ...  # pyright: ignore[reportAny]
    def __le__(self, x: float, /) -> Any: ...  # pyright: ignore[reportAny]
    def __ge__(self, x: float, /) -> Any: ...  # pyright: ignore[reportAny]
    def __gt__(self, x: float, /) -> Any: ...  # pyright: ignore[reportAny]

    # binary arithmetic operators
    def __matmul__(self, x: Self, /) -> Self: ...
    def __add__(self, x: int | Self, /) -> Self: ...
    def __sub__(self, x: int | Self, /) -> Self: ...
    def __mul__(self, x: int | Self, /) -> Self: ...
    def __pow__(self, x: int | Self, /) -> Self: ...
    def __mod__(self, x: int | Self, /) -> Self: ...

    def __rmatmul__(self, x: Self, /) -> Self: ...
    def __radd__(self, x: int, /) -> Self: ...
    def __rsub__(self, x: int, /) -> Self: ...
    def __rmul__(self, x: int, /) -> Self: ...
    def __rpow__(self, x: int, /) -> Self: ...
    def __rmod__(self, x: int, /) -> Self: ...

    # NOTE: uncommenting these methods causes causes type-checking performance issues
    # def __truediv__(self, x: float | Self, /) -> Array[_T_co] | Array[Float0]: ...
    # def __floordiv__(self, x: float | Self, /) -> Array[_T_co] | Array[Float0]: ...
    # def __rtruediv__(self, x: float, /) -> Array[_T_co] | Array[Float0]: ...
    # def __rfloordiv__(self, x: float, /) -> Array[_T_co] | Array[Float0]: ...

    # bitwise operators
    # TODO: fix the `numpy.ndarray` bitwise ops (currently impossible to match against)
    # NOTE: uncommenting the following causes pyrght performance issues
    # def __and__(self, x: Any, /) -> Array[Integer_co]: ...
    # def __or__(self, x: Any, /) -> Array[Integer_co]: ...
    # def __xor__(self, x: Any, /) -> Array[Integer_co]: ...

    # subscripting operators
    def __getitem__(self, k: _IndexMulti | Array[Bool], /) -> Self: ...

    # TODO: require the scalar value to match the array dtype
    def __setitem__(self, k: _IndexMulti | Array, v: complex | Array, /) -> None: ...

    # builtin type conversion
    def __bool__(self, /) -> bool: ...
    def __index__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __float__(self, /) -> float: ...
    def __complex__(self, /) -> complex: ...

    # unary arithmetic operators
    def __pos__(self, /) -> Self: ...
    def __neg__(self, /) -> Self: ...

    # TODO: Re-enable after https://github.com/numpy/numpy/pull/27659
    # def __abs__(self, /) -> Array[Any]: ...
    # def __invert__(self, /) -> Self: ...
