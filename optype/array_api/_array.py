# mypy: disable-error-code="no-any-explicit, no-any-decorated"
from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from optype._core._can import CanIndex


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeVar, override, runtime_checkable
else:
    from typing_extensions import Protocol, Self, TypeVar, override, runtime_checkable


if TYPE_CHECKING:
    from ._common import APIVersion


from optype.dlpack import CanDLPack, CanDLPackDevice


__all__ = ["Array", "CanArrayNamespace"]


Shape: TypeAlias = tuple[int, ...]
DType: TypeAlias = object  # TODO
Device: TypeAlias = object  # TODO

_IndexSingle: TypeAlias = CanIndex | slice | types.EllipsisType | None
_IndexMulti: TypeAlias = _IndexSingle | tuple[_IndexSingle, ...]

_NamespaceT_co = TypeVar("_NamespaceT_co", covariant=True, default=types.ModuleType)
_ShapeT_co = TypeVar("_ShapeT_co", covariant=True, bound=Shape, default=Shape)
_DTypeT_co = TypeVar("_DTypeT_co", covariant=True, bound=DType, default=Any)
_DeviceT_co = TypeVar("_DeviceT_co", covariant=True, bound=Device, default=Any)

_MT = TypeVar("_MT", bound=int)
_NT = TypeVar("_NT", bound=int)

_Shape_mT = TypeVar(
    "_Shape_mT",
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    # because we can't see more that 4 dimensions anyway
    tuple[int, ...],
)
_ShapeT = TypeVar("_ShapeT", bound=Shape)
_DTypeT = TypeVar("_DTypeT", bound=DType)
_DeviceT = TypeVar("_DeviceT", bound=Device)


@runtime_checkable
class CanArrayNamespace(Protocol[_NamespaceT_co]):
    def __array_namespace__(
        self,
        /,
        *,
        api_version: APIVersion | None = None,
    ) -> _NamespaceT_co: ...


# pyright has a weird variance-related bug here
@runtime_checkable
class Array(  # noqa: PLW1641  # pyright: ignore[reportInvalidTypeVarUse]
    CanArrayNamespace[_NamespaceT_co],
    CanDLPack,
    CanDLPackDevice,
    Protocol[_ShapeT_co, _DTypeT_co, _DeviceT_co, _NamespaceT_co],
):
    @property
    def shape(self, /) -> _ShapeT_co: ...
    @property
    def dtype(self, /) -> _DTypeT_co: ...
    @property
    def device(self, /) -> _DeviceT_co: ...

    @property
    def ndim(self, /) -> int: ...
    @property
    def size(self, /) -> int: ...

    @property
    def T(  # noqa: N802
        self: Array[tuple[_MT, _NT], _DTypeT, _DeviceT],
        /,
    ) -> Array[tuple[_NT, _MT], _DTypeT, _DeviceT]: ...
    @property
    def mT(  # noqa: N802
        self: Array[_Shape_mT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_Shape_mT, _DTypeT, _DeviceT]: ...

    # builtin type conversion
    def __bool__(self: Array[tuple[Literal[1], ...]], /) -> bool: ...
    # TODO: require self dtype to be integral
    def __index__(self: Array[tuple[()]], /) -> int: ...
    # TODO: require self dtype to be real
    def __int__(self: Array[tuple[()]], /) -> int: ...
    def __float__(self: Array[tuple[()]], /) -> float: ...
    def __complex__(self: Array[tuple[()]], /) -> complex: ...

    # unary arithmetic operators
    def __pos__(self, /) -> Self: ...
    def __neg__(self, /) -> Self: ...
    # TODO: overload on self dtype: real -> self, complex -> float
    def __abs__(self, /) -> Array[_ShapeT_co, Any, _DeviceT_co]: ...
    # TODO: require self and return dtype as bool or int
    def __invert__(self, /) -> Array[_ShapeT_co, Any, _DeviceT_co]: ...

    # rich comparison operators
    # TODO: use specific boolean return dtype for all comparison ops
    @override
    def __eq__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: complex | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    @override
    def __ne__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: complex | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    # TODO: require self dtype as real for all inequality comparison ops
    def __lt__(
        self,
        x: float | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    def __le__(
        self,
        x: float | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    def __ge__(
        self,
        x: float | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    def __gt__(
        self,
        x: float | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...

    # binary arithmetic operators
    # NOTE: the aray-api does not specify `__divmod__`
    def __add__(
        self,
        x: complex | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    def __sub__(
        self,
        x: complex | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    def __mul__(
        self,
        x: complex | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    # TODO: overloads for specific shapes, see the docs for the exact overloads
    def __matmul__(
        self,
        x: complex | Array[tuple[int, ...], _DTypeT, _DeviceT],
        /,
    ) -> Array[tuple[int, ...], _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    # TODO: disallow int / int (it's unspecified)
    # TODO: always return float or complex dtype
    def __truediv__(
        self,
        x: complex | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    # TODO: disallow complex dtypes
    # TODO: always return float dtype
    def __floordiv__(
        self,
        x: float | Array[_ShapeT, Any, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, Any, _DeviceT_co | _DeviceT]: ...
    def __mod__(
        self,
        x: float | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    # NOTE: the aray-api does not specify the modulus parameter
    def __pow__(
        self,
        x: complex | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...

    # bitwise operators
    # TODO: always require and return integer dtypes for the bitshift operators
    def __lshift__(
        self,
        x: int | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    def __rshift__(
        self,
        x: int | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    # TODO: always require and return int or int dtypes for the logical bitwise ops
    def __and__(
        self,
        x: int | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    def __xor__(
        self,
        x: int | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...
    def __or__(
        self,
        x: int | Array[_ShapeT, _DTypeT, _DeviceT],
        /,
    ) -> Array[_ShapeT_co | _ShapeT, _DTypeT_co | _DTypeT, _DeviceT_co | _DeviceT]: ...

    # subscripting operators
    # TODO: overloads for specific shapes
    # TODO: require the index arrays to be of boolean dtype
    def __getitem__(
        self,
        k: _IndexMulti | Array,
        /,
    ) -> Array[tuple[int, ...], _DTypeT_co, _DeviceT_co]: ...
    # TODO: require the scalar value to match the array dtype
    def __setitem__(
        self: Array[Any, _DTypeT, _DeviceT],
        k: _IndexMulti | Array,
        v: complex | Array,
        /,
    ) -> None: ...

    # dlpack
    def to_device(
        self,
        device: _DeviceT,
        /,
        *,
        stream: int | None = None,
    ) -> Array[_ShapeT_co, _DTypeT_co, _DeviceT]: ...
