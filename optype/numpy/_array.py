from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

import optype.numpy._compat as _x


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeVar, overload, runtime_checkable
else:
    from typing_extensions import Protocol, Self, TypeVar, overload, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    'Array',
    'CanArray',
    'CanArrayFinalize',
    'CanArrayWrap',
    'HasArrayInterface',
    'HasArrayPriority',
]


_AnyShape: TypeAlias = tuple[int, ...]

_ShapeT = TypeVar('_ShapeT', bound=_AnyShape, default=_AnyShape)
_ShapeT_co = TypeVar('_ShapeT_co', bound=_AnyShape, covariant=True, default=_AnyShape)

_SCT = TypeVar('_SCT', bound=np.generic, default=np.generic)
_DT = TypeVar('_DT', bound=np.dtype[np.generic], default=np.dtype[np.generic])
_DT_co = TypeVar(
    '_DT_co',
    bound=np.dtype[np.generic],
    covariant=True,
    default=np.dtype[np.generic],
)


# NumPy array with optional type params for shape and generic dtype.
Array: TypeAlias = np.ndarray[_ShapeT, np.dtype[_SCT]]


if _x.NP2 and not _x.NP20:
    @runtime_checkable
    class CanArray(Protocol[_ShapeT_co, _DT_co]):
        @overload
        def __array__(self, dtype: None = ..., /) -> np.ndarray[_ShapeT_co, _DT_co]: ...
        @overload
        def __array__(self, dtype: _DT, /) -> np.ndarray[_ShapeT_co, _DT]: ...
else:
    @runtime_checkable
    class CanArray(Protocol[_ShapeT, _DT_co]):
        @overload
        def __array__(self, dtype: None = ..., /) -> np.ndarray[_ShapeT, _DT_co]: ...
        @overload
        def __array__(self, dtype: _DT, /) -> np.ndarray[_ShapeT_co, _DT]: ...


###########################


# this is almost always a `ndarray`, but setting a `bound` might break in some
# edge cases
_T_contra = TypeVar('_T_contra', contravariant=True, default=object)


@runtime_checkable
class CanArrayFinalize(Protocol[_T_contra]):
    def __array_finalize__(self, obj: _T_contra, /) -> None: ...


@runtime_checkable
class CanArrayWrap(Protocol):
    if _x.NP2:
        def __array_wrap__(
            self,
            array: np.ndarray[_ShapeT, _DT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            return_scalar: bool = ...,
            /,
        ) -> np.ndarray[_ShapeT, _DT] | Self: ...
    else:
        def __array_wrap__(
            self,
            array: np.ndarray[_ShapeT, _DT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            /,
        ) -> np.ndarray[_ShapeT, _DT] | Self: ...


_ArrayInterfaceT_co = TypeVar(
    '_ArrayInterfaceT_co',
    bound='Mapping[str, object]',
    covariant=True,
    default=dict[str, object],
)


@runtime_checkable
class HasArrayInterface(Protocol[_ArrayInterfaceT_co]):
    @property
    def __array_interface__(self, /) -> _ArrayInterfaceT_co: ...


@runtime_checkable
class HasArrayPriority(Protocol):
    @property
    def __array_priority__(self, /) -> float: ...
