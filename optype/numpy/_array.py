from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

import optype.numpy._compat as _x
from optype.numpy._dtype import DType


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeVar, runtime_checkable
else:
    from typing_extensions import Protocol, Self, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "Array",
    "CanArray",
    "CanArrayFinalize",
    "CanArrayWrap",
    "HasArrayInterface",
    "HasArrayPriority",
]


_AnyShape: TypeAlias = tuple[int, ...]

# NumPy array with optional type params for shape and generic dtype.
_ShapeT = TypeVar("_ShapeT", bound=_AnyShape)
_ScalarT = TypeVar("_ScalarT", bound=np.generic, default=np.generic)
Array: TypeAlias = np.ndarray[_ShapeT, np.dtype[_ScalarT]]


_ShapeT0 = TypeVar("_ShapeT0", bound=_AnyShape, default=_AnyShape)
_ShapeT_co = TypeVar("_ShapeT_co", covariant=True, bound=_AnyShape, default=_AnyShape)

_DTypeT = TypeVar("_DTypeT", bound=DType)
_DTypeT_co = TypeVar("_DTypeT_co", bound=DType, covariant=True, default=DType)

if _x.NP2 and not _x.NP20:
    # numpy >= 2.1: shape is covariant
    @runtime_checkable
    class CanArray(Protocol[_ShapeT_co, _DTypeT_co]):
        def __array__(self, /) -> np.ndarray[_ShapeT_co, _DTypeT_co]: ...

else:
    # numpy < 2.1: shape is invariant
    @runtime_checkable
    class CanArray(Protocol[_ShapeT0, _DTypeT_co]):
        def __array__(self, /) -> np.ndarray[_ShapeT0, _DTypeT_co]: ...

###########################


# this is almost always a `ndarray`, but setting a `bound` might break in some
# edge cases
_T_contra = TypeVar("_T_contra", contravariant=True, default=object)


@runtime_checkable
class CanArrayFinalize(Protocol[_T_contra]):
    def __array_finalize__(self, obj: _T_contra, /) -> None: ...


@runtime_checkable
class CanArrayWrap(Protocol):
    if _x.NP2:

        def __array_wrap__(
            self,
            array: np.ndarray[_ShapeT, _DTypeT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            return_scalar: bool = ...,
            /,
        ) -> np.ndarray[_ShapeT, _DTypeT] | Self: ...

    else:

        def __array_wrap__(
            self,
            array: np.ndarray[_ShapeT, _DTypeT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            /,
        ) -> np.ndarray[_ShapeT, _DTypeT] | Self: ...


_ArrayInterfaceT_co = TypeVar(
    "_ArrayInterfaceT_co",
    covariant=True,
    bound="Mapping[str, object]",
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
