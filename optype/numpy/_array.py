from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

import optype.numpy._compat as _x
from optype._core._utils import set_module

from ._dtype import DType
from ._shape import AtLeast0D


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeAliasType, TypeVar, runtime_checkable
else:
    from typing_extensions import (
        Protocol,
        Self,
        TypeAliasType,
        TypeVar,
        runtime_checkable,
    )

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


_ND = TypeVar("_ND", bound=AtLeast0D)
_ND0_co = TypeVar("_ND0_co", covariant=True, bound=AtLeast0D, default=AtLeast0D)
_ND0 = TypeVar("_ND0", bound=AtLeast0D, default=AtLeast0D)
_ST = TypeVar("_ST", bound=np.generic, default=np.generic)
_DT = TypeVar("_DT", bound=DType)
_DT_co = TypeVar("_DT_co", bound=DType, covariant=True, default=DType)


Array = TypeAliasType("Array", np.ndarray[_ND, np.dtype[_ST]], type_params=(_ND, _ST))
"""
```py
type Array[
    ND: (int, ...) = AtLeast0D,
    ST: np.generic = np.generic,
] = np.ndarray[ND, ST]
```
"""


###########################

if _x.NP2 and not _x.NP20:
    # numpy >= 2.1: shape is covariant
    @runtime_checkable
    @set_module("optype.numpy")
    class CanArray(Protocol[_ND0_co, _DT_co]):
        def __array__(self, /) -> np.ndarray[_ND0_co, _DT_co]: ...

else:
    # numpy < 2.1: shape is invariant
    @runtime_checkable
    @set_module("optype.numpy")
    class CanArray(Protocol[_ND0, _DT_co]):
        def __array__(self, /) -> np.ndarray[_ND0, _DT_co]: ...


# this is almost always a `ndarray`, but setting a `bound` might break in some
# edge cases
_T_contra = TypeVar("_T_contra", contravariant=True, default=object)


@runtime_checkable
@set_module("optype.numpy")
class CanArrayFinalize(Protocol[_T_contra]):
    def __array_finalize__(self, obj: _T_contra, /) -> None: ...


@runtime_checkable
@set_module("optype.numpy")
class CanArrayWrap(Protocol):
    if _x.NP2:

        def __array_wrap__(
            self,
            array: np.ndarray[_ND, _DT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            return_scalar: bool = ...,
            /,
        ) -> np.ndarray[_ND, _DT] | Self: ...

    else:

        def __array_wrap__(
            self,
            array: np.ndarray[_ND, _DT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            /,
        ) -> np.ndarray[_ND, _DT] | Self: ...


_ArrayInterfaceT_co = TypeVar(
    "_ArrayInterfaceT_co",
    covariant=True,
    bound="Mapping[str, object]",
    default=dict[str, object],
)


@runtime_checkable
@set_module("optype.numpy")
class HasArrayInterface(Protocol[_ArrayInterfaceT_co]):
    @property
    def __array_interface__(self, /) -> _ArrayInterfaceT_co: ...


@runtime_checkable
@set_module("optype.numpy")
class HasArrayPriority(Protocol):
    @property
    def __array_priority__(self, /) -> float: ...
