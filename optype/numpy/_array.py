"""Interfaces and type aliases for NumPy arrays, dtypes, and ufuncs."""
import sys
from collections.abc import Callable, Mapping
from types import NotImplementedType
from typing import (
    Any,
    ClassVar,
    Final,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np

from optype import CanContains, CanIter, CanLen, CanNext

from ._shape import AtLeast0D


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

_NP_V1: Final[bool] = np.__version__.startswith('1.')


_S = TypeVar('_S', bound=np.generic)
_S_co = TypeVar('_S_co', bound=np.generic, covariant=True)
_ND = TypeVar('_ND', bound=AtLeast0D)

Array: TypeAlias = np.ndarray[_ND, np.dtype[_S]]


@runtime_checkable
class CanArray(Protocol[_ND, _S_co]):
    @overload
    def __array__(
        self,
        __dtype: None = None,
        copy: bool | None = None,
    ) -> Array[_ND, _S_co]: ...
    @overload
    def __array__(
        self,
        __dtype: np.dtype[_S],
        copy: bool | None = None,
    ) -> Array[_ND, _S]: ...


_V_co = TypeVar('_V_co', bound=object, covariant=True)


@runtime_checkable
class _NestedSequence(Protocol[_V_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, __i: int) -> _V_co | Self: ...


_PyScalar: TypeAlias = bool | int | float | complex | str | bytes
_S_py = TypeVar('_S_py', bound=_PyScalar)

# `SomeArray` is an array-like with at least 0 dimensions, and type params
# - shape, `: tuple[int, ...]`
# - scalar type (numpy), `: np.generic`
# - scalar type (python), `: bool | int | float | complex | str | bytes`
SomeArray: TypeAlias = (
    CanArray[_ND, _S]
    | _NestedSequence[CanArray[AtLeast0D, _S]]
    | _S_py
    | _NestedSequence[_S_py]
)


_T = TypeVar('_T')


class _Container(CanLen, CanContains[_T], CanIter[CanNext[_T]]): ...


_F_contra = TypeVar(
    '_F_contra',
    bound=Callable[..., object],
    contravariant=True,
)
_Y_co = TypeVar('_Y_co', bound=object, covariant=True)


@runtime_checkable
class CanArrayFunction(Protocol[_F_contra, _Y_co]):
    def __array_function__(
        self,
        func: _F_contra,
        types: _Container[type['CanArrayFunction[Any, Any]']],
        # ParamSpec can only be used on *args and **kwargs for some reason...
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> NotImplementedType | _Y_co: ...


_A_contra = TypeVar('_A_contra', bound=Array[Any, Any], contravariant=True)
_A_co = TypeVar('_A_co', bound=Array[Any, Any], covariant=True)


@runtime_checkable
class CanArrayFinalize(Protocol[_A_contra]):
    def __array_finalize__(self, __arr: _A_contra) -> None: ...


@runtime_checkable
class CanArrayWrap(Protocol[_A_contra, _A_co]):
    if _NP_V1:
        def __array_wrap__(
            self,
            __arr: _A_contra,
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = None,
        ) -> _A_co:
            ...
    else:
        def __array_wrap__(
            self,
            __arr: _A_contra,
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = None,
            return_scalar: bool = False,
        ) -> _A_co:
            ...


_P_co = TypeVar('_P_co', bound=float, covariant=True)


@runtime_checkable
class _HasArrayPriorityProperty(Protocol[_P_co]):
    @property
    def __array_priority__(self) -> _P_co: ...


@runtime_checkable
class _HasArrayPriorityClassVar(Protocol):
    __array_priority__: ClassVar[float]


HasArrayPriority: TypeAlias = (
    _HasArrayPriorityProperty[_P_co] | _HasArrayPriorityClassVar
)
