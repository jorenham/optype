"""
Interfaces and type aliases for NumPy arrays, dtypes, and ufuncs.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import (
        Protocol,
        Required,
        TypeVar,
        TypedDict,
        overload,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        Protocol,
        Required,  # noqa: TCH002
        TypeVar,
        TypedDict,
        overload,
        runtime_checkable,
    )


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import NotImplementedType

    from optype import CanBuffer, CanIter, CanIterSelf

    from ._aliases import Array


__all__ = (
    'ArgArray',
    'CanArray',
    'CanArrayFinalize',
    'CanArrayFunction',
    'CanArrayWrap',
    'HasArrayInterface',
    'HasArrayPriority',
)


_NP_V2: Final[bool] = (_NP_VERSION := np.__version__).startswith('2.')
if not _NP_V2:
    assert _NP_VERSION.startswith('1.'), f'numpy {_NP_VERSION} is unsupported'


_Shape0: TypeAlias = tuple[int, ...]

_S_CanArray = TypeVar('_S_CanArray', infer_variance=True, bound=_Shape0)
_T_CanArray = TypeVar('_T_CanArray', infer_variance=True, bound=np.generic)
_DT_CanArray = TypeVar('_DT_CanArray', bound=np.dtype[Any])


@runtime_checkable
class CanArray(Protocol[_S_CanArray, _T_CanArray]):
    @overload
    def __array__(
        self,
        dtype: None = ...,
        /,
    ) -> Array[_S_CanArray, _T_CanArray]: ...
    @overload
    def __array__(
        self,
        dtype: _DT_CanArray,
        /,
    ) -> np.ndarray[_S_CanArray, _DT_CanArray]: ...


_V__NestedSequence = TypeVar(
    '_V__NestedSequence',
    infer_variance=True,
    bound=object,
)


@runtime_checkable
class _NestedSequence(Protocol[_V__NestedSequence]):
    def __len__(self, /) -> int: ...
    def __getitem__(self, i: int, /) -> (
        _V__NestedSequence
        | _NestedSequence[_V__NestedSequence]
    ): ...


_S_ArgArray = TypeVar('_S_ArgArray', bound=_Shape0)
_T_ArgArray_np = TypeVar('_T_ArgArray_np', bound=np.generic)
_T_ArgArray_py = TypeVar(
    '_T_ArgArray_py',
    bound=bool | int | float | complex | str | bytes,
)

ArgArray: TypeAlias = (
    CanArray[_S_ArgArray, _T_ArgArray_np]
    | _NestedSequence[CanArray[Any, _T_ArgArray_np]]
    | _T_ArgArray_py
    | _NestedSequence[_T_ArgArray_py]
)
"""
Generic array-like that can be passed to e.g. `np.array` or `np.asaray`.

    - `Shape: tuple[int, ...]`
    - `TypeNP: np.generic`
    - `TypePY: bool | int | float | complex | str | bytes`
"""


_F_CanArrayFunction = TypeVar(
    '_F_CanArrayFunction',
    infer_variance=True,
    bound='Callable[..., Any]',
)
_R_CanArrayFunction = TypeVar(
    '_R_CanArrayFunction',
    infer_variance=True,
    bound=object,
)


@runtime_checkable
class CanArrayFunction(Protocol[_F_CanArrayFunction, _R_CanArrayFunction]):
    def __array_function__(
        self,
        /,
        func: _F_CanArrayFunction,
        # although this could be tighter, this ensures numpy.typing compat
        types: CanIter[CanIterSelf[type[CanArrayFunction[Any, Any]]]],
        # ParamSpec can only be used on *args and **kwargs for some reason...
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> NotImplementedType | _R_CanArrayFunction: ...


# this is almost always a `ndarray`, but setting a `bound` might break in some
# edge cases
_T_CanArrayFinalize = TypeVar('_T_CanArrayFinalize', infer_variance=True)


@runtime_checkable
class CanArrayFinalize(Protocol[_T_CanArrayFinalize]):
    def __array_finalize__(self, obj: _T_CanArrayFinalize, /) -> None: ...


_S_CanArrayWrap = TypeVar('_S_CanArrayWrap')
_D_CanArrayWrap = TypeVar('_D_CanArrayWrap', bound=np.dtype[Any])


@runtime_checkable
class CanArrayWrap(Protocol):
    if _NP_V2:
        def __array_wrap__(
            self,
            array: np.ndarray[_S_CanArrayWrap, _D_CanArrayWrap],
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = ...,
            return_scalar: bool = ...,
            /,
        ) -> np.ndarray[_S_CanArrayWrap, _D_CanArrayWrap]: ...
    else:
        def __array_wrap__(
            self,
            array: np.ndarray[_S_CanArrayWrap, _D_CanArrayWrap],
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = ...,
            /,
        ) -> np.ndarray[_S_CanArrayWrap, _D_CanArrayWrap]: ...


_ArrayInterfaceDescr: TypeAlias = list[
    tuple[str, str]
    | tuple[str, str, tuple[int, ...]]
    | tuple[str, '_ArrayInterfaceDescr']
]


class _ArrayInterface(TypedDict, total=False):
    version: Required[int]
    shape: Required[tuple[int, ...]]
    typestr: Required[str]

    offset: int
    strides: tuple[int, ...] | None
    data: tuple[int, bool] | CanBuffer[Any] | None
    mask: HasArrayInterface | None
    descr: _ArrayInterfaceDescr


_V_HasArrayInterface = TypeVar(
    '_V_HasArrayInterface',
    infer_variance=True,
    bound='Mapping[str, Any]',
    default=_ArrayInterface,
)


@runtime_checkable
class HasArrayInterface(Protocol[_V_HasArrayInterface]):
    @property
    def __array_interface__(self, /) -> _V_HasArrayInterface: ...


_V_HasArrayPriority = TypeVar(
    '_V_HasArrayPriority',
    infer_variance=True,
    bound=float,
    default=float,
)


@runtime_checkable
class HasArrayPriority(Protocol[_V_HasArrayPriority]):
    @property
    def __array_priority__(self, /) -> _V_HasArrayPriority: ...
