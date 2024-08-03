from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import (
        Protocol,
        Self,
        TypeVar,
        overload,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        Protocol,
        Self,  # noqa: TCH002
        TypeVar,
        overload,
        runtime_checkable,
    )


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = (
    'Array',
    'CanArray',
    'CanArrayFinalize',
    'CanArrayWrap',
    'HasArrayInterface',
    'HasArrayPriority',
)

_NP_VERSION: Final = np.__version__
_NP_V2: Final = _NP_VERSION.startswith('2.')

if not _NP_V2:
    assert _NP_VERSION.startswith('1.'), f'numpy {_NP_VERSION} is unsupported'


_AnyND: TypeAlias = tuple[int, ...]
_ND_Array = TypeVar('_ND_Array', bound=_AnyND, default=_AnyND)
_ST_Array = TypeVar('_ST_Array', bound=np.generic, default=np.generic)

Array: TypeAlias = np.ndarray[_ND_Array, np.dtype[_ST_Array]]
"""NumPy array with optional type params for shape and generic dtype."""


_ND_CanArray = TypeVar(
    '_ND_CanArray',
    infer_variance=True,
    bound=_AnyND,
    default=_AnyND,
)
_ST_CanArray = TypeVar(
    '_ST_CanArray',
    infer_variance=True,
    bound=np.generic,
    default=np.generic,
)
_DT_CanArray = TypeVar('_DT_CanArray', bound=np.dtype[Any])


@runtime_checkable
class CanArray(Protocol[_ND_CanArray, _ST_CanArray]):
    @overload
    def __array__(
        self,
        dtype: None = ...,
        /,
    ) -> Array[_ND_CanArray, _ST_CanArray]: ...
    @overload
    def __array__(
        self,
        dtype: _DT_CanArray,
        /,
    ) -> np.ndarray[_ND_CanArray, _DT_CanArray]: ...


###########################


# this is almost always a `ndarray`, but setting a `bound` might break in some
# edge cases
_T_CanArrayFinalize = TypeVar(
    '_T_CanArrayFinalize',
    infer_variance=True,
    bound=object,
    default=object,
)


@runtime_checkable
class CanArrayFinalize(Protocol[_T_CanArrayFinalize]):
    def __array_finalize__(self, obj: _T_CanArrayFinalize, /) -> None: ...


_ND_CanArrayWrap = TypeVar('_ND_CanArrayWrap')
_DT_CanArrayWrap = TypeVar('_DT_CanArrayWrap', bound=np.dtype[Any])


@runtime_checkable
class CanArrayWrap(Protocol):
    if _NP_V2:
        def __array_wrap__(
            self,
            array: np.ndarray[_ND_CanArrayWrap, _DT_CanArrayWrap],
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = ...,
            return_scalar: bool = ...,
            /,
        ) -> np.ndarray[_ND_CanArrayWrap, _DT_CanArrayWrap] | Self: ...
    else:
        def __array_wrap__(
            self,
            array: np.ndarray[_ND_CanArrayWrap, _DT_CanArrayWrap],
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = ...,
            /,
        ) -> np.ndarray[_ND_CanArrayWrap, _DT_CanArrayWrap] | Self: ...


_V_HasArrayInterface = TypeVar(
    '_V_HasArrayInterface',
    infer_variance=True,
    bound='Mapping[str, Any]',
    default=dict[str, Any],
)


@runtime_checkable
class HasArrayInterface(Protocol[_V_HasArrayInterface]):
    @property
    def __array_interface__(self, /) -> _V_HasArrayInterface: ...


@runtime_checkable
class HasArrayPriority(Protocol):
    @property
    def __array_priority__(self, /) -> float: ...
