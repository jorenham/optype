# ruff: noqa: ERA001, PYI018, F401

"""Interfaces and type aliases for NumPy arrays, dtypes, and ufuncs."""
import sys
from typing import (
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from ._shape import AtLeast0D


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


_T = TypeVar('_T')
_V_co = TypeVar('_V_co', bound=object, covariant=True)


@runtime_checkable
class _NestedSequence(Protocol[_V_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, __i: int) -> _V_co | Self: ...


_PyScalar: TypeAlias = bool | int | float | complex | str | bytes
_S_py = TypeVar('_S_py', bound=_PyScalar)

# `SomeScalar` has 2 bounded type params:
# - scalar type (numpy), `: np.generic`
# - scalar type (python), `: bool | int | float | complex | str | bytes`
SomeScalar: TypeAlias = CanArray[tuple[()], _S] | _S_py

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
