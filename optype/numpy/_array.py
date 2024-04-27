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

from optype._can import CanSequence


if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

# ndim == 0
ShapeScalar: TypeAlias = tuple[()]
# ndim >= 0
ShapeArray: TypeAlias = tuple[int, ...]
# ndim >= 1
ShapeTensor: TypeAlias = tuple[int, Unpack[tuple[int, ...]]]

_S = TypeVar('_S', bound=np.generic)
_S_co = TypeVar('_S_co', bound=np.generic, covariant=True)

_ND = TypeVar('_ND', bound=ShapeArray)
_ND1 = TypeVar('_ND1', bound=ShapeTensor)
_D0 = TypeVar('_D0', bound=int)
_D1 = TypeVar('_D1', bound=int)
_D2 = TypeVar('_D2', bound=int)

Array: TypeAlias = np.ndarray[_ND, np.dtype[_S]]
Tensor: TypeAlias = np.ndarray[_ND1, np.dtype[_S]]
Array0D: TypeAlias = Array[tuple[()], _S]
Array1D: TypeAlias = Array[tuple[_D0], _S]
Array2D: TypeAlias = Array[tuple[_D0, _D1], _S]
Array3D: TypeAlias = Array[tuple[_D0, _D1, _D2], _S]


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


# there is currently no way to way for typecheckers to determine the literal
# length of a sequence (e.g. tuple or list), so there's no point in adding
# a "shape" type parameter here
_V = TypeVar('_V')
_NestedSequence: TypeAlias = CanSequence[int, '_V | _NestedSequence[_V]']

_SV = TypeVar('_SV', bound=bool | int | float | complex | str | bytes)

# `SomeScalar` has 2 bounded type params:
# - scalar type (numpy), `: np.generic`
# - scalar type (python), `: bool | int | float | complex | str | bytes`
SomeScalar: TypeAlias = CanArray[ShapeScalar, _S] | _SV
# `SomeArray` is a scalar or a tensor, and has 3 bounded type params:
# - shape, `: tuple[int, ...]`
# - scalar type (numpy), `: np.generic`
# - scalar type (python), `: bool | int | float | complex | str | bytes`
SomeArray: TypeAlias = (
    CanArray[_ND, _S]
    | _NestedSequence[CanArray[ShapeArray, _S]]
    | _SV
    | _NestedSequence[_SV]
)

# `SomeTensor` must be at least 1-D, and has 3 bounded type params:
# - shape, `: tuple[int, *tuple[int, ...]]`
# - scalar type (numpy), `: np.generic`
# - scalar type (python), `: bool | int | float | complex | str | bytes`
SomeTensor: TypeAlias = (
    CanArray[_ND1, _S]
    | _NestedSequence[CanArray[ShapeArray, _S]]
    | _NestedSequence[_SV]
)
