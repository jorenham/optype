from __future__ import annotations

import sys
from typing import Any, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import TypeVar, Unpack
else:
    from typing_extensions import TypeVar, Unpack


__all__ = (
    'Array',
    'DType',
    'UniTuple0',
    'UniTuple1',
    'UniTuple2',
)


_T_UniTuple = TypeVar('_T_UniTuple', bound=object)

UniTuple0: TypeAlias = tuple[_T_UniTuple, ...]
"""Tuple with 0 or more elements of the same type."""

UniTuple1: TypeAlias = tuple[
    _T_UniTuple,
    Unpack[tuple[_T_UniTuple, ...]],
]
"""Tuple with at least 1 element of the same type."""

UniTuple2: TypeAlias = tuple[
    _T_UniTuple,
    Unpack[tuple[
        _T_UniTuple,
        Unpack[tuple[_T_UniTuple, ...]],
    ]],
]
"""Tuple with at least 2 elements of the same type."""

_S_Array = TypeVar('_S_Array', bound=UniTuple0[int], default=UniTuple0[int])
_T_Array = TypeVar('_T_Array', bound=np.generic, default=Any)
Array: TypeAlias = np.ndarray[_S_Array, np.dtype[_T_Array]]
"""NumPy array with optional type params for shape and generic dtype."""


_T_DType = TypeVar('_T_DType', bound=np.generic, default=Any)
DType: TypeAlias = np.dtype[_T_DType]
"""Alias for `numpy.dtype[T: numpy.generic = Any]`."""
