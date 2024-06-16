import sys
from typing import Any, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import Protocol, TypeVar, runtime_checkable


__all__ = 'ArgDType', 'HasDType'


_T_dtype = TypeVar(
    '_T_dtype',
    infer_variance=True,
    bound=np.dtype[Any],
    default=np.dtype[Any],
)


@runtime_checkable
class HasDType(Protocol[_T_dtype]):
    """
    `HasDType[T: np.dtype[Any] = np.dtype[Any]]`

    Interface for objects (or types) with a `dtype` attribute or property,
    e.g. a `numpy.ndarray` instance or an instance of a concrete subtype of
    `numpy.generic`.

    The generic type parameter is bound to `np.generic`, and is optional.
    """
    @property
    def dtype(self, /) -> _T_dtype: ...


_T_DType = TypeVar('_T_DType', bound=np.generic, default=Any)
ArgDType: TypeAlias = (
    np.dtype[_T_DType]
    | type[_T_DType]
    | HasDType[np.dtype[_T_DType]]
)
"""
Subset of `npt.DTypeLike`, with optional type parameter, bound to `np.generic`.
Useful for overloaded methods with a `dtype` parameter.
"""
