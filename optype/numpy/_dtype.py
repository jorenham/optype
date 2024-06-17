import sys
from typing import Any, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import Protocol, TypeVar, runtime_checkable


__all__ = 'ArgDType', 'DType', 'HasDType'


_T_DType = TypeVar('_T_DType', bound=np.generic, default=Any)
DType: TypeAlias = np.dtype[_T_DType]
"""Alias for `numpy.dtype[T: numpy.generic = Any]`."""


_T_HasDType = TypeVar(
    '_T_HasDType',
    infer_variance=True,
    bound=np.dtype[Any],
    default=Any,
)


@runtime_checkable
class HasDType(Protocol[_T_HasDType]):
    """
    `HasDType[DT: np.dtype[Any] = Any]`

    Runtime checkable protocol for objects (or types) that have a `dtype`
    attribute (or property), such as `numpy.ndarray` instances, or
    `numpy.generic` "scalar" instances.

    Anything that implements this interface can be used with the `numpy.dtype`
    constructor, i.e. its constructor is compatible with a signature that
    looks something like `(HasDType[DT: numpy.DType], ...) -> DT`.
    """
    @property
    def dtype(self, /) -> _T_HasDType: ...


_T_DType = TypeVar('_T_DType', bound=np.generic, default=Any)
ArgDType: TypeAlias = (
    np.dtype[_T_DType]
    | type[_T_DType]
    | HasDType[np.dtype[_T_DType]]
)
"""
Subset of `npt.DTypeLike`, with optional type parameter, bound to `np.generic`.
Useful for overloading a `dtype` parameter.
"""
