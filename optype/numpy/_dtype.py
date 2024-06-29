import sys
from typing import Any, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import Protocol, TypeVar, runtime_checkable


__all__ = 'AnyDType', 'DType', 'HasDType'


_ST_DType = TypeVar('_ST_DType', bound=np.generic, default=np.generic)
DType: TypeAlias = np.dtype[_ST_DType]
"""Alias for `numpy.dtype[T: numpy.generic = Any]`."""


_DT_HasDType = TypeVar(
    '_DT_HasDType',
    infer_variance=True,
    bound=np.dtype[Any],
    default=np.dtype[Any],
)


@runtime_checkable
class HasDType(Protocol[_DT_HasDType]):
    """HasDType[DT: np.dtype[Any] = Any]

    Runtime checkable protocol for objects (or types) that have a `dtype`
    attribute (or property), such as `numpy.ndarray` instances, or
    `numpy.generic` "scalar" instances.

    Anything that implements this interface can be used with the `numpy.dtype`
    constructor, i.e. its constructor is compatible with a signature that
    looks something like `(HasDType[DT: numpy.DType], ...) -> DT`.
    """
    @property
    def dtype(self, /) -> _DT_HasDType: ...


_T_AnyDType = TypeVar('_T_AnyDType', bound=np.generic, default=np.generic)
AnyDType: TypeAlias = (
    type[_T_AnyDType]
    | np.dtype[_T_AnyDType]
    | HasDType[np.dtype[_T_AnyDType]]
)
