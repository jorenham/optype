import sys
from typing import Any, TypeAlias

import numpy as np


if sys.version_info >= (3, 13):
    from typing import Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import Protocol, TypeVar, runtime_checkable


__all__ = 'DType', 'HasDType'


_ST = TypeVar('_ST', bound=np.generic, default=Any)
DType: TypeAlias = np.dtype[_ST]
"""Alias for `numpy.dtype[T: numpy.generic = Any]`."""


_DT = TypeVar(
    '_DT',
    infer_variance=True,
    bound=np.dtype[Any],
    default=np.dtype[Any],
)


@runtime_checkable
class HasDType(Protocol[_DT]):
    """HasDType[DT: np.dtype[Any] = Any]

    Runtime checkable protocol for objects (or types) that have a `dtype`
    attribute (or property), such as `numpy.ndarray` instances, or
    `numpy.generic` "scalar" instances.

    Anything that implements this interface can be used with the `numpy.dtype`
    constructor, i.e. its constructor is compatible with a signature that
    looks something like `(HasDType[DT: numpy.DType], ...) -> DT`.
    """
    @property
    def dtype(self, /) -> _DT: ...
