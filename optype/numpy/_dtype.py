import sys
from typing import Protocol, TypeAlias

if sys.version_info >= (3, 13):
    from typing import TypeVar, runtime_checkable
else:
    from typing_extensions import TypeVar, runtime_checkable

import numpy as np

from optype._utils import set_module

__all__ = ["DType", "HasDType"]


_ST = TypeVar("_ST", bound=np.generic, default=np.generic)
DType: TypeAlias = np.dtype[_ST]
"""Alias for `numpy.dtype[T: numpy.generic = np.generic]`."""


_DT_co = TypeVar("_DT_co", bound=DType, default=DType, covariant=True)


@runtime_checkable
@set_module("optype.numpy")
class HasDType(Protocol[_DT_co]):
    """HasDType[DT: np.dtype[np.generic] = np.dtype[np.generic]]

    Runtime checkable protocol for objects (or types) that have a `dtype`
    attribute (or property), such as `numpy.ndarray` instances, or
    `numpy.generic` "scalar" instances.

    Anything that implements this interface can be used with the `numpy.dtype`
    constructor, i.e. its constructor is compatible with a signature that
    looks something like `(HasDType[DT: numpy.DType], ...) -> DT`.
    """

    @property
    def dtype(self, /) -> _DT_co: ...
