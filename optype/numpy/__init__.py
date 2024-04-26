__all__ = (
    'HasDType',
    'SomeDType',
    'dtype',
)

from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np

from . import dtype


_T = TypeVar('_T', bound=np.generic)
_T_co = TypeVar('_T_co', bound=np.generic, covariant=True)


@runtime_checkable
class HasDType(Protocol[_T_co]):
    @property
    def dtype(self) -> np.dtype[_T_co]: ...


# subset of `npt.DTypeLike`, with type parameter `T: np.generic = np.generic`
# useful for overloaded methods with a `dtype` parameter
SomeDType: TypeAlias = np.dtype[_T] | HasDType[_T] | _T
