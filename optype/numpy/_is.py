import sys
from typing import Any, TypeAlias

if sys.version_info >= (3, 13):
    from typing import TypeIs, TypeVar
else:
    from typing_extensions import TypeIs, TypeVar

import numpy as np

from ._array import Array0D, Array1D, Array2D, Array3D, ArrayND
from ._dtype import ToDType

__all__ = [
    "is_array_0d",
    "is_array_1d",
    "is_array_2d",
    "is_array_3d",
    "is_array_nd",
    "is_dtype",
    "is_sctype",
]


def __dir__() -> list[str]:
    return __all__


Shape: TypeAlias = tuple[int, ...]
ShapeT = TypeVar("ShapeT", bound=Shape, default=Shape)
DTypeT = TypeVar("DTypeT", bound=np.dtype[Any])
ScalarT = TypeVar("ScalarT", bound=np.generic, default=np.generic)


def is_dtype(
    x: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[np.dtype[ScalarT]]:
    return isinstance(x, np.dtype) and (dtype is None or np.issubdtype(x, dtype))


def is_sctype(
    x: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[type[ScalarT]]:
    return (
        isinstance(x, type)
        and issubclass(x, np.generic)
        and (dtype is None or np.issubdtype(x, dtype))
    )


def is_array_nd(
    a: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[ArrayND[ScalarT, ShapeT]]:
    """Checks if `a` is a `ndarray` of the given dtype (defaults to `generic`)."""
    return isinstance(a, np.ndarray) and (
        dtype is None or np.issubdtype(a.dtype, dtype)
    )


def is_array_0d(
    a: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[Array0D[ScalarT]]:
    """Checks if `a` is a 0-d `ndarray` of the given dtype (defaults to `generic`)."""
    return is_array_nd(a, dtype) and a.ndim == 0


def is_array_1d(
    a: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[Array1D[ScalarT]]:
    """Checks if `a` is a 1-d `ndarray` of the given dtype (defaults to `generic`)."""
    return is_array_nd(a, dtype) and a.ndim == 1


def is_array_2d(
    a: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[Array2D[ScalarT]]:
    """Checks if `a` is a 2-d `ndarray` of the given dtype (defaults to `generic`)."""
    return is_array_nd(a, dtype) and a.ndim == 2


def is_array_3d(
    a: object,
    /,
    dtype: ToDType[ScalarT] | None = None,
) -> TypeIs[Array3D[ScalarT]]:
    """Checks if `a` is a 3-d `ndarray` of the given dtype (defaults to `generic`)."""
    return is_array_nd(a, dtype) and a.ndim == 3
