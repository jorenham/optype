# ruff: noqa: F403
from . import (
    _any_array,
    _any_dtype,
    _array,
    _dtype,
    _scalar,
    _sequence_nd,
    _shape,
    _to,
    _ufunc,
    compat,
    ctypeslib,
)
from ._any_array import *
from ._any_dtype import *
from ._array import *
from ._dtype import *
from ._scalar import *
from ._sequence_nd import *
from ._shape import *
from ._to import *
from ._ufunc import *


__all__ = ["compat", "ctypeslib"]
__all__ += _any_array.__all__
__all__ += _any_dtype.__all__
__all__ += _array.__all__
__all__ += _dtype.__all__
__all__ += _scalar.__all__
__all__ += _sequence_nd.__all__
__all__ += _shape.__all__
__all__ += _to.__all__
__all__ += _ufunc.__all__


def __dir__() -> list[str]:
    return __all__
