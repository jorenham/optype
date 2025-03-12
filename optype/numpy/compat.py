"""Compatibility with older numpy versions."""

from ._compat import NP125, Long as long, ULong as ulong  # noqa: N813
from ._scalar import (
    cfloating as complexfloating,
    cfloating32 as complexfloating64,
    cfloating64 as complexfloating128,
    floating,
    floating16,
    floating32,
    floating64,
    inexact,
    inexact32,
    inexact64,
    integer,
    integer8,
    integer16,
    integer32,
    integer64,
    number,
    number8,
    number16,
    number32,
    number64,
    sinteger as signedinteger,
    uinteger as unsignedinteger,
)

if NP125:
    from numpy.exceptions import (
        AxisError,
        ComplexWarning,
        DTypePromotionError,
        ModuleDeprecationWarning,
        TooHardError,
        VisibleDeprecationWarning,
    )
else:
    from numpy import (
        AxisError,
        ComplexWarning,
        ModuleDeprecationWarning,
        TooHardError,
        VisibleDeprecationWarning,
    )

    class DTypePromotionError(TypeError): ...


__all__ = [
    "AxisError",
    "ComplexWarning",
    "DTypePromotionError",
    "ModuleDeprecationWarning",
    "TooHardError",
    "VisibleDeprecationWarning",
    "complexfloating",
    "complexfloating64",
    "complexfloating128",
    "floating",
    "floating16",
    "floating32",
    "floating64",
    "inexact",
    "inexact32",
    "inexact64",
    "integer",
    "integer8",
    "integer16",
    "integer32",
    "integer64",
    "long",
    "number",
    "number8",
    "number16",
    "number32",
    "number64",
    "signedinteger",
    "ulong",
    "unsignedinteger",
]
