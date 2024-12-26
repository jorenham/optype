"""
Compatibility with older numpy versions.
"""

from ._compat import NP125, Long as long, ULong as ulong  # noqa: N813
from ._scalar import (
    cfloating as complexfloating,
    floating,
    inexact,
    integer,
    number,
    sinteger as signedinteger,
    uinteger as unsignedinteger,
)


__all__ = [
    "AxisError",
    "ComplexWarning",
    "DTypePromotionError",
    "ModuleDeprecationWarning",
    "TooHardError",
    "VisibleDeprecationWarning",
    "complexfloating",
    "floating",
    "inexact",
    "integer",
    "long",
    "number",
    "signedinteger",
    "ulong",
    "unsignedinteger",
]

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
