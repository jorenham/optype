"""
Compatibility with older numpy versions.
"""

from ._compat import NP2, Long as long, ULong as ulong  # noqa: N813
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

if NP2:
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

    try:
        from numpy import DTypePromotionError
    except ImportError:
        # numpy<1.25
        class DTypePromotionError(TypeError): ...
