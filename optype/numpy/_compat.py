from typing import TypeAlias

import numpy as np
from numpy_typing_compat import NUMPY_GE_2_0

__all__ = ["Bool", "Long", "ULong"]


# mypy: disable-error-code="no-redef"
# pyright: reportRedeclaration=false

if NUMPY_GE_2_0:
    Bool: TypeAlias = np.bool
    ULong: TypeAlias = np.ulong
    Long: TypeAlias = np.long
else:
    Bool: TypeAlias = np.bool_  # type: ignore[misc]
    ULong: TypeAlias = np.uint  # type: ignore[misc]
    Long: TypeAlias = np.int_  # type: ignore[misc]
