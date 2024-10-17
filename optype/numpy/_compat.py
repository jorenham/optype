from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np


__all__ = ["NP2", "NP20", "Bool", "Long", "ULong"]


# `numpy>=2.0`
NP2: Final[bool] = np.__version__.startswith("2.")
# `numpy>=2.0,<2.1`
NP20: Final[bool] = np.__version__.startswith("2.0")


if NP2:
    Bool: TypeAlias = np.bool
    ULong: TypeAlias = np.ulong
    Long: TypeAlias = np.long
else:
    Bool: TypeAlias = np.bool_
    ULong: TypeAlias = np.uint
    Long: TypeAlias = np.int_
