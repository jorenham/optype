from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np


__all__ = ['Bool', 'Long', 'ULong']


_NP_V2: Final[bool] = np.__version__.startswith('2.')


if _NP_V2:
    Bool: TypeAlias = np.bool
    ULong: TypeAlias = np.ulong
    Long: TypeAlias = np.long
else:
    Bool: TypeAlias = np.bool_
    ULong: TypeAlias = np.uint
    Long: TypeAlias = np.int_
