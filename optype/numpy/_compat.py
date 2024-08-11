from __future__ import annotations

import sys
from typing import Final, TypeAlias

import numpy as np


__all__ = [
    'NP2',
    'NP20',
    'NP21',
    'Bool',
    'Long',
    'StringDType',
    'ULong',
]


NP2: Final[bool] = np.__version__.startswith('2.')
NP20: Final[bool] = np.__version__.startswith('2.0')
NP21: Final[bool] = np.__version__.startswith('2.1')


if NP2:
    Bool: TypeAlias = np.bool
    ULong: TypeAlias = np.ulong
    Long: TypeAlias = np.long
else:
    Bool: TypeAlias = np.bool_
    ULong: TypeAlias = np.uint
    Long: TypeAlias = np.int_

if NP21:
    StringDType: TypeAlias = np.dtypes.StringDType
elif NP2:
    StringDType: TypeAlias = np.dtype[str]  # type: ignore[type-var]  # pyright: ignore[reportInvalidTypeArguments]
elif sys.version_info >= (3, 13):
    from typing import Never as StringDType
else:
    from typing_extensions import Never as StringDType
