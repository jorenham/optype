from typing import Final, TypeAlias

import numpy as np

__all__ = ["Bool", "Long", "ULong"]


# >=1.23
NP123: Final = True
# >=1.24
NP124: Final = NP123 and not np.__version__.startswith("1.23.")
# >=1.25
NP125: Final = NP124 and not np.__version__.startswith("1.24.")
# >=1.26
NP126: Final = NP125 and not np.__version__.startswith("1.25.")
# >=2.0
NP20: Final = NP126 and not np.__version__.startswith("1.26.")
# >=2.1
NP21: Final = NP20 and not np.__version__.startswith("2.0.")
# >=2.2
NP22: Final = NP21 and not np.__version__.startswith("2.1.")
# >=2.3
NP23: Final = NP22 and not np.__version__.startswith("2.2.")
# >=3.0
NP30: Final = False


if NP20:
    Bool: TypeAlias = np.bool
    ULong: TypeAlias = np.ulong
    Long: TypeAlias = np.long
else:
    Bool: TypeAlias = np.bool_
    ULong: TypeAlias = np.uint
    Long: TypeAlias = np.int_
