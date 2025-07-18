from typing import Final, TypeAlias

import numpy as np

__all__ = ["Bool", "Long", "ULong"]

# >=1.25
NP125: Final = True
# >=2.0
NP20: Final = np.__version__.startswith("2.")
# >=2.1
NP21: Final = NP20 and not np.__version__.startswith("2.0.")
# >=2.2
NP22: Final = NP21 and not np.__version__.startswith("2.1.")
# >=2.3
NP23: Final = NP22 and not np.__version__.startswith("2.2.")
# >=2.4
NP24: Final = NP23 and not np.__version__.startswith("2.3.")


if NP20:
    Bool: TypeAlias = np.bool
    ULong: TypeAlias = np.ulong
    Long: TypeAlias = np.long
else:
    Bool: TypeAlias = np.bool_
    ULong: TypeAlias = np.uint
    Long: TypeAlias = np.int_
