"""The render backends: the `Backend` interface and its implementations."""

from typing import Final, Literal

from ._base import Backend
from ._compat import COMPAT
from ._terse import TERSE

type BackendName = Literal["terse", "compat"]

BACKENDS: Final[dict[BackendName, Backend]] = {"terse": TERSE, "compat": COMPAT}
