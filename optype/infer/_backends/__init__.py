"""The render backends: the `Backend` interface and its implementations."""

from typing import Final, Literal

from ._base import Backend
from ._compat import COMPAT, CompatBackend
from ._terse import TERSE, TerseBackend

__all__ = (
    "BACKENDS",
    "COMPAT",
    "TERSE",
    "Backend",
    "BackendName",
    "CompatBackend",
    "TerseBackend",
)

type BackendName = Literal["terse", "compat"]

BACKENDS: Final[dict[BackendName, Backend]] = {"terse": TERSE, "compat": COMPAT}
