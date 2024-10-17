# pyright: reportPrivateUsage=false
import ctypes as ct
from ctypes import _CData as CType  # noqa: PLC2701
from typing import TypeAlias

from typing_extensions import TypeVar

__all__ = "CScalar", "CType"

_T = TypeVar("_T", default=object)
CScalar: TypeAlias = ct._SimpleCData[_T]  # noqa: SLF001
