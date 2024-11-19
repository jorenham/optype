import sys
from collections.abc import Sequence
from typing import TypeAlias, TypeVar

import numpy as np

from ._array import CanArrayND
from ._scalar import floating, integer, number


if sys.version_info >= (3, 13):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


__all__ = [
    "ToArray1D",
    "ToArray2D",
    "ToArrayND",
    "ToBool",
    "ToBool1D",
    "ToBool2D",
    "ToBoolND",
    "ToComplex",
    "ToComplex1D",
    "ToComplex2D",
    "ToComplexND",
    "ToFloat",
    "ToFloat1D",
    "ToFloat2D",
    "ToFloatND",
    "ToInt",
    "ToInt1D",
    "ToInt2D",
    "ToIntND",
    "ToScalar",
]

T = TypeVar("T")
ST = TypeVar("ST", bound=np.generic)

_To1D: TypeAlias = CanArrayND[ST] | Sequence[T | ST]
_To2D: TypeAlias = CanArrayND[ST] | Sequence[_To1D[ST, T]]

ToScalar: TypeAlias = complex | bytes | str | np.generic
ToArray1D: TypeAlias = _To1D[np.generic, complex | bytes | str]
ToArray2D: TypeAlias = _To2D[np.generic, complex | bytes | str]
ToArrayND: TypeAlias = _To1D[np.generic, complex | bytes | str] | Sequence["ToArrayND"]

ToBool: TypeAlias = bool | np.bool_
ToBool1D: TypeAlias = _To1D[np.bool_, bool]
ToBool2D: TypeAlias = _To2D[np.bool_, bool]
ToBoolND: TypeAlias = _To1D[np.bool_, bool] | Sequence["ToBoolND"]

_i_co = TypeAliasType("_i_co", integer | np.bool_)  # type: ignore[no-any-explicit]
ToInt: TypeAlias = int | _i_co
ToInt1D: TypeAlias = _To1D[_i_co, int]
ToInt2D: TypeAlias = _To2D[_i_co, int]
ToIntND: TypeAlias = _To1D[_i_co, int] | Sequence["ToIntND"]

_f_co = TypeAliasType("_f_co", floating | integer | np.bool_)  # type: ignore[no-any-explicit]
ToFloat: TypeAlias = float | _f_co
ToFloat1D: TypeAlias = _To1D[_f_co, float]
ToFloat2D: TypeAlias = _To2D[_f_co, float]
ToFloatND: TypeAlias = _To1D[_f_co, float] | Sequence["ToFloatND"]

_c_co = TypeAliasType("_c_co", number | np.bool_)  # type: ignore[no-any-explicit]
ToComplex: TypeAlias = complex | _c_co
ToComplex1D: TypeAlias = _To1D[_c_co, complex]
ToComplex2D: TypeAlias = _To2D[_c_co, complex]
ToComplexND: TypeAlias = _To1D[_c_co, complex] | Sequence["ToComplexND"]
