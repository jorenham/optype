from __future__ import annotations

import sys
from collections.abc import Sequence as Seq
from typing import TypeAlias, TypeVar

import numpy as np

from ._array import CanArrayND
from ._scalar import floating, integer, number
from ._sequence_nd import SequenceND


if sys.version_info >= (3, 13):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


__all__ = [
    "ToArray1D",
    "ToArray2D",
    "ToArray3D",
    "ToArrayND",
    "ToBool",
    "ToBool1D",
    "ToBool2D",
    "ToBool3D",
    "ToBoolND",
    "ToComplex",
    "ToComplex1D",
    "ToComplex2D",
    "ToComplex3D",
    "ToComplexND",
    "ToFloat",
    "ToFloat1D",
    "ToFloat2D",
    "ToFloat3D",
    "ToFloatND",
    "ToInt",
    "ToInt1D",
    "ToInt2D",
    "ToInt3D",
    "ToIntND",
    "ToScalar",
]

T = TypeVar("T")
ST = TypeVar("ST", bound=np.generic)


_To1D: TypeAlias = CanArrayND[ST] | Seq[ST | T]
_To2D: TypeAlias = CanArrayND[ST] | Seq[CanArrayND[ST]] | Seq[Seq[ST | T]]
_To3D: TypeAlias = (
    CanArrayND[ST]
    | Seq[CanArrayND[ST]]
    | Seq[Seq[CanArrayND[ST]]]
    | Seq[Seq[Seq[ST | T]]]
)
# recursive sequence type cannot be used due to a mypy bug:
# https://github.com/python/mypy/issues/18184
_ToND: TypeAlias = CanArrayND[ST] | SequenceND[T | ST] | SequenceND[CanArrayND[ST]]

ToScalar: TypeAlias = complex | bytes | str | np.generic
ToArray1D: TypeAlias = _To1D[np.generic, complex | bytes | str]
ToArray2D: TypeAlias = _To2D[np.generic, complex | bytes | str]
ToArray3D: TypeAlias = _To3D[np.generic, complex | bytes | str]
ToArrayND: TypeAlias = _ToND[np.generic, complex | bytes | str]

ToBool: TypeAlias = bool | np.bool_
ToBool1D: TypeAlias = _To1D[np.bool_, bool]
ToBool2D: TypeAlias = _To2D[np.bool_, bool]
ToBool3D: TypeAlias = _To3D[np.bool_, bool]
ToBoolND: TypeAlias = _ToND[np.bool_, bool]

integer_co = TypeAliasType("integer_co", integer | np.bool_)  # type: ignore[no-any-explicit]
ToInt: TypeAlias = int | integer_co
ToInt1D: TypeAlias = _To1D[integer_co, int]
ToInt2D: TypeAlias = _To2D[integer_co, int]
ToInt3D: TypeAlias = _To3D[integer_co, int]
ToIntND: TypeAlias = _ToND[integer_co, int]

floating_co = TypeAliasType("floating_co", floating | integer | np.bool_)  # type: ignore[no-any-explicit]
ToFloat: TypeAlias = float | floating_co
ToFloat1D: TypeAlias = _To1D[floating_co, float]
ToFloat2D: TypeAlias = _To2D[floating_co, float]
ToFloat3D: TypeAlias = _To3D[floating_co, float]
ToFloatND: TypeAlias = _ToND[floating_co, float]

complexfloating_co = TypeAliasType("complexfloating_co", number | np.bool_)  # type: ignore[no-any-explicit]
ToComplex: TypeAlias = complex | complexfloating_co
ToComplex1D: TypeAlias = _To1D[complexfloating_co, complex]
ToComplex2D: TypeAlias = _To2D[complexfloating_co, complex]
ToComplex3D: TypeAlias = _To3D[complexfloating_co, complex]
ToComplexND: TypeAlias = _ToND[complexfloating_co, complex]
