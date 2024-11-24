# mypy: disable-error-code="no-any-explicit"
import sys
from collections.abc import Sequence as Seq
from typing import Literal, TypeAlias, TypeVar

import numpy as np

import optype.typing as opt

from ._array import CanArrayND
from ._scalar import floating, integer, number
from ._sequence_nd import SequenceND


if sys.version_info >= (3, 13):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


__all__ = [
    "ToArray1D", "ToArray2D", "ToArray3D", "ToArrayND",
    "ToBool", "ToBool1D", "ToBool2D", "ToBool3D", "ToBoolND",
    "ToComplex", "ToComplex1D", "ToComplex2D", "ToComplex3D", "ToComplexND",
    "ToFloat", "ToFloat1D", "ToFloat2D", "ToFloat3D", "ToFloatND",
    "ToInt", "ToInt1D", "ToInt2D", "ToInt3D", "ToIntND",
    "ToJustInt", "ToJustInt1D", "ToJustInt2D", "ToJustInt3D", "ToJustIntND",
    "ToScalar",
]  # fmt: skip

ST = TypeVar("ST", bound=np.generic)
T = TypeVar("T")

_To1D = TypeAliasType(
    "_To1D",
    CanArrayND[ST] | Seq[ST | T],
    type_params=(ST, T),
)
_To2D = TypeAliasType(
    "_To2D",
    CanArrayND[ST] | Seq[CanArrayND[ST]] | Seq[Seq[ST | T]],
    type_params=(ST, T),
)
_To3D = TypeAliasType(
    "_To3D",
    (
        CanArrayND[ST]
        | Seq[CanArrayND[ST]]
        | Seq[Seq[CanArrayND[ST]]]
        | Seq[Seq[Seq[ST | T]]]
    ),
    type_params=(ST, T),
)
# recursive sequence type cannot be used due to a mypy bug:
# https://github.com/python/mypy/issues/18184
_ToND: TypeAlias = CanArrayND[ST] | SequenceND[T | ST] | SequenceND[CanArrayND[ST]]

_PyBool: TypeAlias = bool | Literal[0, 1]  # 0 and 1 are sometimes used as bool values
_PyScalar: TypeAlias = complex | bytes | str  # `complex` somehow includes `float | int`

_Int = TypeAliasType("_Int", integer | np.bool_)
_Float = TypeAliasType("_Float", floating | integer | np.bool_)
_Complex = TypeAliasType("_Complex", number | np.bool_)

ToScalar: TypeAlias = np.generic | _PyScalar
ToArray1D: TypeAlias = _To1D[np.generic, _PyScalar]
ToArray2D: TypeAlias = _To2D[np.generic, _PyScalar]
ToArray3D: TypeAlias = _To3D[np.generic, _PyScalar]
ToArrayND: TypeAlias = _ToND[np.generic, _PyScalar]

ToBool: TypeAlias = np.bool_ | _PyBool
ToBool1D: TypeAlias = _To1D[np.bool_, _PyBool]
ToBool2D: TypeAlias = _To2D[np.bool_, _PyBool]
ToBool3D: TypeAlias = _To3D[np.bool_, _PyBool]
ToBoolND: TypeAlias = _ToND[np.bool_, _PyBool]

ToJustInt: TypeAlias = integer | opt.JustInt
ToJustInt1D: TypeAlias = _To1D[integer, opt.JustInt]
ToJustInt2D: TypeAlias = _To2D[integer, opt.JustInt]
ToJustInt3D: TypeAlias = _To3D[integer, opt.JustInt]
ToJustIntND: TypeAlias = _ToND[integer, opt.JustInt]

ToInt: TypeAlias = _Int | int
ToInt1D: TypeAlias = _To1D[_Int, int]
ToInt2D: TypeAlias = _To2D[_Int, int]
ToInt3D: TypeAlias = _To3D[_Int, int]
ToIntND: TypeAlias = _ToND[_Int, int]

ToFloat: TypeAlias = _Float | float
ToFloat1D: TypeAlias = _To1D[_Float, float]
ToFloat2D: TypeAlias = _To2D[_Float, float]
ToFloat3D: TypeAlias = _To3D[_Float, float]
ToFloatND: TypeAlias = _ToND[_Float, float]

ToComplex: TypeAlias = _Complex | complex
ToComplex1D: TypeAlias = _To1D[_Complex, complex]
ToComplex2D: TypeAlias = _To2D[_Complex, complex]
ToComplex3D: TypeAlias = _To3D[_Complex, complex]
ToComplexND: TypeAlias = _ToND[_Complex, complex]
