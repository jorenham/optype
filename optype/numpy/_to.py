# mypy: disable-error-code="no-any-explicit"
import sys
from collections.abc import Sequence as Seq
from typing import Literal, TypeAlias, TypeVar

import numpy as np

import optype.typing as opt

from ._array import CanArray1D, CanArray2D, CanArray3D, CanArrayND
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
    "ToArrayStrict1D",
    "ToArrayStrict2D",
    "ToArrayStrict3D",
    "ToBool",
    "ToBool1D",
    "ToBool2D",
    "ToBool3D",
    "ToBoolND",
    "ToBoolStrict1D",
    "ToBoolStrict2D",
    "ToBoolStrict3D",
    "ToComplex",
    "ToComplex1D",
    "ToComplex2D",
    "ToComplex3D",
    "ToComplexND",
    "ToComplexStrict1D",
    "ToComplexStrict2D",
    "ToComplexStrict3D",
    "ToFloat",
    "ToFloat1D",
    "ToFloat2D",
    "ToFloat3D",
    "ToFloatND",
    "ToFloatStrict1D",
    "ToFloatStrict2D",
    "ToFloatStrict3D",
    "ToInt",
    "ToInt1D",
    "ToInt2D",
    "ToInt3D",
    "ToIntND",
    "ToIntStrict1D",
    "ToIntStrict2D",
    "ToIntStrict3D",
    "ToJustInt",
    "ToJustInt1D",
    "ToJustInt2D",
    "ToJustInt3D",
    "ToJustIntND",
    "ToJustIntStrict1D",
    "ToJustIntStrict2D",
    "ToJustIntStrict3D",
    "ToScalar",
]  # fmt: skip

ST = TypeVar("ST", bound=np.generic)
T = TypeVar("T")

_To1D = TypeAliasType(
    "_To1D",
    CanArrayND[ST] | Seq[ST | T],
    type_params=(ST, T),
)
_ToStrict1D = TypeAliasType(
    "_ToStrict1D",
    CanArray1D[ST] | Seq[ST | T],
    type_params=(ST, T),
)

_To2D = TypeAliasType(
    "_To2D",
    CanArrayND[ST] | Seq[CanArrayND[ST]] | Seq[Seq[ST | T]],
    type_params=(ST, T),
)
_ToStrict2D = TypeAliasType(
    "_ToStrict2D",
    CanArray2D[ST] | Seq[CanArray1D[ST]] | Seq[Seq[ST | T]],
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
_ToStrict3D = TypeAliasType(
    "_ToStrict3D",
    (
        CanArray3D[ST]
        | Seq[CanArray2D[ST]]
        | Seq[Seq[CanArray1D[ST]]]
        | Seq[Seq[Seq[ST | T]]]
    ),
    type_params=(ST, T),
)

# recursive sequence type cannot be used due to a mypy bug:
# https://github.com/python/mypy/issues/18184
_ToND: TypeAlias = CanArrayND[ST] | SequenceND[T | ST] | SequenceND[CanArrayND[ST]]

_PyBool: TypeAlias = bool | Literal[0, 1]  # 0 and 1 are sometimes used as bool values
_PyScalar: TypeAlias = complex | bytes | str  # `complex` equivs `complex | float | int`

_Int = TypeAliasType("_Int", integer | np.bool_)
_Float = TypeAliasType("_Float", floating | integer | np.bool_)
_Complex = TypeAliasType("_Complex", number | np.bool_)

ToScalar: TypeAlias = np.generic | _PyScalar
ToArray1D: TypeAlias = _To1D[np.generic, _PyScalar]
ToArrayStrict1D: TypeAlias = _ToStrict1D[np.generic, _PyScalar]
ToArray2D: TypeAlias = _To2D[np.generic, _PyScalar]
ToArrayStrict2D: TypeAlias = _ToStrict2D[np.generic, _PyScalar]
ToArray3D: TypeAlias = _To3D[np.generic, _PyScalar]
ToArrayStrict3D: TypeAlias = _ToStrict3D[np.generic, _PyScalar]
ToArrayND: TypeAlias = _ToND[np.generic, _PyScalar]

ToBool: TypeAlias = np.bool_ | _PyBool
ToBool1D: TypeAlias = _To1D[np.bool_, _PyBool]
ToBoolStrict1D: TypeAlias = _ToStrict1D[np.bool_, _PyBool]
ToBool2D: TypeAlias = _To2D[np.bool_, _PyBool]
ToBoolStrict2D: TypeAlias = _ToStrict2D[np.bool_, _PyBool]
ToBool3D: TypeAlias = _To3D[np.bool_, _PyBool]
ToBoolStrict3D: TypeAlias = _ToStrict3D[np.bool_, _PyBool]
ToBoolND: TypeAlias = _ToND[np.bool_, _PyBool]

ToJustInt: TypeAlias = integer | opt.JustInt
ToJustInt1D: TypeAlias = _To1D[integer, opt.JustInt]
ToJustIntStrict1D: TypeAlias = _ToStrict1D[integer, opt.JustInt]
ToJustInt2D: TypeAlias = _To2D[integer, opt.JustInt]
ToJustIntStrict2D: TypeAlias = _ToStrict2D[integer, opt.JustInt]
ToJustInt3D: TypeAlias = _To3D[integer, opt.JustInt]
ToJustIntStrict3D: TypeAlias = _ToStrict3D[integer, opt.JustInt]
ToJustIntND: TypeAlias = _ToND[integer, opt.JustInt]

ToInt: TypeAlias = _Int | int
ToInt1D: TypeAlias = _To1D[_Int, int]
ToIntStrict1D: TypeAlias = _ToStrict1D[_Int, int]
ToInt2D: TypeAlias = _To2D[_Int, int]
ToIntStrict2D: TypeAlias = _ToStrict2D[_Int, int]
ToInt3D: TypeAlias = _To3D[_Int, int]
ToIntStrict3D: TypeAlias = _ToStrict3D[_Int, int]
ToIntND: TypeAlias = _ToND[_Int, int]

ToFloat: TypeAlias = _Float | float
ToFloat1D: TypeAlias = _To1D[_Float, float]
ToFloatStrict1D: TypeAlias = _ToStrict1D[_Float, float]
ToFloat2D: TypeAlias = _To2D[_Float, float]
ToFloatStrict2D: TypeAlias = _ToStrict2D[_Float, float]
ToFloat3D: TypeAlias = _To3D[_Float, float]
ToFloatStrict3D: TypeAlias = _ToStrict3D[_Float, float]
ToFloatND: TypeAlias = _ToND[_Float, float]

ToComplex: TypeAlias = _Complex | complex
ToComplex1D: TypeAlias = _To1D[_Complex, complex]
ToComplexStrict1D: TypeAlias = _ToStrict1D[_Complex, complex]
ToComplex2D: TypeAlias = _To2D[_Complex, complex]
ToComplexStrict2D: TypeAlias = _ToStrict2D[_Complex, complex]
ToComplex3D: TypeAlias = _To3D[_Complex, complex]
ToComplexStrict3D: TypeAlias = _ToStrict3D[_Complex, complex]
ToComplexND: TypeAlias = _ToND[_Complex, complex]
