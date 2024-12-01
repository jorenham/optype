# mypy: disable-error-code="no-any-explicit"
import sys
from collections.abc import Sequence as Seq
from typing import Literal, TypeAlias, TypeVar

import numpy as np

import optype.typing as opt

from ._array import CanArray0D, CanArray1D, CanArray2D, CanArray3D, CanArrayND
from ._scalar import floating, integer, number
from ._sequence_nd import SequenceND as SeqND


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


V = TypeVar("V")
S = TypeVar("S", bound=np.generic)

_To0D: TypeAlias = V | S | CanArray0D[S]

_To1D = TypeAliasType("_To1D", CanArrayND[S] | Seq[_To0D[V, S]], type_params=(V, S))
_To2D = TypeAliasType("_To2D", CanArrayND[S] | Seq[_To1D[V, S]], type_params=(V, S))
_To3D = TypeAliasType("_To3D", CanArrayND[S] | Seq[_To2D[V, S]], type_params=(V, S))
_ToND = TypeAliasType(
    "_ToND",
    CanArrayND[S] | SeqND[CanArrayND[S]] | SeqND[_To0D[V, S]],
    type_params=(V, S),
)

_ToStrict1D = TypeAliasType(
    "_ToStrict1D",
    CanArray1D[S] | Seq[_To0D[V, S]],
    type_params=(V, S),
)
_ToStrict2D = TypeAliasType(
    "_ToStrict2D",
    CanArray2D[S] | Seq[_ToStrict1D[V, S]],
    type_params=(V, S),
)
_ToStrict3D = TypeAliasType(
    "_ToStrict3D",
    CanArray3D[S] | Seq[_ToStrict2D[V, S]],
    type_params=(V, S),
)


_PyBool: TypeAlias = bool | Literal[0, 1]  # 0 and 1 are sometimes used as bool values
_PyScalar: TypeAlias = complex | bytes | str  # `complex` equivs `complex | float | int`

_Int = TypeAliasType("_Int", integer | np.bool_)
_Float = TypeAliasType("_Float", floating | integer | np.bool_)
_Complex = TypeAliasType("_Complex", number | np.bool_)

ToScalar: TypeAlias = _PyScalar | np.generic
ToArray1D: TypeAlias = _To1D[_PyScalar, np.generic]
ToArray2D: TypeAlias = _To2D[_PyScalar, np.generic]
ToArray3D: TypeAlias = _To3D[_PyScalar, np.generic]
ToArrayND: TypeAlias = _ToND[_PyScalar, np.generic]
ToArrayStrict1D: TypeAlias = _ToStrict1D[_PyScalar, np.generic]
ToArrayStrict2D: TypeAlias = _ToStrict2D[_PyScalar, np.generic]
ToArrayStrict3D: TypeAlias = _ToStrict3D[_PyScalar, np.generic]

ToBool: TypeAlias = _PyBool | np.bool_
ToBool1D: TypeAlias = _To1D[_PyBool, np.bool_]
ToBool2D: TypeAlias = _To2D[_PyBool, np.bool_]
ToBool3D: TypeAlias = _To3D[_PyBool, np.bool_]
ToBoolND: TypeAlias = _ToND[_PyBool, np.bool_]
ToBoolStrict1D: TypeAlias = _ToStrict1D[_PyBool, np.bool_]
ToBoolStrict2D: TypeAlias = _ToStrict2D[_PyBool, np.bool_]
ToBoolStrict3D: TypeAlias = _ToStrict3D[_PyBool, np.bool_]

ToJustInt: TypeAlias = opt.JustInt | integer
ToJustInt1D: TypeAlias = _To1D[opt.JustInt, integer]
ToJustInt2D: TypeAlias = _To2D[opt.JustInt, integer]
ToJustInt3D: TypeAlias = _To3D[opt.JustInt, integer]
ToJustIntND: TypeAlias = _ToND[opt.JustInt, integer]
ToJustIntStrict1D: TypeAlias = _ToStrict1D[opt.JustInt, integer]
ToJustIntStrict3D: TypeAlias = _ToStrict3D[opt.JustInt, integer]
ToJustIntStrict2D: TypeAlias = _ToStrict2D[opt.JustInt, integer]

ToInt: TypeAlias = int | _Int
ToInt1D: TypeAlias = _To1D[int, _Int]
ToInt2D: TypeAlias = _To2D[int, _Int]
ToInt3D: TypeAlias = _To3D[int, _Int]
ToIntND: TypeAlias = _ToND[int, _Int]
ToIntStrict1D: TypeAlias = _ToStrict1D[int, _Int]
ToIntStrict2D: TypeAlias = _ToStrict2D[int, _Int]
ToIntStrict3D: TypeAlias = _ToStrict3D[int, _Int]

ToFloat: TypeAlias = float | _Float
ToFloat1D: TypeAlias = _To1D[float, _Float]
ToFloat2D: TypeAlias = _To2D[float, _Float]
ToFloat3D: TypeAlias = _To3D[float, _Float]
ToFloatND: TypeAlias = _ToND[float, _Float]
ToFloatStrict1D: TypeAlias = _ToStrict1D[float, _Float]
ToFloatStrict2D: TypeAlias = _ToStrict2D[float, _Float]
ToFloatStrict3D: TypeAlias = _ToStrict3D[float, _Float]

ToComplex: TypeAlias = complex | _Complex
ToComplex1D: TypeAlias = _To1D[complex, _Complex]
ToComplex2D: TypeAlias = _To2D[complex, _Complex]
ToComplex3D: TypeAlias = _To3D[complex, _Complex]
ToComplexND: TypeAlias = _ToND[complex, _Complex]
ToComplexStrict1D: TypeAlias = _ToStrict1D[complex, _Complex]
ToComplexStrict2D: TypeAlias = _ToStrict2D[complex, _Complex]
ToComplexStrict3D: TypeAlias = _ToStrict3D[complex, _Complex]
