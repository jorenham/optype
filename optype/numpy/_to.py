import sys
from collections.abc import Sequence as Seq
from typing import Any, Literal, Protocol, TypeAlias

if sys.version_info >= (3, 13):
    from typing import TypeAliasType, TypeVar
else:
    from typing_extensions import TypeAliasType, TypeVar

import numpy as np

import optype.numpy.compat as npc
from ._array import CanArray0D, CanArray1D, CanArray2D, CanArray3D, CanArrayND
from ._compat import NP22
from ._sequence_nd import SequenceND as SeqND
from optype._core._just import JustComplex, JustFloat, JustInt

__all__ = [  # noqa: RUF022
    "ToScalar",
    "ToArray1D", "ToArrayStrict1D",
    "ToArray2D", "ToArrayStrict2D",
    "ToArray3D", "ToArrayStrict3D",
    "ToArrayND",

    "ToFalse", "ToTrue",
    "ToJustFalse", "ToJustTrue",

    "ToBool", "ToJustBool",
    "ToBool1D", "ToJustBool1D", "ToBoolStrict1D", "ToJustBoolStrict1D",
    "ToBool2D", "ToJustBool2D", "ToBoolStrict2D", "ToJustBoolStrict2D",
    "ToBool3D", "ToJustBool3D", "ToBoolStrict3D", "ToJustBoolStrict3D",
    "ToBoolND", "ToJustBoolND",

    "ToInt", "ToJustInt",
    "ToInt1D", "ToJustInt1D", "ToIntStrict1D", "ToJustIntStrict1D",
    "ToInt2D", "ToJustInt2D", "ToIntStrict2D", "ToJustIntStrict2D",
    "ToInt3D", "ToJustInt3D", "ToIntStrict3D", "ToJustIntStrict3D",
    "ToIntND", "ToJustIntND",

    "ToJustInt64",
    "ToJustInt64_1D", "ToJustInt64Strict1D",
    "ToJustInt64_2D", "ToJustInt64Strict2D",
    "ToJustInt64_3D", "ToJustInt64Strict3D",
    "ToJustInt64_ND",

    "ToFloat64", "ToJustFloat64",
    "ToFloat64_1D", "ToJustFloat64_1D", "ToFloat64Strict1D", "ToJustFloat64Strict1D",
    "ToFloat64_2D", "ToJustFloat64_2D", "ToFloat64Strict2D", "ToJustFloat64Strict2D",
    "ToFloat64_3D", "ToJustFloat64_3D", "ToFloat64Strict3D", "ToJustFloat64Strict3D",
    "ToFloat64_ND", "ToJustFloat64_ND",

    "ToFloat", "ToJustFloat",
    "ToFloat1D", "ToJustFloat1D", "ToFloatStrict1D", "ToJustFloatStrict1D",
    "ToFloat2D", "ToJustFloat2D", "ToFloatStrict2D", "ToJustFloatStrict2D",
    "ToFloat3D", "ToJustFloat3D", "ToFloatStrict3D", "ToJustFloatStrict3D",
    "ToFloatND", "ToJustFloatND",

    "ToComplex128", "ToJustComplex128",
    "ToComplex128_1D", "ToJustComplex128_1D",
    "ToComplex128_2D", "ToJustComplex128_2D",
    "ToComplex128_3D", "ToJustComplex128_3D",
    "ToComplex128_ND", "ToJustComplex128_ND",
    "ToComplex128Strict1D", "ToJustComplex128Strict1D",
    "ToComplex128Strict2D", "ToJustComplex128Strict2D",
    "ToComplex128Strict3D", "ToJustComplex128Strict3D",

    "ToComplex", "ToJustComplex",
    "ToComplex1D", "ToJustComplex1D", "ToComplexStrict1D", "ToJustComplexStrict1D",
    "ToComplex2D", "ToJustComplex2D", "ToComplexStrict2D", "ToJustComplexStrict2D",
    "ToComplex3D", "ToJustComplex3D", "ToComplexStrict3D", "ToJustComplexStrict3D",
    "ToComplexND", "ToJustComplexND",
]  # fmt: skip


def __dir__() -> list[str]:
    return __all__


###


_PyBool: TypeAlias = bool | Literal[0, 1]  # 0 and 1 are sometimes used as bool values
_PyScalar: TypeAlias = complex | bytes | str  # `complex` equivs `complex | float | int`

T = TypeVar("T", default=_PyScalar)
SCT = TypeVar("SCT", bound=np.generic, default=Any)
SCT0 = TypeVar("SCT0", bound=np.generic)
SCT_co = TypeVar("SCT_co", bound=np.generic, covariant=True)


class _CanArray(Protocol[SCT_co]):
    def __array__(self, /) -> np.ndarray[tuple[Any, ...], np.dtype[SCT_co]]: ...


_To0D = TypeAliasType("_To0D", T | CanArray0D[SCT0], type_params=(SCT0, T))
_To1D = TypeAliasType(
    "_To1D",
    CanArrayND[SCT0] | Seq[T | CanArray0D[SCT0]],
    type_params=(SCT0, T),
)
_To2D = TypeAliasType(
    "_To2D",
    CanArrayND[SCT0] | Seq[_To1D[SCT0, T]],
    type_params=(SCT0, T),
)
_To3D = TypeAliasType(
    "_To3D",
    CanArrayND[SCT0] | Seq[_To2D[SCT0, T]],
    type_params=(SCT0, T),
)
_ToND = TypeAliasType(
    "_ToND",
    CanArrayND[SCT0] | SeqND[T | _CanArray[SCT0]],
    type_params=(SCT0, T),
)

_ToStrict1D = TypeAliasType(
    "_ToStrict1D",
    CanArray1D[SCT0] | Seq[T | CanArray0D[SCT0]],
    type_params=(SCT0, T),
)
_ToStrict2D = TypeAliasType(
    "_ToStrict2D",
    CanArray2D[SCT0] | Seq[_ToStrict1D[SCT0, T]],
    type_params=(SCT0, T),
)
_ToStrict3D = TypeAliasType(
    "_ToStrict3D",
    CanArray3D[SCT0] | Seq[_ToStrict2D[SCT0, T]],
    type_params=(SCT0, T),
)


###


# TODO: Export and document.

integer_co = TypeAliasType("integer_co", npc.integer | np.bool_)
floating_co = TypeAliasType("floating_co", npc.floating | integer_co)
complexfloating_co = TypeAliasType("complexfloating_co", npc.number | np.bool_)
float64_co = TypeAliasType(
    "float64_co",
    npc.floating64 | npc.floating32 | npc.floating16 | integer_co,
)
complex128_co = TypeAliasType(
    "complex128_co",
    npc.number64 | npc.number32 | npc.number16 | integer_co,
)


###


# scalar- and array-likes, with "coercible" shape-types

ToScalar: TypeAlias = _PyScalar | np.generic
ToArray1D = TypeAliasType("ToArray1D", _To1D[SCT, T], type_params=(T, SCT))
ToArray2D = TypeAliasType("ToArray2D", _To2D[SCT, T], type_params=(T, SCT))
ToArray3D = TypeAliasType("ToArray3D", _To3D[SCT, T], type_params=(T, SCT))
ToArrayND = TypeAliasType("ToArrayND", _ToND[SCT, T], type_params=(T, SCT))

if NP22:
    ToFalse = TypeAliasType("ToFalse", "Literal[False, 0] | np.bool[Literal[False]]")
    ToTrue = TypeAliasType("ToTrue", "Literal[True, 1] | np.bool[Literal[True]]")

    ToJustFalse = TypeAliasType(
        "ToJustFalse",
        "Literal[False] | np.bool[Literal[False]]",
    )
    ToJustTrue = TypeAliasType("ToJustTrue", "Literal[True] | np.bool[Literal[True]]")
else:
    ToFalse = TypeAliasType("ToFalse", Literal[False, 0])
    ToTrue = TypeAliasType("ToTrue", Literal[True, 1])

    ToJustFalse = TypeAliasType("ToJustFalse", Literal[False])
    ToJustTrue = TypeAliasType("ToJustTrue", Literal[True])

ToBool: TypeAlias = _PyBool | np.bool_
ToBool1D: TypeAlias = _To1D[np.bool_, _PyBool]
ToBool2D: TypeAlias = _To2D[np.bool_, _PyBool]
ToBool3D: TypeAlias = _To3D[np.bool_, _PyBool]
ToBoolND: TypeAlias = _ToND[np.bool_, _PyBool]

ToInt: TypeAlias = int | integer_co
ToInt1D: TypeAlias = _To1D[integer_co, int]
ToInt2D: TypeAlias = _To2D[integer_co, int]
ToInt3D: TypeAlias = _To3D[integer_co, int]
ToIntND: TypeAlias = _ToND[integer_co, int]

ToFloat64: TypeAlias = float | float64_co
ToFloat64_1D: TypeAlias = _To1D[float64_co, float]
ToFloat64_2D: TypeAlias = _To2D[float64_co, float]
ToFloat64_3D: TypeAlias = _To3D[float64_co, float]
ToFloat64_ND: TypeAlias = _ToND[float64_co, float]

ToFloat: TypeAlias = float | floating_co
ToFloat1D: TypeAlias = _To1D[floating_co, float]
ToFloat2D: TypeAlias = _To2D[floating_co, float]
ToFloat3D: TypeAlias = _To3D[floating_co, float]
ToFloatND: TypeAlias = _ToND[floating_co, float]

ToComplex128: TypeAlias = complex | complex128_co
ToComplex128_1D: TypeAlias = _To1D[complex128_co, complex]
ToComplex128_2D: TypeAlias = _To2D[complex128_co, complex]
ToComplex128_3D: TypeAlias = _To3D[complex128_co, complex]
ToComplex128_ND: TypeAlias = _ToND[complex128_co, complex]

ToComplex: TypeAlias = complex | complexfloating_co
ToComplex1D: TypeAlias = _To1D[complexfloating_co, complex]
ToComplex2D: TypeAlias = _To2D[complexfloating_co, complex]
ToComplex3D: TypeAlias = _To3D[complexfloating_co, complex]
ToComplexND: TypeAlias = _ToND[complexfloating_co, complex]

# scalar- and array-likes, with "just" that scalar type

ToJustBool: TypeAlias = bool | np.bool_
ToJustBool1D: TypeAlias = _To1D[np.bool_, bool]
ToJustBool2D: TypeAlias = _To2D[np.bool_, bool]
ToJustBool3D: TypeAlias = _To3D[np.bool_, bool]
ToJustBoolND: TypeAlias = _ToND[np.bool_, bool]

ToJustInt64: TypeAlias = JustInt | np.int64
ToJustInt64_1D: TypeAlias = _To1D[np.int64, JustInt]
ToJustInt64_2D: TypeAlias = _To2D[np.int64, JustInt]
ToJustInt64_3D: TypeAlias = _To3D[np.int64, JustInt]
ToJustInt64_ND: TypeAlias = _ToND[np.int64, JustInt]

ToJustInt: TypeAlias = JustInt | npc.integer
ToJustInt1D: TypeAlias = _To1D[npc.integer, JustInt]
ToJustInt2D: TypeAlias = _To2D[npc.integer, JustInt]
ToJustInt3D: TypeAlias = _To3D[npc.integer, JustInt]
ToJustIntND: TypeAlias = _ToND[npc.integer, JustInt]

ToJustFloat64: TypeAlias = JustFloat | npc.floating64
ToJustFloat64_1D: TypeAlias = _To1D[npc.floating64, JustFloat]
ToJustFloat64_2D: TypeAlias = _To2D[npc.floating64, JustFloat]
ToJustFloat64_3D: TypeAlias = _To3D[npc.floating64, JustFloat]
ToJustFloat64_ND: TypeAlias = _ToND[npc.floating64, JustFloat]

ToJustFloat: TypeAlias = JustFloat | npc.floating
ToJustFloat1D: TypeAlias = _To1D[npc.floating, JustFloat]
ToJustFloat2D: TypeAlias = _To2D[npc.floating, JustFloat]
ToJustFloat3D: TypeAlias = _To3D[npc.floating, JustFloat]
ToJustFloatND: TypeAlias = _ToND[npc.floating, JustFloat]

ToJustComplex: TypeAlias = JustComplex | npc.complexfloating
ToJustComplex1D: TypeAlias = _To1D[npc.complexfloating, JustComplex]
ToJustComplex2D: TypeAlias = _To2D[npc.complexfloating, JustComplex]
ToJustComplex3D: TypeAlias = _To3D[npc.complexfloating, JustComplex]
ToJustComplexND: TypeAlias = _ToND[npc.complexfloating, JustComplex]

ToJustComplex128: TypeAlias = JustComplex | npc.complexfloating128
ToJustComplex128_1D: TypeAlias = _To1D[npc.complexfloating128, JustComplex]
ToJustComplex128_2D: TypeAlias = _To2D[npc.complexfloating128, JustComplex]
ToJustComplex128_3D: TypeAlias = _To3D[npc.complexfloating128, JustComplex]
ToJustComplex128_ND: TypeAlias = _ToND[npc.complexfloating128, JustComplex]

# array-likes, with "coercible" shape-types, and "strict" shape-types

ToArrayStrict1D = TypeAliasType(
    "ToArrayStrict1D",
    _ToStrict1D[SCT, T],
    type_params=(T, SCT),
)
ToArrayStrict2D = TypeAliasType(
    "ToArrayStrict2D",
    _ToStrict2D[SCT, T],
    type_params=(T, SCT),
)
ToArrayStrict3D = TypeAliasType(
    "ToArrayStrict3D",
    _ToStrict3D[SCT, T],
    type_params=(T, SCT),
)

ToBoolStrict1D: TypeAlias = _ToStrict1D[np.bool_, _PyBool]
ToBoolStrict2D: TypeAlias = _ToStrict2D[np.bool_, _PyBool]
ToBoolStrict3D: TypeAlias = _ToStrict3D[np.bool_, _PyBool]

ToIntStrict1D: TypeAlias = _ToStrict1D[integer_co, int]
ToIntStrict2D: TypeAlias = _ToStrict2D[integer_co, int]
ToIntStrict3D: TypeAlias = _ToStrict3D[integer_co, int]

ToFloat64Strict1D: TypeAlias = _ToStrict1D[float64_co, float]
ToFloat64Strict2D: TypeAlias = _ToStrict2D[float64_co, float]
ToFloat64Strict3D: TypeAlias = _ToStrict3D[float64_co, float]

ToFloatStrict1D: TypeAlias = _ToStrict1D[floating_co, float]
ToFloatStrict2D: TypeAlias = _ToStrict2D[floating_co, float]
ToFloatStrict3D: TypeAlias = _ToStrict3D[floating_co, float]

ToComplex128Strict1D: TypeAlias = _ToStrict1D[complex128_co, complex]
ToComplex128Strict2D: TypeAlias = _ToStrict2D[complex128_co, complex]
ToComplex128Strict3D: TypeAlias = _ToStrict3D[complex128_co, complex]

ToComplexStrict1D: TypeAlias = _ToStrict1D[complexfloating_co, complex]
ToComplexStrict2D: TypeAlias = _ToStrict2D[complexfloating_co, complex]
ToComplexStrict3D: TypeAlias = _ToStrict3D[complexfloating_co, complex]

# array-likes, with "just" that scalar type, and "strict" shape-types

ToJustBoolStrict1D: TypeAlias = _ToStrict1D[np.bool_, bool]
ToJustBoolStrict2D: TypeAlias = _ToStrict2D[np.bool_, bool]
ToJustBoolStrict3D: TypeAlias = _ToStrict3D[np.bool_, bool]

ToJustInt64Strict1D: TypeAlias = _ToStrict1D[np.int64, JustInt]
ToJustInt64Strict2D: TypeAlias = _ToStrict2D[np.int64, JustInt]
ToJustInt64Strict3D: TypeAlias = _ToStrict3D[np.int64, JustInt]

ToJustIntStrict1D: TypeAlias = _ToStrict1D[npc.integer, JustInt]
ToJustIntStrict2D: TypeAlias = _ToStrict2D[npc.integer, JustInt]
ToJustIntStrict3D: TypeAlias = _ToStrict3D[npc.integer, JustInt]

ToJustFloat64Strict1D: TypeAlias = _ToStrict1D[npc.floating64, JustFloat]
ToJustFloat64Strict2D: TypeAlias = _ToStrict2D[npc.floating64, JustFloat]
ToJustFloat64Strict3D: TypeAlias = _ToStrict3D[npc.floating64, JustFloat]

ToJustFloatStrict1D: TypeAlias = _ToStrict1D[npc.floating, JustFloat]
ToJustFloatStrict2D: TypeAlias = _ToStrict2D[npc.floating, JustFloat]
ToJustFloatStrict3D: TypeAlias = _ToStrict3D[npc.floating, JustFloat]

ToJustComplex128Strict1D: TypeAlias = _ToStrict1D[npc.complexfloating128, JustComplex]
ToJustComplex128Strict2D: TypeAlias = _ToStrict2D[npc.complexfloating128, JustComplex]
ToJustComplex128Strict3D: TypeAlias = _ToStrict3D[npc.complexfloating128, JustComplex]

ToJustComplexStrict1D: TypeAlias = _ToStrict1D[npc.complexfloating, JustComplex]
ToJustComplexStrict2D: TypeAlias = _ToStrict2D[npc.complexfloating, JustComplex]
ToJustComplexStrict3D: TypeAlias = _ToStrict3D[npc.complexfloating, JustComplex]
