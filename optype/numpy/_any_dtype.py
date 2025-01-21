"""
The allowed `np.dtype` arguments for specific scalar types.
The names are analogous to those in `numpy.dtypes`.
"""

# pyright: reportInvalidTypeArguments=false

import sys
from typing import Protocol, TypeAlias as Alias

if sys.version_info >= (3, 13):
    from typing import Never, TypeAliasType, TypeVar
else:
    from typing_extensions import Never, TypeAliasType, TypeVar

import numpy as np

import optype.numpy._compat as _x
import optype.numpy._dtype as _dt
import optype.numpy._dtype_attr as _a
import optype.numpy._scalar as _sc
from optype._core._just import JustComplex, JustFloat, JustInt, JustObject

# ruff: noqa: RUF022
__all__ = [
    "AnyDType",
    "AnyNumberDType",
    "AnyIntegerDType",
    "AnyInexactDType",
    "AnyFlexibleDType",
    "AnyUnsignedIntegerDType",
    "AnySignedIntegerDType",
    "AnyFloatingDType",
    "AnyComplexFloatingDType",
    "AnyCharacterDType",

    "AnyBoolDType",

    "AnyUIntDType",
    "AnyUInt8DType",
    "AnyUInt8DType",
    "AnyUInt16DType",
    "AnyUInt32DType",
    "AnyUInt64DType",
    "AnyUIntPDType",
    "AnyUByteDType",
    "AnyUShortDType",
    "AnyUIntCDType",
    "AnyULongDType",
    "AnyULongLongDType",

    "AnyIntDType",
    "AnyInt8DType",
    "AnyInt8DType",
    "AnyInt16DType",
    "AnyInt32DType",
    "AnyInt64DType",
    "AnyIntPDType",
    "AnyByteDType",
    "AnyShortDType",
    "AnyIntCDType",
    "AnyLongDType",
    "AnyLongLongDType",

    "AnyFloat16DType",
    "AnyFloat32DType",
    "AnyFloat64DType",
    "AnyLongDoubleDType",

    "AnyComplex64DType",
    "AnyComplex128DType",
    "AnyCLongDoubleDType",

    "AnyDateTime64DType",
    "AnyTimeDelta64DType",

    "AnyBytesDType",
    "AnyBytes8DType",
    "AnyStrDType",
    "AnyVoidDType",
    "AnyObjectDType",

    "AnyStringDType",
]  # fmt: skip


def __dir__() -> list[str]:
    return __all__


###


_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_co = TypeVar("_SCT_co", bound=np.generic, covariant=True)


# instead of using `HasDType[np.dtype[_SCT]]`, we use this (more specific) protocol
# for improved readability of introspection and type-checker errors
class _HasScalarType(Protocol[_SCT_co]):
    @property
    def dtype(self, /) -> np.dtype[_SCT_co]: ...


# TODO(jorenham): Export this as `onp.ToDType` (and document).
_ToDType: Alias = type[_SCT] | np.dtype[_SCT] | _HasScalarType[_SCT]


_IsInt: Alias = type[JustInt]
_IsFloat: Alias = type[JustFloat]
_IsComplex: Alias = type[JustComplex]
_IsObject: Alias = type[JustObject]


###

# bool

AnyBoolDType = TypeAliasType(
    "AnyBoolDType", type[bool] | _ToDType[_x.Bool] | _a.b1_code
)

# signed integers

AnyInt8DType = TypeAliasType("AnyInt8DType", _ToDType[np.int8] | _a.i1_code)
AnyByteDType = AnyInt8DType
AnyInt16DType = TypeAliasType("AnyInt16DType", _ToDType[np.int16] | _a.i2_code)
AnyShortDType = AnyInt16DType
AnyInt32DType = TypeAliasType("AnyInt32DType", _ToDType[np.int32] | _a.i4_code)
AnyIntCDType = AnyInt32DType
AnyInt64DType = TypeAliasType("AnyInt64DType", _ToDType[np.int64] | _a.i8_code)
AnyLongLongDType = AnyInt64DType

# unsigned integers

AnyUInt8DType = TypeAliasType("AnyUInt8DType", _ToDType[np.uint8] | _a.u1_code)
AnyUByteDType = AnyUInt8DType
AnyUInt16DType = TypeAliasType("AnyUInt16DType", _ToDType[np.uint16] | _a.u2_code)
AnyUShortDType = AnyUInt16DType
AnyUInt32DType = TypeAliasType("AnyUInt32DType", _ToDType[np.uint32] | _a.u4_code)
AnyUIntCDType = AnyUInt32DType
AnyUInt64DType = TypeAliasType("AnyUInt64DType", _ToDType[np.uint64] | _a.u8_code)
AnyULongLongDType = AnyUInt64DType

# real floating

AnyFloat16DType = TypeAliasType("AnyFloat16DType", _ToDType[np.float16] | _a.f2_code)
AnyFloat32DType = TypeAliasType("AnyFloat32DType", _ToDType[np.float32] | _a.f4_code)
AnyFloat64DType = TypeAliasType(
    "AnyFloat64DType",
    _IsFloat | _ToDType[np.float64] | _a.f8_code | None,
)
AnyLongDoubleDType = TypeAliasType(
    "AnyLongDoubleDType",
    _ToDType[np.longdouble] | _a.g_code,
)
AnyFloatingDType = TypeAliasType(
    "AnyFloatingDType",
    _IsFloat | _ToDType[_sc.floating] | _a.fx_code,
)

# complex floating

AnyComplex64DType = TypeAliasType(
    "AnyComplex64DType",
    _ToDType[np.complex64] | _a.c8_code,
)
AnyComplex128DType = TypeAliasType(
    "AnyComplex128DType",
    _IsComplex | _ToDType[np.complex128] | _a.c16_code,
)
AnyCLongDoubleDType = TypeAliasType(
    "AnyCLongDoubleDType",
    _ToDType[np.clongdouble] | _a.G_code,
)
AnyComplexFloatingDType = TypeAliasType(
    "AnyComplexFloatingDType",
    _IsComplex | _ToDType[_sc.cfloating] | _a.cx_code,
)

# temporal

AnyDateTime64DType = TypeAliasType(
    "AnyDateTime64DType",
    _ToDType[np.datetime64] | _a.M8_code,
)

AnyTimeDelta64DType = TypeAliasType(
    "AnyTimeDelta64DType",
    _ToDType[np.timedelta64] | _a.m8_code,
)

# flexible

AnyBytesDType = TypeAliasType(
    "AnyBytesDType",
    type[bytes] | _ToDType[np.bytes_] | _a.S0_code,
)
AnyBytes8DType = TypeAliasType("AnyBytes8DType", _a.S1_code)
AnyStrDType = TypeAliasType("AnyStrDType", type[str] | _ToDType[np.str_] | _a.U0_code)
AnyCharacterDType = TypeAliasType(
    "AnyCharacterDType",
    type[bytes | str] | _ToDType[np.character] | _a.SU_code,
)

# TODO: Include structured DType values, e.g. `dtype(('u8', 4))`
AnyVoidDType = TypeAliasType(
    "AnyVoidDType",
    type[memoryview] | _ToDType[np.void] | _a.V0_code,
)
AnyFlexibleDType = TypeAliasType(
    "AnyFlexibleDType",
    type[bytes | str | memoryview] | _ToDType[np.flexible] | _a.SUV_code,
)

# object

AnyObjectDType = TypeAliasType(
    "AnyObjectDType",
    _IsObject | _ToDType[np.object_] | _a.O_code,
)
AnyInexactDType = TypeAliasType(
    "AnyInexactDType",
    _IsFloat | _IsComplex | _ToDType[_sc.inexact] | _a.fc_code,
)
AnyNumberDType = TypeAliasType(
    "AnyNumberDType",
    _IsInt | _IsFloat | _IsComplex | _ToDType[_sc.number] | _a.uifc_code,
)


# NOTE: At the moment, `np.dtypes.StringDType.type: type[str]`, which is
# impossible (i.e. `dtype[str]` isn't valid, as `str` isn't a `np.generic`)

AnyULongDType = TypeAliasType("AnyULongDType", _ToDType[_x.ULong] | _a.L_code)
AnyUIntPDType = TypeAliasType("AnyUIntPDType", _ToDType[np.uintp] | _a.u0_code)

AnyUnsignedIntegerDType = TypeAliasType(
    "AnyUnsignedIntegerDType",
    _ToDType[_sc.uinteger] | _a.ux_code,
)
AnySignedIntegerDType = TypeAliasType(
    "AnySignedIntegerDType",
    _IsInt | _ToDType[_sc.sinteger] | _a.ix_code,
)
AnyIntegerDType = TypeAliasType(
    "AnyIntegerDType",
    _IsInt | _ToDType[_sc.integer] | _a.ui_code,
)

if _x.NP20:
    AnyIntPDType = TypeAliasType(
        "AnyIntPDType",
        _IsInt | _ToDType[np.intp] | _a.i0_code,
    )
    AnyLongDType = TypeAliasType("AnyLongDType", _ToDType[_x.Long] | _a.l_code)
    AnyIntDType = AnyIntPDType
    AnyUIntDType = AnyUIntPDType

    # NOTE: `np.dtypes.StringDType` didn't exist in the stubs prior to 2.1 (so
    # I (@jorenham) added them, see https://github.com/numpy/numpy/pull/27008).
    if _x.NP21:
        # `numpy>=2.1`
        _HasStringDType: Alias = _dt.HasDType[np.dtypes.StringDType]  # type: ignore[type-var]
        AnyStringDType = TypeAliasType("AnyStringDType", _HasStringDType | _a.T_code)  # type: ignore[type-var]
        AnyDType = TypeAliasType(
            "AnyDType",
            type | _ToDType[np.generic] | str | _HasStringDType,
        )
    else:
        AnyStringDType = TypeAliasType("AnyStringDType", np.dtype[Never] | _a.T_code)
        AnyDType = TypeAliasType("AnyDType", type | _ToDType[np.generic] | str)
else:
    # assuming that `c_void_p == c_size_t`
    AnyIntPDType = TypeAliasType("AnyIntPDType", _ToDType[np.intp] | _a.i0_code)
    AnyLongDType = TypeAliasType("AnyLongDType", _IsInt | _ToDType[_x.Long] | _a.l_code)
    AnyIntDType: Alias = AnyLongDType
    AnyUIntDType: Alias = AnyULongDType

    AnyStringDType = TypeAliasType("AnyStringDType", Never)
    AnyDType = TypeAliasType("AnyDType", type | _ToDType[np.generic] | str)
