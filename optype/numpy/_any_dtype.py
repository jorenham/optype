# mypy: disable-error-code="no-any-explicit, no-any-decorated"
# pyright: reportInvalidTypeArguments=false

"""
The allowed `np.dtype` arguments for specific scalar types.
The names are analogous to those in `numpy.dtypes`.
"""

from __future__ import annotations

import sys
from typing import (
    Literal as L,  # noqa: N817
    TypeAlias as Alias,
)

import numpy as np
from typing_extensions import TypeAliasType

import optype.numpy._compat as _x
import optype.numpy._dtype as _dt
import optype.numpy._scalar as _sc
import optype.typing as opt


if sys.version_info >= (3, 13):
    from typing import LiteralString, Never, Protocol, TypeVar
else:
    from typing_extensions import LiteralString, Never, Protocol, TypeVar


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


_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)

_IsFloat: Alias = type[opt.Just[float]]
_IsComplex: Alias = type[opt.Just[complex]]
_IsObject: Alias = type[opt.Just[object]]


# instead of using `HasDType[np.dtype[_SCT]]`, we use this (more specific) protocol
# for improved readability of introspection and type-checker errors
class _HasScalarType(Protocol[_SCT_co]):
    @property
    def dtype(self, /) -> np.dtype[_SCT_co]: ...


_ToDType: Alias = type[_SCT] | np.dtype[_SCT] | _HasScalarType[_SCT]


# bool

_Name_b1: Alias = L["bool", "bool_"]  # 'bool0' was removed in NumPy 2.0
_Char_b1: Alias = L["?", "b1", "|b1"]
_Code_b1: Alias = L[_Name_b1, _Char_b1]
AnyBoolDType = TypeAliasType("AnyBoolDType", type[bool] | _ToDType[_x.Bool] | _Code_b1)

# signed integers

_Name_i1: Alias = L["int8", "byte"]
_Char_i1: Alias = L["b", "i1", "|i1"]
_Code_i1: Alias = L[_Name_i1, _Char_i1]
AnyInt8DType = TypeAliasType("AnyInt8DType", _ToDType[np.int8] | _Code_i1)
AnyByteDType: Alias = AnyInt8DType

_Name_i2: Alias = L["int16", "short"]
_Char_i2: Alias = L["h", "i2", "<i2", ">i2"]
_Code_i2: Alias = L[_Name_i2, _Char_i2]
AnyInt16DType = TypeAliasType("AnyInt16DType", _ToDType[np.int16] | _Code_i2)
AnyShortDType: Alias = AnyInt16DType

# `intc` is an alias for `int32` on linux, and almost always of same size if not
_Name_i: Alias = L["intc"]
_Char_i: Alias = L["i"]
_Code_i: Alias = L[_Char_i, _Name_i]
AnyIntCDType = TypeAliasType("AnyIntCDType", _ToDType[np.intc] | _Code_i)

_Name_i4: Alias = L["int32", _Name_i]
_Char_i4: Alias = L[_Char_i, "i4", "<i4", ">i4"]
_Code_i4: Alias = L[_Name_i4, _Char_i4]
AnyInt32DType = TypeAliasType("AnyInt32DType", _ToDType[np.int32] | _Code_i4)

_Name_q: Alias = L["longlong"]
_Char_q: Alias = L["q"]
_Code_q: Alias = L[_Name_q, _Char_q]
AnyLongLongDType = TypeAliasType("AnyLongLongDType", _ToDType[np.longlong] | _Code_q)

_Name_i8: Alias = L["int64", _Name_q]
_Char_i8: Alias = L[_Char_q, "i8", "<i8", ">i8"]
_Code_i8: Alias = L[_Name_i8, _Char_i8]
AnyInt64DType = TypeAliasType("AnyInt64DType", _ToDType[np.int64] | _Code_i8)

# `Int_`, `IntP`, and `Long` are defined later as they differ in numpy 1 and 2
_Name_i0_common: Alias = L["int", "int_"]
_Char_l: Alias = L["l", "<l", ">l"]
_Char_p: Alias = L["p", "<p", ">p"]  # not associated to any scalar type in numpy>=2.0

# unsigned integers

_Name_u1: Alias = L["uint8", "ubyte"]
_Char_u1: Alias = L["B", "u1", "|u1"]  # irrelevant byteorder, but `.str == "|u1"`
_Code_u1: Alias = L[_Name_u1, _Char_u1]
AnyUInt8DType = TypeAliasType("AnyUInt8DType", _ToDType[np.uint8] | _Code_u1)
AnyUByteDType: Alias = AnyUInt8DType

_Name_u2: Alias = L["uint16", "ushort"]
_Char_u2: Alias = L["H", "u2", "<u2", ">u2"]
_Code_u2: Alias = L[_Name_u2, _Char_u2]
AnyUInt16DType = TypeAliasType("AnyUInt16DType", _ToDType[np.uint16] | _Code_u2)
AnyUShortDType: Alias = AnyUInt16DType

# `uintc` is an alias for `uint32` on linux (e.g. the Win16 API uses 16-bit, but
# realistically no-one is using that anymore)
_Name_I: Alias = L["uintc"]
_Char_I: Alias = L["I"]
_Code_I: Alias = L[_Name_I, _Char_I]
AnyUIntCDType = TypeAliasType("AnyUIntCDType", _ToDType[np.uintc] | _Code_I)

_Name_u4: Alias = L["uint32", _Name_I]
_Char_u4: Alias = L[_Char_I, "u4", "<u4", ">u4"]
_Code_u4: Alias = L[_Name_u4, _Char_u4]
AnyUInt32DType = TypeAliasType("AnyUInt32DType", _ToDType[np.uint32] | _Code_u4)

_Name_Q: Alias = L["ulonglong"]
_Char_Q: Alias = L["Q"]
_Code_Q: Alias = L[_Name_Q, _Char_Q]
AnyULongLongDType = TypeAliasType("AnyULongLongDType", _ToDType[np.ulonglong] | _Code_Q)

_Name_u8: Alias = L["uint64", _Name_Q]
_Char_u8: Alias = L[_Char_Q, "u8", "<u8", ">u8"]
_Code_u8: Alias = L[_Name_u8, _Char_u8]
AnyUInt64DType = TypeAliasType("AnyUInt64DType", _ToDType[np.uint64] | _Code_u8)

# `UInt`, `UIntP`, and `ULong` are defined later as they differ in numpy 1 and 2
_Name_u0_common: Alias = L["uint"]
_Char_L: Alias = L["L", "<L", ">L"]
_Char_P: Alias = L["P", "<P", ">P"]  # not associated to any scalar type in numpy>=2.0

_Name_ux: Alias = L[
    "uint", "uintp",
    "uint8", "uint16", "uint32", "uint64",
    "ubyte", "ushort", "uintc", "ulong", "ulonglong",
]  # fmt: skip
_Char_ux_common: Alias = L[_Char_u1, _Char_u2, _Char_u4, _Char_u8, _Char_L, _Char_P]


# real floating

_Name_f2: Alias = L["float16", "half"]
_Char_f2: Alias = L["e", "f2", "<f2", ">f2"]
_Code_f2: Alias = L[_Name_f2, _Char_f2]
AnyFloat16DType = TypeAliasType("AnyFloat16DType", _ToDType[np.float16] | _Code_f2)

_Name_f4: Alias = L["float32", "single"]
_Char_f4: Alias = L["f", "f4", "<f4", ">f4"]
_Code_f4: Alias = L[_Name_f4, _Char_f4]
AnyFloat32DType = TypeAliasType("AnyFloat32DType", _ToDType[np.float32] | _Code_f4)

_Name_f8: Alias = L["float", "float64", "double"]
_Char_f8: Alias = L["d", "f8", "<f8", ">f8"]
_Code_f8: Alias = L[_Name_f8, _Char_f8]
AnyFloat64DType = TypeAliasType(
    "AnyFloat64DType",
    _IsFloat | _ToDType[np.float64] | _Code_f8 | None,
)

_Name_g: Alias = L["longdouble", "float80", "float96", "float128"]
_Char_g: Alias = L[
    "g", "f10", "f12", "f16", "<f10", "<f12", "<f16", ">f10", ">f12", ">f16"
]
_Code_g: Alias = L[_Name_g, _Char_g]
AnyLongDoubleDType = TypeAliasType(
    "AnyLongDoubleDType",
    _ToDType[np.longdouble] | _Code_g,
)

_Code_fx: Alias = L[_Code_f2, _Code_f4, _Code_f8, _Code_g]
AnyFloatingDType = TypeAliasType(
    "AnyFloatingDType",
    _IsFloat | _ToDType[_sc.floating] | _Code_fx,
)

# complex floating

_Name_c8: Alias = L["complex64", "csingle"]
_Char_c8: Alias = L["F", "c8", "<c8", ">c8"]
_Code_c8: Alias = L[_Name_c8, _Char_c8]
AnyComplex64DType = TypeAliasType(
    "AnyComplex64DType",
    _ToDType[np.complex64] | _Code_c8,
)

_Name_c16: Alias = L["complex", "complex128", "cdouble"]
_Char_c16: Alias = L["D", "c16", "<c16", ">c16"]
_Code_c16: Alias = L[_Name_c16, _Char_c16]
AnyComplex128DType = TypeAliasType(
    "AnyComplex128DType",
    _IsComplex | _ToDType[np.complex128] | _Code_c16,
)

_Name_G: Alias = L["clongdouble", "float160", "float192", "float256"]
_Char_G: Alias = L[
    "G", "c20", "c24", "c32", "<c20", "<c24", "<c32", ">c20", ">c24", ">c32"
]
_Code_G: Alias = L[_Name_G, _Char_G]
AnyCLongDoubleDType = TypeAliasType(
    "AnyCLongDoubleDType",
    _ToDType[np.clongdouble] | _Code_G,
)

_Code_cx: Alias = L[_Code_c8, _Code_c16, _Code_G]
AnyComplexFloatingDType = TypeAliasType(
    "AnyComplexFloatingDType",
    _IsComplex | _ToDType[_sc.cfloating] | _Code_cx,
)

# temporal
# NOTE: The units only include the "unit units", even though e.g.`[5s]` is also valid.

_Name_M8: Alias = L[
    "datetime64",
    "datetime64[as]",
    "datetime64[fs]",
    "datetime64[ps]",
    "datetime64[ns]",
    "datetime64[us]",
    "datetime64[ms]",
    "datetime64[s]",
    "datetime64[m]",
    "datetime64[h]",
    "datetime64[D]",
    "datetime64[W]",
    "datetime64[M]",
    "datetime64[Y]",
]
_Char_M8: Alias = L[
    "M",
    "M8", "<M8", ">M8",
    "M8[as]", "<M8[as]", ">M8[as]",
    "M8[fs]", "<M8[fs]", ">M8[fs]",
    "M8[ps]", "<M8[ps]", ">M8[ps]",
    "M8[ns]", "<M8[ns]", ">M8[ns]",
    "M8[us]", "<M8[us]", ">M8[us]",
    "M8[s]", "<M8[s]", ">M8[s]",
    "M8[m]", "<M8[m]", ">M8[m]",
    "M8[h]", "<M8[h]", ">M8[h]",
    "M8[D]", "<M8[D]", ">M8[D]",
    "M8[W]", "<M8[W]", ">M8[W]",
    "M8[M]", "<M8[M]", ">M8[M]",
    "M8[Y]", "<M8[Y]", ">M8[Y]",
]  # fmt: skip
_Code_M8: Alias = L[_Name_M8, _Char_M8]
AnyDateTime64DType = TypeAliasType(
    "AnyDateTime64DType",
    _ToDType[np.datetime64] | _Code_M8,
)

_Name_m8: Alias = L[
    "timedelta64",
    "timedelta64[as]",
    "timedelta64[fs]",
    "timedelta64[ps]",
    "timedelta64[ns]",
    "timedelta64[us]",
    "timedelta64[ms]",
    "timedelta64[s]",
    "timedelta64[m]",
    "timedelta64[h]",
    "timedelta64[D]",
    "timedelta64[W]",
    "timedelta64[M]",
    "timedelta64[Y]",
]
_Char_m8: Alias = L[
    "m",
    "m8", "<m8", ">m8",
    "m8[as]", "<m8[as]", ">m8[as]",
    "m8[fs]", "<m8[fs]", ">m8[fs]",
    "m8[ps]", "<m8[ps]", ">m8[ps]",
    "m8[ns]", "<m8[ns]", ">m8[ns]",
    "m8[us]", "<m8[us]", ">m8[us]",
    "m8[s]", "<m8[s]", ">m8[s]",
    "m8[m]", "<m8[m]", ">m8[m]",
    "m8[h]", "<m8[h]", ">m8[h]",
    "m8[D]", "<m8[D]", ">m8[D]",
    "m8[W]", "<m8[W]", ">m8[W]",
    "m8[M]", "<m8[M]", ">m8[M]",
    "m8[Y]", "<m8[Y]", ">m8[Y]",
]  # fmt: skip
_Code_m8: Alias = L[_Name_m8, _Char_m8]
AnyTimeDelta64DType = TypeAliasType(
    "AnyTimeDelta64DType",
    _ToDType[np.timedelta64] | _Code_m8,
)

# flexible

_Name_U: Alias = L["str", "str_", "unicode"]
_Char_U: Alias = L["U", "U0", "|U0", "<U0", ">U0"]
_Code_U: Alias = L[_Name_U, _Char_U]
AnyStrDType = TypeAliasType("AnyStrDType", type[str] | _ToDType[np.str_] | _Code_U)

_Name_S: Alias = L["bytes", "bytes_"]
_Char_S: Alias = L["S", "S0", "|S0", "<S0", ">S0"]
_Code_S: Alias = L[_Name_S, _Char_S]
AnyBytesDType = TypeAliasType(
    "AnyBytesDType",
    type[bytes] | _ToDType[np.bytes_] | _Code_S,
)

# NOTE: `dtype("c").name == "bytes8"`, but `dtype("bytes8")` raises a TypeError.
_Name_c: Alias = L["bytes8"]
# NOTE: `dtype("c") == dtype("S1"), but `dtype("c").char != dtype("S1")`
_Char_c: Alias = L["c", "S1", "|S1", "<S1", ">S1"]
_Code_c: Alias = L[_Name_c, _Char_c]
AnyBytes8DType = TypeAliasType("AnyBytes8DType", _Code_c)

_Code_SU: Alias = L[_Code_U, _Code_S, _Code_c]
AnyCharacterDType = TypeAliasType(
    "AnyCharacterDType",
    type[bytes | str] | _ToDType[np.character] | _Code_SU,
)

# TODO: Include structured DType values, e.g. `dtype(('u8', 4))`
_Name_V: Alias = L["void"]  # 'void0' was removed in NumPy 2.0
_Char_V: Alias = L["V", "V0", "|V0"]
_Code_V: Alias = L[_Name_V, _Char_V]
AnyVoidDType = TypeAliasType(
    "AnyVoidDType",
    type[memoryview] | _ToDType[np.void] | _Code_V,
)

# flexible
_Code_SUV: Alias = L[_Code_SU, _Code_V]
AnyFlexibleDType = TypeAliasType(
    "AnyFlexibleDType",
    type[bytes | str | memoryview] | _ToDType[np.flexible] | _Code_SUV,
)

# object
_Name_O: Alias = L["object", "object_"]
_Char_O: Alias = L["O", "|O"]
_Code_O: Alias = L[_Name_O, _Char_O]
# NOTE: `type[object]` isn't included, since this could lead to many bugs
#   e.g. in `numpy<2.1` we have `dtype(type[str | float]) -> dtype[object_]`...
AnyObjectDType = TypeAliasType(
    "AnyObjectDType",
    _IsObject | _ToDType[np.object_] | _Code_O,
)

_Code_fc: Alias = L[_Code_fx, _Code_cx]
AnyInexactDType = TypeAliasType(
    "AnyInexactDType",
    _IsFloat | _IsComplex | _ToDType[_sc.inexact] | _Code_fc,
)

_Name_ix: Alias = L[
    "int", "int_", "intp",
    "int8", "int16", "int32", "int64",
    "byte", "short", "intc", "long", "longlong",
]  # fmt: skip
_Char_ix_common: Alias = L[_Char_i1, _Char_i2, _Char_i4, _Char_i8, _Char_l, _Char_p]

# NOTE: At the moment, `np.dtypes.StringDType.type: type[str]`, which is
# impossible (i.e. `dtype[str]` isn't valid, as `str` isn't a `np.generic`)
_Name_T = Never
_Char_T: Alias = L["T"]
_Code_T: Alias = _Char_T

if _x.NP20:
    _Name_u0: Alias = L[_Name_u0_common, "uintp"]
    _Char_u0: Alias = L["N", "<N", ">N"]
    _Code_u0: Alias = L[_Name_u0, _Char_u0]
    AnyUIntPDType = TypeAliasType("AnyUIntPDType", _ToDType[np.uintp] | _Code_u0)
    AnyUIntDType: Alias = AnyUIntPDType

    _Name_L: Alias = L["ulong"]
    _Code_L: Alias = L[_Name_L, _Char_L]
    AnyULongDType = TypeAliasType("AnyULongDType", _ToDType[_x.ULong] | _Code_L)

    _Name_i0: Alias = L["intp", _Name_i0_common]
    _Char_i0: Alias = L["n", "<n", ">n"]
    _Code_i0: Alias = L[_Name_i0, _Char_i0]
    AnyIntPDType = TypeAliasType(
        "AnyIntPDType",
        type[opt.JustInt] | _ToDType[np.intp] | _Code_i0,
    )
    AnyIntDType: Alias = AnyIntPDType

    _Name_l: Alias = L["long"]
    _Code_l: Alias = L[_Name_l, _Char_l]
    AnyLongDType = TypeAliasType("AnyLongDType", _ToDType[_x.Long] | _Code_l)

    _Char_ux: Alias = L[_Char_ux_common, _Char_u0, _Char_P]
    _Code_ux: Alias = L[_Name_ux, _Char_ux]
    AnyUnsignedIntegerDType = TypeAliasType(
        "AnyUnsignedIntegerDType",
        _ToDType[_sc.uinteger] | _Code_ux,
    )

    _Char_ix: Alias = L[_Char_ix_common, _Char_i0, _Char_p]
    _Code_ix: Alias = L[_Name_ix, _Char_ix]
    AnySignedIntegerDType = TypeAliasType(
        "AnySignedIntegerDType",
        _ToDType[_sc.sinteger] | _Code_ix,
    )

    _Code_ui: Alias = L[_Code_ux, _Code_ix]
    AnyIntegerDType = TypeAliasType("AnyIntegerDType", _ToDType[_sc.integer] | _Code_ui)

    _Code_uifc: Alias = L[_Code_ui, _Code_fc]
    # NOTE: this doesn't include `int` or `float` or `complex`, since that
    # would automatically include `bool`.
    AnyNumberDType = TypeAliasType("AnyNumberDType", _ToDType[_sc.number] | _Code_uifc)

    # NOTE: `np.dtypes.StringDType` didn't exist in the stubs prior to 2.1 (so
    # I (@jorenham) added them, see https://github.com/numpy/numpy/pull/27008).
    if _x.NP21:
        # `numpy>=2.1`
        _HasStringDType: Alias = _dt.HasDType[np.dtypes.StringDType]  # type: ignore[type-var]
        AnyStringDType = TypeAliasType("AnyStringDType", _HasStringDType | _Code_T)  # type: ignore[type-var]
        AnyDType = TypeAliasType(
            "AnyDType",
            type | _ToDType[np.generic] | _HasStringDType | str,
        )
    else:
        AnyStringDType = TypeAliasType("AnyStringDType", np.dtype[Never] | _Code_T)
        AnyDType = TypeAliasType("AnyDType", type | _ToDType[np.generic] | str)
else:
    _Name_u0: Alias = L["uintp"]  # 'uint0' is removed in NumPy 2.0
    _Char_u0: Alias = _Char_P
    _Code_u0: Alias = L[_Name_u0, _Char_u0]
    # assuming that `c_void_p == c_size_t`
    AnyUIntPDType = TypeAliasType("AnyUIntPDType", _ToDType[np.uintp,] | _Code_u0)

    _Name_L: Alias = L[_Name_u0_common, "ulong"]
    _Code_L: Alias = L[_Name_L, _Char_L]
    AnyULongDType = TypeAliasType("AnyULongDType", _ToDType[_x.ULong] | _Code_L)
    AnyUIntDType: Alias = AnyULongDType

    _Name_i0: Alias = L["intp"]  # 'int0' is removed in NumPy 2.0
    _Char_i0: Alias = _Char_p
    _Code_i0: Alias = L[_Name_i0, _Char_i0]
    AnyIntPDType = TypeAliasType("AnyIntPDType", _ToDType[np.intp] | _Code_i0)

    _Name_l: Alias = L["long", _Name_i0_common]
    _Code_l: Alias = L[_Name_l, _Char_l]
    AnyLongDType = TypeAliasType("AnyLongDType", _ToDType[_x.Long] | _Code_l)
    AnyIntDType: Alias = AnyLongDType

    _Char_ux: Alias = L[_Char_ux_common, _Char_u0]
    _Code_ux: Alias = L[_Name_ux, _Char_ux]
    AnyUnsignedIntegerDType = TypeAliasType(
        "AnyUnsignedIntegerDType",
        _ToDType[_sc.uinteger] | _Code_ux,
    )

    _Char_ix: Alias = L[_Char_ix_common, _Char_i0]
    _Code_ix: Alias = L[_Name_ix, _Char_ix]
    AnySignedIntegerDType = TypeAliasType(
        "AnySignedIntegerDType",
        _ToDType[_sc.sinteger] | _Code_ix,
    )

    _Code_ui: Alias = L[_Code_ux, _Code_ix]
    AnyIntegerDType = TypeAliasType("AnyIntegerDType", _ToDType[_sc.integer] | _Code_ui)

    _Code_uifc: Alias = L[_Code_ui, _Code_fc]
    AnyNumberDType = TypeAliasType("AnyNumberDType", _ToDType[_sc.number] | _Code_uifc)

    AnyStringDType = TypeAliasType("AnyStringDType", Never)

    AnyDType = TypeAliasType("AnyDType", type | _ToDType[np.generic] | LiteralString)
