# ruff: noqa: PYI042
# mypy: disable-error-code="no-any-explicit, no-any-decorated"
from __future__ import annotations

import sys
from typing import Literal as L, Protocol, TypeAlias, runtime_checkable  # noqa: N817

from optype import CanBool, CanComplex, CanFloat, CanIndex


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar


__all__ = [  # noqa: RUF022
    "DType",
    "Numeric",
    "Integral",
    "Real",
    "SignedInteger",
    "UnsignedInteger",
    "ComplexFloating",
    "RealFloating",
    "Inexact",
    "Bool",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float16", "Float32", "Float64",
    "Complex64", "Complex128",
    "DefaultInt", "DefaultFloat", "DefaultComplex",
]  # fmt: skip


_1: TypeAlias = L[1]
_2: TypeAlias = L[2]
_4: TypeAlias = L[4]
_8: TypeAlias = L[8]
_16: TypeAlias = L[16]

_NByte_i: TypeAlias = L[1, 2, 4, 8]
_NByte_f: TypeAlias = L[2, 4, 8]
_NByte_c: TypeAlias = L[8, 16]
_NByte_fc: TypeAlias = L[_NByte_f, _NByte_c]
_NByte_ifc: TypeAlias = L[_NByte_i, _NByte_c]


# numpy

_KindT_co = TypeVar("_KindT_co", bound=str, covariant=True, default=str)
_CharT_co = TypeVar("_CharT_co", bound=str, covariant=True, default=str)
_NameT_co = TypeVar("_NameT_co", bound=str, covariant=True, default=str)
_ItemN_co = TypeVar("_ItemN_co", bound=int, covariant=True, default=int)
_DataN_co = TypeVar("_DataN_co", bound=int, covariant=True, default=int)
_T_co = TypeVar(
    "_T_co",
    covariant=True,
    default=CanBool | CanIndex | CanFloat | CanComplex,
)


@runtime_checkable
class _DTypeNP(Protocol[_KindT_co, _CharT_co, _NameT_co, _ItemN_co, _DataN_co, _T_co]):
    @property
    def kind(self) -> _KindT_co: ...
    @property
    def char(self) -> _CharT_co: ...
    @property
    def name(self) -> _NameT_co: ...
    @property
    def itemsize(self) -> _ItemN_co: ...
    @property
    def alignment(self) -> _DataN_co: ...
    @property
    def type(self) -> type[_T_co]: ...


# These correspond to the `numpy.dtypes.{}DType` data types
_b1_np: TypeAlias = _DTypeNP[L["b"], L["?"], L["bool"], _1, _1, CanBool]
_i1_np: TypeAlias = _DTypeNP[L["i"], L["b"], L["int8"], _1, _1, CanIndex]
_i2_np: TypeAlias = _DTypeNP[L["i"], L["h"], L["int16"], _2, _2, CanIndex]
_i4_np: TypeAlias = _DTypeNP[L["i"], L["i", "l"], L["int32"], _4, _4, CanIndex]
_i8_np: TypeAlias = _DTypeNP[L["i"], L["l", "q"], L["int64"], _8, _8, CanIndex]
_u1_np: TypeAlias = _DTypeNP[L["u"], L["B"], L["uint8"], _1, _1, CanIndex]
_u2_np: TypeAlias = _DTypeNP[L["u"], L["H"], L["uint16"], _2, _2, CanIndex]
_u4_np: TypeAlias = _DTypeNP[L["u"], L["I", "L"], L["uint32"], _4, _4, CanIndex]
_u8_np: TypeAlias = _DTypeNP[L["u"], L["L", "Q"], L["uint64"], _8, _8, CanIndex]
# `float16` isn't part of the array-api spec, but the array-api does allow using it
_f2_np: TypeAlias = _DTypeNP[L["f"], L["e"], L["float16"], _2, _2, CanFloat]
_f4_np: TypeAlias = _DTypeNP[L["f"], L["f"], L["float32"], _4, _4, CanFloat]
_f8_np: TypeAlias = _DTypeNP[L["f"], L["d"], L["float64"], _8, _8, CanFloat]
_c8_np: TypeAlias = _DTypeNP[L["c"], L["F"], L["complex64"], _8, _4, CanComplex]
_c16_np: TypeAlias = _DTypeNP[L["c"], L["D"], L["complex128"], _16, _8, CanComplex]

# the `kind` of array-api dtypes, as used in `isdtype()`
# https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype
__Char_i0: TypeAlias = L["i", "l", "q"]
__Char_i: TypeAlias = L["b", "h", __Char_i0]
__Char_u: TypeAlias = L["B", "H", "I", "L", "Q"]
__Char_f: TypeAlias = L["e", "f", "d"]
__Char_c: TypeAlias = L["F", "D"]

__Name_i: TypeAlias = L["int8", "int16", "int32", "int64"]
__Name_i0: TypeAlias = L["int32", "int64"]
__Name_u: TypeAlias = L["uint8", "uint16", "uint32", "uint64"]
__Name_f: TypeAlias = L["float16", "float32", "float64"]
__Name_c: TypeAlias = L["complex64", "complex128"]

_i0_np: TypeAlias = _DTypeNP[L["i"], __Char_i0, __Name_i0, L[4, 8], L[4, 8], CanIndex]
_f0_np: TypeAlias = _f8_np
_c0_np: TypeAlias = _c16_np

_i_np: TypeAlias = _DTypeNP[L["i"], __Char_i, __Name_i, _NByte_i, _NByte_i, CanIndex]
_u_np: TypeAlias = _DTypeNP[L["u"], __Char_u, __Name_u, _NByte_i, _NByte_i, CanIndex]
_f_np: TypeAlias = _DTypeNP[L["f"], __Char_f, __Name_f, _NByte_f, _NByte_f, CanFloat]
_c_np: TypeAlias = _DTypeNP[L["c"], __Char_c, __Name_c, _NByte_c, _NByte_f, CanComplex]

_iu_np: TypeAlias = _DTypeNP[
    L["i", "u"],
    L[__Char_i, __Char_u],
    L[__Name_i, __Name_u],
    _NByte_i,
    _NByte_i,
    CanIndex,
]
_fc_np: TypeAlias = _DTypeNP[
    L["f", "c"],
    L[__Char_f, __Char_c],
    L[__Name_f, __Name_c],
    _NByte_fc,
    _NByte_f,
    CanFloat | CanComplex,
]
_iuf_np: TypeAlias = _DTypeNP[
    L["i", "u", "f"],
    L[__Char_i, __Char_u, __Char_f],
    L[__Name_i, __Name_u, __Name_f],
    _NByte_i,
    _NByte_i,
    CanIndex | CanFloat,
]
_iufc_np: TypeAlias = _DTypeNP[
    L["i", "u", "f", "c"],
    L[__Char_i, __Char_u, __Char_f, __Char_c],
    L[__Name_i, __Name_u, __Name_f, __Name_c],
    _NByte_ifc,
    _NByte_i,
    CanIndex | CanFloat | CanComplex,
]


# pytorch

_SignB_co = TypeVar("_SignB_co", covariant=True, bound=bool, default=bool)
_FloatB_co = TypeVar("_FloatB_co", covariant=True, bound=bool, default=bool)
_ComplexB_co = TypeVar("_ComplexB_co", covariant=True, bound=bool, default=bool)


@runtime_checkable
class _DTypeTorch(Protocol[_ItemN_co, _SignB_co, _FloatB_co, _ComplexB_co]):
    # https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_C/__init__.pyi.in#L181-L189
    @property
    def itemsize(self, /) -> _ItemN_co: ...
    @property
    def is_signed(self, /) -> _SignB_co: ...
    @property
    def is_floating_point(self, /) -> _FloatB_co: ...
    @property
    def is_complex(self, /) -> _ComplexB_co: ...


_TorchT = TypeVar("_TorchT", bound=_DTypeTorch)


_F: TypeAlias = L[False]
_T: TypeAlias = L[True]

_b1_torch: TypeAlias = _DTypeTorch[_1, _F, _F, _F]
_i1_torch: TypeAlias = _DTypeTorch[_1, _T, _F, _F]
_i2_torch: TypeAlias = _DTypeTorch[_2, _T, _F, _F]
_i4_torch: TypeAlias = _DTypeTorch[_4, _T, _F, _F]
_i8_torch: TypeAlias = _DTypeTorch[_8, _T, _F, _F]
_u1_torch: TypeAlias = _DTypeTorch[_1, _F, _F, _F]
_u2_torch: TypeAlias = _DTypeTorch[_2, _F, _F, _F]
_u4_torch: TypeAlias = _DTypeTorch[_4, _F, _F, _F]
_u8_torch: TypeAlias = _DTypeTorch[_8, _F, _F, _F]
_f2_torch: TypeAlias = _DTypeTorch[_2, _T, _T, _F]
_f4_torch: TypeAlias = _DTypeTorch[_4, _T, _T, _F]
_f8_torch: TypeAlias = _DTypeTorch[_8, _T, _T, _F]
_c8_torch: TypeAlias = _DTypeTorch[_8, _T, _F, _T]
_c16_torch: TypeAlias = _DTypeTorch[_16, _T, _F, _T]


_i_torch: TypeAlias = _DTypeTorch[_NByte_i, _T, _F, _F]
_u_torch: TypeAlias = _DTypeTorch[_NByte_i, _F, _F, _F]
_f_torch: TypeAlias = _DTypeTorch[_NByte_f, _T, _T, _F]
_c_torch: TypeAlias = _DTypeTorch[_NByte_c, _T, _F, _T]
_iu_torch: TypeAlias = _DTypeTorch[_NByte_i, bool, _F, _F]
_fc_torch: TypeAlias = _DTypeTorch[_NByte_fc, bool, bool, bool]
_iuf_torch: TypeAlias = _DTypeTorch[_NByte_i, bool, bool, _F]
_iufc_torch: TypeAlias = _DTypeTorch[_NByte_ifc, bool, bool, bool]

_i0_torch: TypeAlias = _DTypeTorch[L[4, 8], _T, _F, _F]
_f0_torch: TypeAlias = _DTypeTorch[L[4, 8], _T, _T, _F]
_c0_torch: TypeAlias = _c_torch

# array-api-strict

_NPT = TypeVar("_NPT", bound=_DTypeNP)
_NPT_co = TypeVar("_NPT_co", covariant=True, bound=_DTypeNP)


@runtime_checkable
class _DTypeXP(Protocol[_NPT_co]):
    # https://github.com/data-apis/array-api-strict/blob/2.1/array_api_strict/_dtypes.py
    @property
    def _np_dtype(self, /) -> _NPT_co: ...
    def __init__(self, /, np_dtype: _NPT_co) -> None: ...


# putting it all together

_DType: TypeAlias = _NPT | _TorchT | _DTypeXP[_NPT]

Bool: TypeAlias = _DType[_b1_np, _b1_torch]
Int8: TypeAlias = _DType[_i1_np, _i1_torch]
Int16: TypeAlias = _DType[_i2_np, _i2_torch]
Int32: TypeAlias = _DType[_i4_np, _i4_torch]
Int64: TypeAlias = _DType[_i8_np, _i8_torch]
UInt8: TypeAlias = _DType[_u1_np, _u1_torch]
UInt16: TypeAlias = _DType[_u2_np, _u2_torch]
UInt32: TypeAlias = _DType[_u4_np, _u4_torch]
UInt64: TypeAlias = _DType[_u8_np, _u8_torch]
Float16: TypeAlias = _DType[_f2_np, _f2_torch]
Float32: TypeAlias = _DType[_f4_np, _f4_torch]
Float64: TypeAlias = _DType[_f8_np, _f8_torch]
Complex64: TypeAlias = _DType[_c8_np, _c8_torch]
Complex128: TypeAlias = _DType[_c16_np, _c16_torch]

DefaultInt: TypeAlias = _DType[_i0_np, _i0_torch]
DefaultFloat: TypeAlias = _DType[_f0_np, _f0_torch]
DefaultComplex: TypeAlias = _DType[_c0_np, _c0_torch]

SignedInteger: TypeAlias = _DType[_i_np, _i_torch]
UnsignedInteger: TypeAlias = _DType[_u_np, _u_torch]
RealFloating: TypeAlias = _DType[_f_np, _f_torch]
ComplexFloating: TypeAlias = _DType[_c_np, _c_torch]

Integral: TypeAlias = _DType[_iu_np, _iu_torch]
Real: TypeAlias = _DType[_iuf_np, _iuf_torch]
Inexact: TypeAlias = _DType[_fc_np, _fc_torch]
Numeric: TypeAlias = _DType[_iufc_np, _iufc_torch]

DType: TypeAlias = _DType[_DTypeNP, _DTypeTorch]
