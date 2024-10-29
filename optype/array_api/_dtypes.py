# ruff: noqa: N801
# mypy: disable-error-code="no-any-explicit, no-any-decorated"
from __future__ import annotations

import sys
from typing import (  # noqa: N817
    Any,
    Literal as L,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from optype import CanBool, CanComplex, CanFloat, CanIndex


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar


__all__ = [  # noqa: RUF022
    "Bool", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
    "Float16", "Float32", "Float64", "Complex64", "Complex128",
    "SignedInteger", "UnsignedInteger", "Integer", "RealFloating", "ComplexFloating",
    "Numeric",
]  # fmt: skip

#
# numpy
#

_KindT_co = TypeVar("_KindT_co", bound=L["b", "i", "u", "f", "c"], covariant=True)
_CharT_co = TypeVar("_CharT_co", bound=str, covariant=True)
_NameT_co = TypeVar("_NameT_co", bound=str, covariant=True)
_ItemN_co = TypeVar("_ItemN_co", bound=int, covariant=True)
_DataN_co = TypeVar("_DataN_co", bound=int, covariant=True)
_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _DType_np(Protocol[_KindT_co, _CharT_co, _NameT_co, _ItemN_co, _DataN_co, _T_co]):
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
    @property
    def ndim(self) -> L[0]: ...
    @property
    def shape(self) -> tuple[()]: ...


# These correspond to the `numpy.dtypes.{}DType` data types
_Bool_np: TypeAlias = _DType_np[L["b"], L["?"], L["bool"], L[1], L[1], CanBool]
_Int8_np: TypeAlias = _DType_np[L["i"], L["b"], L["int8"], L[1], L[1], CanIndex]
_Int16_np: TypeAlias = _DType_np[L["i"], L["h"], L["int16"], L[2], L[2], CanIndex]
_Int32_np: TypeAlias = _DType_np[L["i"], L["i", "l"], L["int32"], L[4], L[4], CanIndex]
_Int64_np: TypeAlias = _DType_np[L["i"], L["l", "q"], L["int64"], L[8], L[8], CanIndex]
_UInt8_np: TypeAlias = _DType_np[L["u"], L["B"], L["uint8"], L[1], L[1], CanIndex]
_UInt16_np: TypeAlias = _DType_np[L["u"], L["H"], L["uint16"], L[2], L[2], CanIndex]
_UInt32_np: TypeAlias = _DType_np[
    L["u"], L["I", "L"], L["uint32"], L[4], L[4], CanIndex
]
_UInt64_np: TypeAlias = _DType_np[
    L["u"], L["L", "Q"], L["uint64"], L[8], L[8], CanIndex
]
# `float16` isn't part of the array-api spec, but the array-api does allow using it
_Float16_np: TypeAlias = _DType_np[L["f"], L["e"], L["float16"], L[2], L[2], CanFloat]
_Float32_np: TypeAlias = _DType_np[L["f"], L["f"], L["float32"], L[4], L[4], CanFloat]
_Float64_np: TypeAlias = _DType_np[L["f"], L["d"], L["float64"], L[8], L[8], float]
_Complex64_np: TypeAlias = _DType_np[
    L["c"], L["F"], L["complex64"], L[8], L[4], CanComplex
]
_Complex128_np: TypeAlias = _DType_np[
    L["c"], L["D"], L["complex128"], L[16], L[8], complex
]

# the `kind` of array-api dtypes, as used in `isdtype()`
# https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype
_SInt_np: TypeAlias = _Int8_np | _Int16_np | _Int32_np | _Int64_np
_UInt_np: TypeAlias = _UInt8_np | _UInt16_np | _UInt32_np | _UInt64_np
_Int_np: TypeAlias = _SInt_np | _UInt_np
_Float_np: TypeAlias = _Float16_np | _Float32_np | _Float64_np
_Complex_np: TypeAlias = _Complex64_np | _Complex128_np
_Number_np: TypeAlias = _Int_np | _Float_np | _Complex_np

#
# array-api-strict
#

_NPT = TypeVar("_NPT", bound=_DType_np[Any, Any, Any, Any, Any, Any])
_NPT_co = TypeVar(
    "_NPT_co",
    covariant=True,
    bound=_DType_np[Any, Any, Any, Any, Any, Any],
)


# https://github.com/data-apis/array-api-strict/blob/2.1/array_api_strict/_dtypes.py
@runtime_checkable
class _DType_xp(Protocol[_NPT_co]):
    @property
    def _np_dtype(self, /) -> _NPT_co: ...
    def __init__(self, /, np_dtype: _NPT_co) -> None: ...


#
# sparse, with backends: MLIR & finch (julia wrapper)
#

_T_contra = TypeVar("_T_contra", contravariant=True, default=_T_co)
_WidthN_co = TypeVar("_WidthN_co", bound=int, covariant=True)


# https://github.com/pydata/sparse/blob/cc3c8d9/sparse/mlir_backend/_dtypes.py
@runtime_checkable
class _DType_ir(Protocol[_WidthN_co, _NPT_co]):
    @property
    def bit_width(self, /) -> _WidthN_co: ...
    @property
    def np_dtype(self, /) -> _NPT_co: ...


# this is a highly dynamic and untyped wrapper around a julia type value (but it's not
# a python type); so there's not much we can do here
@runtime_checkable
class _DType_jl(Protocol[_NameT_co, _T_co, _T_contra]):
    @property
    def __name__(self, /) -> _NameT_co: ...
    def __call__(self, x: _T_contra, /) -> _T_co: ...


__NameSignedInt: TypeAlias = L["Int8", "Int16", "Int32", "Int64"]
__NameUnsignedInt: TypeAlias = L["UInt8", "UInt16", "UInt32", "UInt64"]
__NameInt: TypeAlias = L[__NameSignedInt, __NameUnsignedInt]
__NameFloat: TypeAlias = L["Float16", "Float32", "Float64"]
__NameComplex: TypeAlias = L["Complex"]  # julia's `ComplexF{16,32,64}` have same names

_Bool_sparse: TypeAlias = _DType_jl[L["Bool"], bool]
_Int8_sparse: TypeAlias = _DType_ir[L[8], _Int8_np] | _DType_jl[L["Int8"], int]
_Int16_sparse: TypeAlias = _DType_ir[L[16], _Int16_np] | _DType_jl[L["Int16"], int]
_Int32_sparse: TypeAlias = _DType_ir[L[32], _Int32_np] | _DType_jl[L["Int32"], int]
_Int64_sparse: TypeAlias = _DType_ir[L[64], _Int64_np] | _DType_jl[L["Int64"], int]
_UInt8_sparse: TypeAlias = _DType_ir[L[8], _UInt8_np] | _DType_jl[L["UInt8"], int]
_UInt16_sparse: TypeAlias = _DType_ir[L[16], _UInt16_np] | _DType_jl[L["UInt16"], int]
_UInt32_sparse: TypeAlias = _DType_ir[L[32], _UInt32_np] | _DType_jl[L["UInt32"], int]
_UInt64_sparse: TypeAlias = _DType_ir[L[64], _UInt64_np] | _DType_jl[L["UInt64"], int]
_Float16_sparse: TypeAlias = (
    _DType_ir[L[16], _Float16_np] | _DType_jl[L["Float16"], float]
)
_Float32_sparse: TypeAlias = (
    _DType_ir[L[32], _Float32_np] | _DType_jl[L["Float32"], float]
)
_Float64_sparse: TypeAlias = (
    _DType_ir[L[64], _Float64_np] | _DType_jl[L["Float64"], float]
)
_Complex64_sparse: TypeAlias = (
    _DType_ir[L[64], _Complex64_np] | _DType_jl[__NameComplex, complex]  # jl.ComplexF32
)
_Complex128_sparse: TypeAlias = (
    _DType_ir[L[128], _Complex128_np]
    | _DType_jl[__NameComplex, complex]  # jl.ComplexF64
)

__NBitInt: TypeAlias = L[8, 16, 32, 64]
_SInt_sparse: TypeAlias = _DType_ir[__NBitInt, _SInt_np] | _DType_jl[__NameInt, int]
_UInt_sparse: TypeAlias = (
    _DType_ir[__NBitInt, _UInt_np] | _DType_jl[__NameUnsignedInt, int]
)
_Int_sparse: TypeAlias = _DType_ir[__NBitInt, _Int_np] | _DType_jl[__NameInt, int]
_Float_sparse: TypeAlias = (
    _DType_ir[L[16, 32, 64], _Float_np] | _DType_jl[__NameFloat, float]
)
_Complex_sparse: TypeAlias = (
    _DType_ir[L[64, 128], _Complex_np] | _DType_jl[__NameComplex, complex]
)
_Number_sparse: TypeAlias = _Int_sparse | _Float_sparse | _Complex_sparse

#
# pytorch
#

_SignB_co = TypeVar("_SignB_co", covariant=True, bound=bool, default=bool)
_FloatB_co = TypeVar("_FloatB_co", covariant=True, bound=bool, default=bool)
_ComplexB_co = TypeVar("_ComplexB_co", covariant=True, bound=bool, default=bool)


# https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_C/__init__.pyi.in#L181-L189
class _DType_torch(Protocol[_ItemN_co, _SignB_co, _FloatB_co, _ComplexB_co]):
    @property
    def itemsize(self, /) -> _ItemN_co: ...
    @property
    def is_signed(self, /) -> _SignB_co: ...
    @property
    def is_floating_point(self, /) -> _FloatB_co: ...
    @property
    def is_complex(self, /) -> _ComplexB_co: ...


_F: TypeAlias = L[False]
_T: TypeAlias = L[True]

_Bool_torch: TypeAlias = _DType_torch[L[1], _F, _F, _F]
_Int8_torch: TypeAlias = _DType_torch[L[1], _T, _F, _F]
_Int16_torch: TypeAlias = _DType_torch[L[2], _T, _F, _F]
_Int32_torch: TypeAlias = _DType_torch[L[4], _T, _F, _F]
_Int64_torch: TypeAlias = _DType_torch[L[8], _T, _F, _F]
_UInt8_torch: TypeAlias = _DType_torch[L[1], _F, _F, _F]
_UInt16_torch: TypeAlias = _DType_torch[L[2], _F, _F, _F]
_UInt32_torch: TypeAlias = _DType_torch[L[4], _F, _F, _F]
_UInt64_torch: TypeAlias = _DType_torch[L[8], _F, _F, _F]
_Float16_torch: TypeAlias = _DType_torch[L[2], _T, _T, _F]
_Float32_torch: TypeAlias = _DType_torch[L[4], _T, _T, _F]
_Float64_torch: TypeAlias = _DType_torch[L[8], _T, _T, _F]
_Complex64_torch: TypeAlias = _DType_torch[L[8], _T, _F, _T]
_Complex128_torch: TypeAlias = _DType_torch[L[16], _T, _F, _T]

_SInt_torch: TypeAlias = _DType_torch[L[1, 2, 4, 8], _T, _F, _F]
_UInt_torch: TypeAlias = _DType_torch[L[1, 2, 4, 8], _F, _F, _F]
_Int_torch: TypeAlias = _DType_torch[L[1, 2, 4, 8], bool, _F, _F]
_Float_torch: TypeAlias = _DType_torch[L[2, 4, 8], _T, _T, _F]
_Complex_torch: TypeAlias = _DType_torch[L[8, 16], _T, _F, _T]
_Number_torch: TypeAlias = _Int_torch | _Float_torch | _Complex_torch

#
# putting it all together
#

_DType_xnp: TypeAlias = _NPT | _DType_xp[_NPT]

Bool: TypeAlias = _DType_xnp[_Bool_np] | _Bool_sparse | _Bool_torch
Int8: TypeAlias = _DType_xnp[_Int8_np] | _Int8_sparse | _Int8_torch
Int16: TypeAlias = _DType_xnp[_Int16_np] | _Int16_sparse | _Int16_torch
Int32: TypeAlias = _DType_xnp[_Int32_np] | _Int32_sparse | _Int32_torch
Int64: TypeAlias = _DType_xnp[_Int64_np] | _Int64_sparse | _Int64_torch
UInt8: TypeAlias = _DType_xnp[_UInt8_np] | _UInt8_sparse | _UInt8_torch
UInt16: TypeAlias = _DType_xnp[_UInt16_np] | _UInt16_sparse | _UInt16_torch
UInt32: TypeAlias = _DType_xnp[_UInt32_np] | _UInt32_sparse | _UInt32_torch
UInt64: TypeAlias = _DType_xnp[_UInt64_np] | _UInt64_sparse | _UInt64_torch
Float16: TypeAlias = _DType_xnp[_Float16_np] | _Float16_sparse | _Float16_torch
Float32: TypeAlias = _DType_xnp[_Float32_np] | _Float32_sparse | _Float32_torch
Float64: TypeAlias = _DType_xnp[_Float64_np] | _Float64_sparse | _Float64_torch
Complex64: TypeAlias = _DType_xnp[_Complex64_np] | _Complex64_sparse | _Complex64_torch
Complex128: TypeAlias = (
    _DType_xnp[_Complex128_np] | _Complex128_sparse | _Complex128_torch
)

SignedInteger: TypeAlias = _DType_xnp[_SInt_np] | _SInt_sparse | _SInt_torch
UnsignedInteger: TypeAlias = _DType_xnp[_UInt_np] | _UInt_sparse | _UInt_torch
Integer: TypeAlias = _DType_xnp[_Int_np] | _Int_sparse | _Int_torch
RealFloating: TypeAlias = _DType_xnp[_Float_np] | _Float_sparse | _Float_torch
ComplexFloating: TypeAlias = _DType_xnp[_Complex_np] | _Complex_sparse | _Complex_torch
Numeric: TypeAlias = _DType_xnp[_Number_np] | _Number_sparse | _Number_torch
DType: TypeAlias = Bool | Numeric
