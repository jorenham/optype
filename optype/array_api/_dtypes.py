# ruff: noqa: PYI042
from __future__ import annotations

import sys
from typing import Literal as L, Protocol, TypeAlias, runtime_checkable  # noqa: N817

# this weird import prevents a circular import (according to pyright)
import optype.array_api._sctypes_numpy as _sct_np


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar


__all__ = [  # noqa: RUF022
    "DType",
    "Numeric",
    "Integer_co",
    "Real",
    "ComplexFloating",
    "RealFloating",
    "Bool",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float16", "Float32", "Float64",
    "Complex64", "Complex128",
    "Int0", "Float0", "Complex0",
]  # fmt: skip


_1: TypeAlias = L[1]
_2: TypeAlias = L[2]
_4: TypeAlias = L[4]
_8: TypeAlias = L[8]
_16: TypeAlias = L[16]

_NByte_i: TypeAlias = L[1, 2, 4, 8]
_NByte_f: TypeAlias = L[2, 4, 8]
_NByte_c: TypeAlias = L[8, 16]
_NByte: TypeAlias = L[1, 2, 4, 8, 16]

# numpy

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _DTypeNP(Protocol[_T_co]):
    @property
    def type(self) -> type[_T_co]: ...


# pytorch

_ItemN_co = TypeVar("_ItemN_co", covariant=True, bound=int)
_SignB_co = TypeVar("_SignB_co", covariant=True, bound=bool)
_FloatB_co = TypeVar("_FloatB_co", covariant=True, bound=bool)
_ComplexB_co = TypeVar("_ComplexB_co", covariant=True, bound=bool)


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


_AnyDTypeTorch: TypeAlias = _DTypeTorch[_NByte, bool, bool, bool]
_TorchT = TypeVar("_TorchT", bound=_AnyDTypeTorch)

_B0: TypeAlias = L[False]
_B1: TypeAlias = L[True]

_b1_torch: TypeAlias = _DTypeTorch[_1, _B0, _B0, _B0]
_i1_torch: TypeAlias = _DTypeTorch[_1, _B1, _B0, _B0]
_i2_torch: TypeAlias = _DTypeTorch[_2, _B1, _B0, _B0]
_i4_torch: TypeAlias = _DTypeTorch[_4, _B1, _B0, _B0]
_i8_torch: TypeAlias = _DTypeTorch[_8, _B1, _B0, _B0]
_u1_torch: TypeAlias = _DTypeTorch[_1, _B0, _B0, _B0]
_u2_torch: TypeAlias = _DTypeTorch[_2, _B0, _B0, _B0]
_u4_torch: TypeAlias = _DTypeTorch[_4, _B0, _B0, _B0]
_u8_torch: TypeAlias = _DTypeTorch[_8, _B0, _B0, _B0]
_f2_torch: TypeAlias = _DTypeTorch[_2, _B1, _B1, _B0]
_f4_torch: TypeAlias = _DTypeTorch[_4, _B1, _B1, _B0]
_f8_torch: TypeAlias = _DTypeTorch[_8, _B1, _B1, _B0]
_c8_torch: TypeAlias = _DTypeTorch[_8, _B1, _B0, _B1]
_c16_torch: TypeAlias = _DTypeTorch[_16, _B1, _B0, _B1]


_f_torch: TypeAlias = _DTypeTorch[_NByte_f, _B1, _B1, _B0]
_c_torch: TypeAlias = _DTypeTorch[_NByte_c, _B1, _B0, _B1]
_biu_torch: TypeAlias = _DTypeTorch[_NByte_i, bool, _B0, _B0]
_biuf_torch: TypeAlias = _DTypeTorch[_NByte_i, bool, bool, _B0]
_iufc_torch: TypeAlias = _AnyDTypeTorch

_i0_torch: TypeAlias = _DTypeTorch[L[4, 8], _B1, _B0, _B0]
_f0_torch: TypeAlias = _DTypeTorch[L[4, 8], _B1, _B1, _B0]
_c0_torch: TypeAlias = _c_torch

# array-api-strict


@runtime_checkable
class _DTypeXP(Protocol[_T_co]):
    # https://github.com/data-apis/array-api-strict/blob/2.1/array_api_strict/_dtypes.py
    @property
    def _np_dtype(self, /) -> _T_co: ...
    def __init__(self, /, np_dtype: _T_co) -> None: ...


# putting it all together

_DType: TypeAlias = _DTypeNP[_T] | _TorchT | _DTypeXP[_DTypeNP[_T]]

Bool: TypeAlias = _DType[_sct_np.Bool, _b1_torch]
Int8: TypeAlias = _DType[_sct_np.Integer, _i1_torch]
Int16: TypeAlias = _DType[_sct_np.Integer, _i2_torch]
Int32: TypeAlias = _DType[_sct_np.Integer, _i4_torch]
Int64: TypeAlias = _DType[_sct_np.Integer, _i8_torch]
UInt8: TypeAlias = _DType[_sct_np.Integer, _u1_torch]
UInt16: TypeAlias = _DType[_sct_np.Integer, _u2_torch]
UInt32: TypeAlias = _DType[_sct_np.Integer, _u4_torch]
UInt64: TypeAlias = _DType[_sct_np.Integer, _u8_torch]
Float16: TypeAlias = _DType[_sct_np.Floating, _f2_torch]
Float32: TypeAlias = _DType[_sct_np.Floating, _f4_torch]
Float64: TypeAlias = _DType[_sct_np.Float64, _f8_torch]
Complex64: TypeAlias = _DType[_sct_np.ComplexFloating, _c8_torch]
Complex128: TypeAlias = _DType[_sct_np.Complex128, _c16_torch]

Int0: TypeAlias = _DType[_sct_np.Integer, _i0_torch]
Float0: TypeAlias = _DType[_sct_np.Float64, _f0_torch]
Complex0: TypeAlias = _DType[_sct_np.Complex128, _c0_torch]

Integer: TypeAlias = _DType[_sct_np.Integer, _biu_torch]
RealFloating: TypeAlias = _DType[_sct_np.Floating, _f_torch]
ComplexFloating: TypeAlias = _DType[_sct_np.ComplexFloating, _c_torch]

Integer_co: TypeAlias = _DType[_sct_np.Integer_co, _biu_torch]
Real: TypeAlias = _DType[_sct_np.Floating_co, _biuf_torch]
Numeric: TypeAlias = _DType[_sct_np.Number, _iufc_torch]

DType: TypeAlias = _DType[_sct_np.ComplexFloating_co, _AnyDTypeTorch]
