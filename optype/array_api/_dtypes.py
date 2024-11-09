# ruff: noqa: PYI042
from __future__ import annotations

import sys
from typing import Protocol, TypeAlias, runtime_checkable

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


# numpy

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _DTypeNP(Protocol[_T_co]):
    @property
    def type(self) -> type[_T_co]: ...


# pytorch


@runtime_checkable
class _DTypeTorch(Protocol):
    # the torch stubs don't use literals, so we can't distinguish between dtypes
    # https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_C/__init__.pyi.in#L181-L189
    @property
    def itemsize(self, /) -> int: ...
    @property
    def is_signed(self, /) -> bool: ...
    @property
    def is_floating_point(self, /) -> bool: ...
    @property
    def is_complex(self, /) -> bool: ...


# array-api-strict


@runtime_checkable
class _DTypeXP(Protocol[_T_co]):
    # https://github.com/data-apis/array-api-strict/blob/2.1/array_api_strict/_dtypes.py
    @property
    def _np_dtype(self, /) -> _T_co: ...
    def __init__(self, /, np_dtype: _T_co) -> None: ...


# putting it all together

_DType: TypeAlias = _DTypeNP[_T] | _DTypeXP[_DTypeNP[_T]] | _DTypeTorch

Bool: TypeAlias = _DType[_sct_np.Bool]
Int8: TypeAlias = _DType[_sct_np.Integer]
Int16: TypeAlias = _DType[_sct_np.Integer]
Int32: TypeAlias = _DType[_sct_np.Integer]
Int64: TypeAlias = _DType[_sct_np.Integer]
UInt8: TypeAlias = _DType[_sct_np.Integer]
UInt16: TypeAlias = _DType[_sct_np.Integer]
UInt32: TypeAlias = _DType[_sct_np.Integer]
UInt64: TypeAlias = _DType[_sct_np.Integer]
Float16: TypeAlias = _DType[_sct_np.Floating]
Float32: TypeAlias = _DType[_sct_np.Floating]
Float64: TypeAlias = _DType[_sct_np.Float64]
Complex64: TypeAlias = _DType[_sct_np.ComplexFloating]
Complex128: TypeAlias = _DType[_sct_np.Complex128]

Int0: TypeAlias = _DType[_sct_np.Integer]
Float0: TypeAlias = _DType[_sct_np.Float64]
Complex0: TypeAlias = _DType[_sct_np.Complex128]

Integer: TypeAlias = _DType[_sct_np.Integer]
RealFloating: TypeAlias = _DType[_sct_np.Floating]
ComplexFloating: TypeAlias = _DType[_sct_np.ComplexFloating]

Integer_co: TypeAlias = _DType[_sct_np.Integer_co]
Real: TypeAlias = _DType[_sct_np.Floating_co]
Numeric: TypeAlias = _DType[_sct_np.Number]

DType: TypeAlias = _DType[_sct_np.ComplexFloating_co]
