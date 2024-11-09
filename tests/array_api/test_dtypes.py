# pyright: reportPrivateUsage=false
from typing import Any, TypeAlias

import numpy as np
import pytest
import torch

import optype.array_api as oxp  # noqa: TCH001
from optype.array_api._dtypes import _DTypeNP, _DTypeTorch


_AnyDTypeNP: TypeAlias = _DTypeNP[Any]


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_torch(dtype: torch.dtype) -> None:
    _dtype: _DTypeTorch = dtype
    assert isinstance(dtype, _DTypeTorch)


def test_bool_np() -> None:
    b1: oxp.Bool = np.dtype(np.bool_)
    b1_biufc: _AnyDTypeNP = b1
    assert isinstance(b1, _DTypeNP)


def test_int8_np() -> None:
    i1: oxp.Int8 = np.dtype(np.int8)
    i1_iu: oxp.Integer_co = i1
    i1_iuf: oxp.Real = i1
    i1_iufc: oxp.Numeric = i1
    i1_biufc: oxp.DType = i1
    assert isinstance(i1, _DTypeNP)


def test_int16_np() -> None:
    i2: oxp.Int16 = np.dtype(np.int16)
    i2_iu: oxp.Integer_co = i2
    i2_iuf: oxp.Real = i2
    i2_iufc: oxp.Numeric = i2
    i2_biufc: oxp.DType = i2
    assert isinstance(i2, _DTypeNP)


def test_int32_np() -> None:
    i4: oxp.Int32 = np.dtype(np.int32)
    i4_i0: oxp.Int0 = i4
    i4_iu: oxp.Integer_co = i4
    i4_iuf: oxp.Real = i4
    i4_iufc: oxp.Numeric = i4
    i4_biufc: oxp.DType = i4
    assert isinstance(i4, _DTypeNP)


def test_int64_np() -> None:
    i8: oxp.Int64 = np.dtype(np.int64)
    i8_i0: oxp.Int0 = i8
    i8_iu: oxp.Integer_co = i8
    i8_iuf: oxp.Real = i8
    i8_iufc: oxp.Numeric = i8
    i8_biufc: oxp.DType = i8
    assert isinstance(i8, _DTypeNP)


def test_uint8_np() -> None:
    u1: oxp.UInt8 = np.dtype(np.uint16)
    u1_iu: oxp.Integer_co = u1
    u1_iuf: oxp.Real = u1
    u1_iufc: oxp.Numeric = u1
    u1_biufc: oxp.DType = u1
    assert isinstance(u1, _DTypeNP)


def test_uint16_np() -> None:
    u2: oxp.UInt16 = np.dtype(np.uint16)
    u2_iu: oxp.Integer_co = u2
    u2_iuf: oxp.Real = u2
    u2_iufc: oxp.Numeric = u2
    u2_biufc: oxp.DType = u2
    assert isinstance(u2, _DTypeNP)


def test_uint32_np() -> None:
    u4: oxp.UInt32 = np.dtype(np.uint32)
    u4_iu: oxp.Integer_co = u4
    u4_iuf: oxp.Real = u4
    u4_iufc: oxp.Numeric = u4
    u4_biufc: oxp.DType = u4
    assert isinstance(u4, _DTypeNP)


def test_uint64_np() -> None:
    u8: oxp.UInt64 = np.dtype(np.uint64)
    u8_iu: oxp.Integer_co = u8
    u8_iuf: oxp.Real = u8
    u8_iufc: oxp.Numeric = u8
    u8_biufc: oxp.DType = u8
    assert isinstance(u8, _DTypeNP)


def test_float32_np() -> None:
    f4: oxp.Float32 = np.dtype(np.float32)
    f4_f: oxp.RealFloating = f4
    f4_iuf: oxp.Real = f4
    f4_iufc: oxp.Numeric = f4
    f4_biufc: oxp.DType = f4
    assert isinstance(f4, _DTypeNP)


def test_float64_np() -> None:
    f8: oxp.Float64 = np.dtype(np.float64)
    f8_f0: oxp.Float0 = f8
    f8_f: oxp.RealFloating = f8
    f8_iuf: oxp.Real = f8
    f8_iufc: oxp.Numeric = f8
    f8_biufc: oxp.DType = f8
    assert isinstance(f8, _DTypeNP)


def test_complex64_np() -> None:
    c8: oxp.Complex64 = np.dtype(np.complex64)
    c8_c: oxp.ComplexFloating = c8
    c8_iufc: oxp.Numeric = c8
    c8_biufc: oxp.DType = c8
    assert isinstance(c8, _DTypeNP)


def test_complex128_np() -> None:
    c16: oxp.Complex128 = np.dtype(np.complex128)
    c16_c0: oxp.Complex0 = c16
    c16_c: oxp.ComplexFloating = c16
    c16_iufc: oxp.Numeric = c16
    c16_biufc: oxp.DType = c16
    assert isinstance(c16, _DTypeNP)
