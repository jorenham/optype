# pyright: reportPrivateUsage=false
import sys
from typing import Any, TypeAlias

import array_api_strict as xp  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

import optype.array_api as oxp  # noqa: TCH001
from optype.array_api._dtypes import _DTypeNP, _DTypeTorch, _DTypeXP


_AnyDTypeNP: TypeAlias = _DTypeNP[Any]


@pytest.mark.skipif(
    sys.platform != "linux" and sys.version_info >= (3, 13),
    reason="pytorch 1.5.1 on python 3.13 requires linux",
)
@pytest.mark.parametrize(
    "dtype_name",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_torch(dtype_name: str) -> None:
    import torch  # noqa: PLC0415

    dtype: torch.dtype = getattr(torch, dtype_name)
    _dtype: _DTypeTorch = dtype
    assert isinstance(dtype, _DTypeTorch)


def test_bool() -> None:
    b1: oxp.Bool = np.dtype(np.bool_)
    b1_biufc: _AnyDTypeNP = b1

    assert isinstance(b1, _DTypeNP)
    assert isinstance(xp.bool, _DTypeXP)


def test_int8() -> None:
    i1: oxp.Int8 = np.dtype(np.int8)
    i1_iu: oxp.Integer_co = i1
    i1_iuf: oxp.Real = i1
    i1_iufc: oxp.Numeric = i1
    i1_biufc: oxp.DType = i1

    assert isinstance(i1, _DTypeNP)
    assert isinstance(xp.int8, _DTypeXP)


def test_int16() -> None:
    i2: oxp.Int16 = np.dtype(np.int16)
    i2_iu: oxp.Integer_co = i2
    i2_iuf: oxp.Real = i2
    i2_iufc: oxp.Numeric = i2
    i2_biufc: oxp.DType = i2
    assert isinstance(i2, _DTypeNP)
    assert isinstance(xp.int16, _DTypeXP)


def test_int32() -> None:
    i4: oxp.Int32 = np.dtype(np.int32)
    i4_i0: oxp.Int0 = i4
    i4_iu: oxp.Integer_co = i4
    i4_iuf: oxp.Real = i4
    i4_iufc: oxp.Numeric = i4
    i4_biufc: oxp.DType = i4
    assert isinstance(i4, _DTypeNP)
    assert isinstance(xp.int32, _DTypeXP)


def test_int64() -> None:
    i8: oxp.Int64 = np.dtype(np.int64)
    i8_i0: oxp.Int0 = i8
    i8_iu: oxp.Integer_co = i8
    i8_iuf: oxp.Real = i8
    i8_iufc: oxp.Numeric = i8
    i8_biufc: oxp.DType = i8
    assert isinstance(i8, _DTypeNP)
    assert isinstance(xp.int64, _DTypeXP)


def test_uint8() -> None:
    u1: oxp.UInt8 = np.dtype(np.uint16)
    u1_iu: oxp.Integer_co = u1
    u1_iuf: oxp.Real = u1
    u1_iufc: oxp.Numeric = u1
    u1_biufc: oxp.DType = u1
    assert isinstance(u1, _DTypeNP)
    assert isinstance(xp.uint8, _DTypeXP)


def test_uint16() -> None:
    u2: oxp.UInt16 = np.dtype(np.uint16)
    u2_iu: oxp.Integer_co = u2
    u2_iuf: oxp.Real = u2
    u2_iufc: oxp.Numeric = u2
    u2_biufc: oxp.DType = u2
    assert isinstance(u2, _DTypeNP)
    assert isinstance(xp.uint16, _DTypeXP)


def test_uint32() -> None:
    u4: oxp.UInt32 = np.dtype(np.uint32)
    u4_iu: oxp.Integer_co = u4
    u4_iuf: oxp.Real = u4
    u4_iufc: oxp.Numeric = u4
    u4_biufc: oxp.DType = u4
    assert isinstance(u4, _DTypeNP)
    assert isinstance(xp.uint32, _DTypeXP)


def test_uint64() -> None:
    u8: oxp.UInt64 = np.dtype(np.uint64)
    u8_iu: oxp.Integer_co = u8
    u8_iuf: oxp.Real = u8
    u8_iufc: oxp.Numeric = u8
    u8_biufc: oxp.DType = u8
    assert isinstance(u8, _DTypeNP)
    assert isinstance(xp.uint64, _DTypeXP)


def test_float32() -> None:
    f4: oxp.Float32 = np.dtype(np.float32)
    f4_f: oxp.RealFloating = f4
    f4_iuf: oxp.Real = f4
    f4_iufc: oxp.Numeric = f4
    f4_biufc: oxp.DType = f4
    assert isinstance(f4, _DTypeNP)
    assert isinstance(xp.float32, _DTypeXP)


def test_float64() -> None:
    f8: oxp.Float64 = np.dtype(np.float64)
    f8_f0: oxp.Float0 = f8
    f8_f: oxp.RealFloating = f8
    f8_iuf: oxp.Real = f8
    f8_iufc: oxp.Numeric = f8
    f8_biufc: oxp.DType = f8
    assert isinstance(f8, _DTypeNP)
    assert isinstance(xp.float64, _DTypeXP)


def test_complex64() -> None:
    c8: oxp.Complex64 = np.dtype(np.complex64)
    c8_c: oxp.ComplexFloating = c8
    c8_iufc: oxp.Numeric = c8
    c8_biufc: oxp.DType = c8
    assert isinstance(c8, _DTypeNP)
    assert isinstance(xp.complex64, _DTypeXP)


def test_complex128() -> None:
    c16: oxp.Complex128 = np.dtype(np.complex128)
    c16_c0: oxp.Complex0 = c16
    c16_c: oxp.ComplexFloating = c16
    c16_iufc: oxp.Numeric = c16
    c16_biufc: oxp.DType = c16
    assert isinstance(c16, _DTypeNP)
    assert isinstance(xp.complex128, _DTypeXP)
