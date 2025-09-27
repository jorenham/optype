import ctypes as ct
import sys

import pytest

import optype.numpy as onp


@pytest.mark.parametrize(
    ("name", "ctype"),
    [
        ("Bool", ct.c_bool),
        ("Int8", ct.c_int8),
        ("UInt8", ct.c_uint8),
        ("Int16", ct.c_int16),
        ("UInt16", ct.c_uint16),
        ("Int32", ct.c_int32),
        ("UInt32", ct.c_uint32),
        ("Int64", ct.c_int64),
        ("UInt64", ct.c_uint64),
        ("Byte", ct.c_byte),
        ("UByte", ct.c_ubyte),
        ("Short", ct.c_short),
        ("UShort", ct.c_ushort),
        ("IntC", ct.c_int),
        ("UIntC", ct.c_uint),
        ("Long", ct.c_long),
        ("ULong", ct.c_ulong),
        ("LongLong", ct.c_longlong),
        ("ULongLong", ct.c_ulonglong),
        ("IntP", ct.c_ssize_t),
        ("UIntP", ct.c_size_t),
        ("Float32", ct.c_float),
        ("Float64", ct.c_double),
        ("LongDouble", ct.c_longdouble),
        ("Bytes", ct.c_char),
        ("Object", ct.py_object),
    ],
)
def test_reexports(name: str, ctype: type) -> None:
    assert getattr(onp.ctypeslib, name) is ctype


@pytest.mark.skipif(
    not hasattr(ct, "c_float_complex"),
    reason="requires complex ctypes",
)
@pytest.mark.parametrize(
    ("name_export", "name_orig"),
    [
        ("Complex64", "c_float_complex"),
        ("Complex128", "c_double_complex"),
        ("CLongDouble", "c_longdouble_complex"),
    ],
)
def test_complex_reexports(name_export: str, name_orig: str) -> None:
    assert sys.platform != "win32"  # we use this assumption from typeshed too

    ctype_expect = getattr(ct, name_orig)
    ctype_actual = getattr(onp.ctypeslib, name_export)
    assert ctype_actual is ctype_expect


def test_static_assignability_abstract() -> None:
    # signedinteger
    i_i8: onp.ctypeslib.SignedInteger = onp.ctypeslib.Int8()
    i_i16: onp.ctypeslib.SignedInteger = onp.ctypeslib.Int16()
    i_i32: onp.ctypeslib.SignedInteger = onp.ctypeslib.Int32()
    i_i64: onp.ctypeslib.SignedInteger = onp.ctypeslib.Int64()

    # unsignedinteger
    u_u8: onp.ctypeslib.UnsignedInteger = onp.ctypeslib.UInt8()
    u_u16: onp.ctypeslib.UnsignedInteger = onp.ctypeslib.UInt16()
    u_u32: onp.ctypeslib.UnsignedInteger = onp.ctypeslib.UInt32()
    u_u64: onp.ctypeslib.UnsignedInteger = onp.ctypeslib.UInt64()

    # integer
    iu_i8: onp.ctypeslib.Integer = onp.ctypeslib.Int8()
    iu_u8: onp.ctypeslib.Integer = onp.ctypeslib.UInt8()
    iu_i16: onp.ctypeslib.Integer = onp.ctypeslib.Int16()
    iu_u16: onp.ctypeslib.Integer = onp.ctypeslib.UInt16()
    iu_i32: onp.ctypeslib.Integer = onp.ctypeslib.Int32()
    iu_u32: onp.ctypeslib.Integer = onp.ctypeslib.UInt32()
    iu_i64: onp.ctypeslib.Integer = onp.ctypeslib.Int64()
    iu_u64: onp.ctypeslib.Integer = onp.ctypeslib.UInt64()

    # floating
    f_f32: onp.ctypeslib.Floating = onp.ctypeslib.Float32()
    f_f64: onp.ctypeslib.Floating = onp.ctypeslib.Float64()
    f_f80: onp.ctypeslib.Floating = onp.ctypeslib.LongDouble()

    # inexact
    fc_f32: onp.ctypeslib.Inexact = onp.ctypeslib.Float32()
    fc_f64: onp.ctypeslib.Inexact = onp.ctypeslib.Float64()
    fc_f80: onp.ctypeslib.Inexact = onp.ctypeslib.LongDouble()

    # number
    iufc_i8: onp.ctypeslib.Number = onp.ctypeslib.Int8()
    iufc_u8: onp.ctypeslib.Number = onp.ctypeslib.UInt8()
    iufc_i16: onp.ctypeslib.Number = onp.ctypeslib.Int16()
    iufc_u16: onp.ctypeslib.Number = onp.ctypeslib.UInt16()
    iufc_i32: onp.ctypeslib.Number = onp.ctypeslib.Int32()
    iufc_u32: onp.ctypeslib.Number = onp.ctypeslib.UInt32()
    iufc_i64: onp.ctypeslib.Number = onp.ctypeslib.Int64()
    iufc_u64: onp.ctypeslib.Number = onp.ctypeslib.UInt64()
    iufc_f32: onp.ctypeslib.Number = onp.ctypeslib.Float32()
    iufc_f64: onp.ctypeslib.Number = onp.ctypeslib.Float64()
    iufc_f80: onp.ctypeslib.Number = onp.ctypeslib.LongDouble()

    # flexible
    s_s: onp.ctypeslib.Flexible = onp.ctypeslib.Bytes()
    # Void is a TypeAliasType, so not instantiable
