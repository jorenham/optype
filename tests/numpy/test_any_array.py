# ruff: noqa: F841, PYI042, PLC2701

import ctypes as ct
from typing import Any, Literal, TypeAlias

import numpy as np
import pytest

import optype.numpy as onp


_0D: TypeAlias = tuple[()]
_1D: TypeAlias = tuple[Literal[1]]
_2D: TypeAlias = tuple[Literal[1], Literal[1]]


# All allowed arguments that when passed to `np.array`, will result in an
# array of the specified scalar type(s).

# fmt: off
_UNSIGNED_INTEGERS_NP = (
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong,
    *((np.ulong,) if hasattr(np, 'ulong') else ()),
)
_UNSIGNED_INTEGERS_CT = (
    ct.c_uint8, ct.c_uint16, ct.c_uint32, ct.c_uint64, ct.c_size_t,
    ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
)
UNSIGNED_INTEGERS: tuple[type[onp.AnyUnsignedInteger], ...] = (
    *_UNSIGNED_INTEGERS_NP,
    *_UNSIGNED_INTEGERS_CT,
)
_SIGNED_INTEGERS_NP = (
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.byte, np.short, np.intc, np.int_, np.longlong,
    *((np.long,) if hasattr(np, 'ulong') else ()),
)
_SIGNED_INTEGERS_CT = (
    ct.c_int8, ct.c_int16, ct.c_int32, ct.c_int64, ct.c_ssize_t,
    ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
)
SIGNED_INTEGERS: tuple[type[onp.AnySignedInteger], ...] = (
    *_SIGNED_INTEGERS_NP,
    *_SIGNED_INTEGERS_CT,
)
INTEGERS: tuple[type[onp.AnyInteger], ...] = (
    *UNSIGNED_INTEGERS,
    *SIGNED_INTEGERS,
)
_FLOATING_NP = (
    np.float16, np.float32, np.float64,
    np.half, np.single, np.double, np.longdouble,
)
_FLOATING_CT = ct.c_float, ct.c_double
FLOATING: tuple[type[onp.AnyFloating], ...] = *_FLOATING_NP, *_FLOATING_CT
COMPLEX_FLOATING: tuple[type[onp.AnyComplexFloating], ...] = (
    np.complex64, np.complex128,
    np.csingle, np.cdouble, np.clongdouble,
)
DATETIME64: tuple[type[onp.AnyDateTime64]] = (np.datetime64,)
TIMEDELTA64: tuple[type[onp.AnyTimeDelta64]] = (np.timedelta64,)
STR: tuple[type[onp.AnyStr], ...] = np.str_, str
BYTES: tuple[type[onp.AnyBytes], ...] = np.bytes_, ct.c_char, bytes
CHARACTER: tuple[type[onp.AnyCharacter], ...] = *STR, *BYTES
VOID: tuple[type[onp.AnyVoid]] = (np.void,)
FLEXIBLE: tuple[type[onp.AnyFlexible], ...] = *VOID, *CHARACTER
BOOL: tuple[type[onp.AnyBool], ...] = np.bool_, ct.c_bool, bool
OBJECT: tuple[type[onp.AnyObject], ...] = np.object_, ct.py_object, object
# fmt: on


def test_any_array() -> None:
    type_np: type[np.integer[Any]] = np.int16
    v = 42

    # scalar

    v_py: onp.AnyArray[_0D] = v
    assert np.shape(v_py) == ()

    v_np: onp.AnyArray[_0D] = type_np(v_py)
    assert int(v_np) == v_py
    assert np.shape(v_np) == ()

    # 0d

    x0_np: onp.AnyArray[_0D] = np.array(v_py)
    assert np.shape(x0_np) == ()

    # 1d

    x1_py: onp.AnyArray[_1D] = [v_py]
    assert np.shape(x1_py) == (1,)

    x1_py_np: onp.AnyArray[_1D] = [x0_np]
    assert np.shape(x1_py_np) == (1,)

    x1_np: onp.AnyArray[_1D] = np.array(x1_py)
    assert np.shape(x1_np) == (1,)

    # 2d

    x2_py: onp.AnyArray[_2D] = [x1_py]
    assert np.shape(x2_py) == (1, 1)

    x2_py_np: onp.AnyArray[_2D] = [x1_np]
    assert np.shape(x2_py_np) == (1, 1)

    x2_np: onp.AnyArray[_2D] = np.array(x2_py)
    assert np.shape(x2_np) == (1, 1)


@pytest.mark.parametrize('sctype', UNSIGNED_INTEGERS)
def test_any_unsigned_integer_array(
    sctype: type[onp.AnyUnsignedInteger],
) -> None:
    v: onp.AnyUnsignedInteger = sctype(42)
    x: onp.AnyUnsignedIntegerArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.unsignedinteger)


@pytest.mark.parametrize('sctype', SIGNED_INTEGERS)
def test_any_signed_integer_array(
    sctype: type[onp.AnySignedInteger],
) -> None:
    v: onp.AnySignedInteger = sctype(42)
    x: onp.AnySignedIntegerArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.signedinteger)


@pytest.mark.parametrize('sctype', INTEGERS)
def test_any_integer_array(
    sctype: type[onp.AnyInteger],
) -> None:
    v: onp.AnyInteger = sctype(42)
    x: onp.AnyIntegerArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.integer)


@pytest.mark.parametrize('sctype', FLOATING)
def test_any_floating_array(
    sctype: type[onp.AnyFloating],
) -> None:
    v: onp.AnyFloating = sctype(42)
    x: onp.AnyFloatingArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.floating)


@pytest.mark.parametrize('sctype', COMPLEX_FLOATING)
def test_any_complex_floating_array(
    sctype: type[onp.AnyComplexFloating],
) -> None:
    v: onp.AnyComplexFloating = sctype(42 + 42j)
    x: onp.AnyComplexFloatingArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.complexfloating)


@pytest.mark.parametrize('sctype', DATETIME64)
def test_any_datetime64_array(
    sctype: type[onp.AnyDateTime64],
) -> None:
    v: onp.AnyDateTime64 = sctype()
    x: onp.AnyDateTime64Array[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.datetime64)


@pytest.mark.parametrize('sctype', TIMEDELTA64)
def test_any_timedelta64_array(
    sctype: type[onp.AnyTimeDelta64],
) -> None:
    v: onp.AnyTimeDelta64 = sctype()
    x: onp.AnyTimeDelta64Array[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.timedelta64)


@pytest.mark.parametrize('sctype', STR)
def test_any_str_array(
    sctype: type[onp.AnyStr],
) -> None:
    v: onp.AnyStr = sctype()
    x: onp.AnyStrArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.str_)


@pytest.mark.parametrize('sctype', BYTES)
def test_any_bytes_array(
    sctype: type[onp.AnyBytes],
) -> None:
    v: onp.AnyBytes = sctype()
    x: onp.AnyBytesArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.bytes_)


@pytest.mark.parametrize('sctype', CHARACTER)
def test_any_character_array(
    sctype: type[onp.AnyCharacter],
) -> None:
    v: onp.AnyCharacter = sctype()
    x: onp.AnyCharacterArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.character)


@pytest.mark.parametrize('sctype', VOID)
def test_any_void_array(
    sctype: type[onp.AnyVoid],
) -> None:
    v: onp.AnyVoid = sctype(0)
    x: onp.AnyVoidArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.void)


@pytest.mark.parametrize('sctype', FLEXIBLE)
def test_any_flexible_array(
    sctype: type[onp.AnyFlexible],
) -> None:
    v: onp.AnyFlexible = sctype(0)
    x: onp.AnyFlexibleArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.flexible)


@pytest.mark.parametrize('sctype', BOOL)
def test_any_bool_array(
    sctype: type[onp.AnyBool],
) -> None:
    v: onp.AnyBool = sctype(True)
    x: onp.AnyBoolArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.bool_)


@pytest.mark.parametrize('sctype', OBJECT)
def test_any_object_array(
    sctype: type[onp.AnyObject],
) -> None:
    v: onp.AnyObject = sctype()
    x: onp.AnyObjectArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.object_)
