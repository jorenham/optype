from __future__ import annotations

import ctypes as ct
import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import numpy as np
import pytest

from optype.numpy import _compat as _x


if TYPE_CHECKING:
    import optype.numpy as onp
    from optype.numpy import _any_scalar as _sc, _ctype as _ct


_0D: TypeAlias = tuple[()]  # noqa: PYI042
_1D: TypeAlias = tuple[Literal[1]]  # noqa: PYI042
_2D: TypeAlias = tuple[Literal[1], Literal[1]]  # noqa: PYI042

_SC = TypeVar('_SC')
_Types: TypeAlias = tuple[type[_SC], ...]


# All allowed arguments that when passed to `np.array`, will result in an
# array of the specified scalar type(s).

_UNSIGNED_INTEGER_NP = (
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.ubyte, np.ushort, np.uintc, _x.ULong, np.ulonglong,
)  # fmt: skip
_UNSIGNED_INTEGER_CT: _Types[_ct.UnsignedInteger] = (
    ct.c_uint8, ct.c_uint16, ct.c_uint32, ct.c_uint64, ct.c_size_t,
    ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
)  # fmt: skip
UNSIGNED_INTEGER: _Types[_sc.AnyUnsignedInteger] = (
    *_UNSIGNED_INTEGER_NP,
    *_UNSIGNED_INTEGER_CT,
)
_SIGNED_INTEGER_NP = (
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.byte, np.short, np.intc, _x.Long, np.longlong,
)  # fmt: skip
_SIGNED_INTEGER_CT: _Types[_ct.SignedInteger] = (
    ct.c_int8, ct.c_int16, ct.c_int32, ct.c_int64, ct.c_ssize_t,
    ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
)  # fmt: skip
SIGNED_INTEGER: _Types[_sc.AnySignedInteger] = (
    *_SIGNED_INTEGER_NP,
    *_SIGNED_INTEGER_CT,
)
INTEGER: _Types[_sc.AnyInteger] = *UNSIGNED_INTEGER, *SIGNED_INTEGER

_FLOATING_NP = (
    np.float16, np.float32, np.float64,
    np.half, np.single, np.double, np.longdouble,
)  # fmt: skip
_FLOATING_CT: _Types[_ct.Floating] = ct.c_float, ct.c_double
FLOATING: _Types[_sc.AnyFloating] = *_FLOATING_NP, *_FLOATING_CT
COMPLEX_FLOATING: _Types[_sc.AnyComplexFloating] = (
    np.complex64, np.complex128,
    np.csingle, np.cdouble, np.clongdouble,
)  # fmt: skip
DATETIME64: _Types[_sc.AnyDateTime64] = np.datetime64, dt.datetime
TIMEDELTA64: _Types[_sc.AnyTimeDelta64] = np.timedelta64, dt.timedelta
STR: _Types[_sc.AnyStr] = np.str_, str
BYTES: _Types[_sc.AnyBytes] = np.bytes_, ct.c_char, bytes
CHARACTER: _Types[_sc.AnyCharacter] = *STR, *BYTES
VOID: _Types[_sc.AnyVoid] = (np.void,)
FLEXIBLE: _Types[_sc.AnyFlexible] = *VOID, *CHARACTER
BOOL: _Types[_sc.AnyBool] = _x.Bool, ct.c_bool, bool
OBJECT: _Types[_sc.AnyObject] = np.object_, ct.py_object, object
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


@pytest.mark.parametrize('sctype', UNSIGNED_INTEGER)
def test_any_unsigned_integer_array(
    sctype: type[_sc.AnyUnsignedInteger],
) -> None:
    v: _sc.AnyUnsignedInteger = sctype(42)
    x: onp.AnyUnsignedIntegerArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.unsignedinteger)


@pytest.mark.parametrize('sctype', SIGNED_INTEGER)
def test_any_signed_integer_array(sctype: type[_sc.AnySignedInteger]) -> None:
    v: _sc.AnySignedInteger = sctype(42)
    x: onp.AnySignedIntegerArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.signedinteger)


@pytest.mark.parametrize('sctype', INTEGER)
def test_any_integer_array(sctype: type[_sc.AnyInteger]) -> None:
    v: _sc.AnyInteger = sctype(42)
    x: onp.AnyIntegerArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.integer)


@pytest.mark.parametrize('sctype', FLOATING)
def test_any_floating_array(sctype: type[_sc.AnyFloating]) -> None:
    v: _sc.AnyFloating = sctype(42)
    x: onp.AnyFloatingArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.floating)


@pytest.mark.parametrize('sctype', COMPLEX_FLOATING)
def test_any_complex_floating_array(
    sctype: type[_sc.AnyComplexFloating],
) -> None:
    v: _sc.AnyComplexFloating = sctype(42 + 42j)
    x: onp.AnyComplexFloatingArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.complexfloating)


@pytest.mark.parametrize('sctype', DATETIME64)
def test_any_datetime64_array(sctype: type[_sc.AnyDateTime64]) -> None:
    v: _sc.AnyDateTime64
    v = sctype.now() if issubclass(sctype, dt.datetime) else sctype()
    x: onp.AnyDateTime64Array[_0D] = np.datetime64(v)
    assert np.issubdtype(x.dtype, np.datetime64)


@pytest.mark.parametrize('sctype', TIMEDELTA64)
def test_any_timedelta64_array(sctype: type[_sc.AnyTimeDelta64]) -> None:
    v: _sc.AnyTimeDelta64 = sctype()
    x: onp.AnyTimeDelta64Array[_0D] = np.timedelta64(v)
    assert np.issubdtype(x.dtype, np.timedelta64)


@pytest.mark.parametrize('sctype', STR)
def test_any_str_array(sctype: type[_sc.AnyStr]) -> None:
    v: _sc.AnyStr = sctype()
    x: onp.AnyStrArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.str_)


@pytest.mark.parametrize('sctype', BYTES)
def test_any_bytes_array(sctype: type[_sc.AnyBytes]) -> None:
    v: _sc.AnyBytes = sctype()
    x: onp.AnyBytesArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.bytes_)


@pytest.mark.parametrize('sctype', CHARACTER)
def test_any_character_array(sctype: type[_sc.AnyCharacter]) -> None:
    v: _sc.AnyCharacter = sctype()
    x: onp.AnyCharacterArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.character)


@pytest.mark.parametrize('sctype', VOID)
def test_any_void_array(sctype: type[_sc.AnyVoid]) -> None:
    v: _sc.AnyVoid = sctype(0)
    x: onp.AnyVoidArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.void)


@pytest.mark.parametrize('sctype', FLEXIBLE)
def test_any_flexible_array(sctype: type[_sc.AnyFlexible]) -> None:
    v: _sc.AnyFlexible = sctype(0)
    x: onp.AnyFlexibleArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.flexible)


@pytest.mark.parametrize('sctype', BOOL)
def test_any_bool_array(sctype: type[_sc.AnyBool]) -> None:
    v: _sc.AnyBool = sctype(True)
    x: onp.AnyBoolArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.bool_)


@pytest.mark.parametrize('sctype', OBJECT)
def test_any_object_array(sctype: type[_sc.AnyObject]) -> None:
    v: _sc.AnyObject = sctype()
    x: onp.AnyObjectArray[_0D] = np.array(v)
    assert np.issubdtype(x.dtype, np.object_)
