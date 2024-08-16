from __future__ import annotations

import ctypes as ct
import datetime as dt
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from optype.numpy import _compat as _x


if TYPE_CHECKING:
    import optype.numpy as onp
    from optype.numpy import ctypeslib as _ct


# All allowed arguments that when passed to `np.array`, will result in an
# array of the specified scalar type(s).

_UNSIGNED_INTEGER_NP: Final = (
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.ubyte, np.ushort, np.uintc, _x.ULong, np.ulonglong,
)  # fmt: skip
_UNSIGNED_INTEGER_CT: Final = (
    ct.c_uint8, ct.c_uint16, ct.c_uint32, ct.c_uint64, ct.c_size_t,
    ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
)  # fmt: skip
UNSIGNED_INTEGER: Final = (*_UNSIGNED_INTEGER_NP, *_UNSIGNED_INTEGER_CT)
_SIGNED_INTEGER_NP: Final = (
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.byte, np.short, np.intc, _x.Long, np.longlong,
)  # fmt: skip
_SIGNED_INTEGER_CT: Final = (
    ct.c_int8, ct.c_int16, ct.c_int32, ct.c_int64, ct.c_ssize_t,
    ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
)  # fmt: skip
SIGNED_INTEGER: Final = (*_SIGNED_INTEGER_NP, *_SIGNED_INTEGER_CT)
INTEGER: Final = (*UNSIGNED_INTEGER, *SIGNED_INTEGER)

_FLOATING_NP: Final = (
    np.float16, np.float32, np.float64,
    np.half, np.single, np.double, np.longdouble,
)  # fmt: skip
_FLOATING_CT: Final = ct.c_float, ct.c_double
FLOATING: Final = *_FLOATING_NP, *_FLOATING_CT
COMPLEX_FLOATING: Final = (
    np.complex64, np.complex128, np.csingle, np.cdouble, np.clongdouble,
)  # fmt: skip
DATETIME64: Final = np.datetime64, dt.datetime
TIMEDELTA64: Final = np.timedelta64, dt.timedelta
STR: Final = np.str_, str
BYTES: Final = np.bytes_, ct.c_char, bytes
CHARACTER: Final = *STR, *BYTES
VOID: Final = (np.void,)  # TODO: structured
FLEXIBLE: Final = (*VOID, *CHARACTER)
BOOL: Final = _x.Bool, ct.c_bool, bool
OBJECT: Final = np.object_, ct.py_object


def test_any_array() -> None:
    type_np = np.int16
    type_ct = ct.c_int16
    v = 42

    # Python scalar

    v_py: onp.AnyArray = v
    assert np.shape(v) == ()

    # C scalar

    v_ct: onp.AnyArray = type_ct(42)

    # NumPy scalar

    v_np = type_np(v)
    v_np_any: onp.AnyArray = v_np
    assert int(v) == v_py
    assert np.shape(v_np) == ()

    # 0d

    x0_np = np.array(v_py)
    x0_np_any: onp.AnyArray = x0_np
    assert np.shape(x0_np) == ()

    # 1d

    x1_py: list[int] = [v]
    x1_py_any: onp.AnyArray = x1_py
    assert np.shape(x1_py) == (1,)

    x1_py_np = [x0_np]
    x1_py_np_any: onp.AnyArray = x1_py_np
    assert np.shape(x1_py_np) == (1,)

    x1_np = np.array(x1_py)
    x1_np_any: onp.AnyArray = x1_np
    assert np.shape(x1_np) == (1,)

    # 2d

    x2_py = [x1_py]
    x2_py_any: onp.AnyArray = x2_py
    assert np.shape(x2_py) == (1, 1)

    x2_py_np = [x1_np]
    x2_py_np_any: onp.AnyArray = x2_py_np
    assert np.shape(x2_py_np) == (1, 1)

    x2_np = np.array(x2_py)
    x2_np_any: onp.AnyArray = x2_np
    assert np.shape(x2_np) == (1, 1)


@pytest.mark.parametrize('sctype', UNSIGNED_INTEGER)
def test_any_unsigned_integer_array(
    sctype: type[np.unsignedinteger[Any] | _ct.UnsignedInteger],
) -> None:
    v = sctype(42)
    x = np.array(v)
    x_any: onp.AnyUnsignedIntegerArray = x
    assert np.issubdtype(x.dtype, np.unsignedinteger)


@pytest.mark.parametrize('sctype', SIGNED_INTEGER)
def test_any_signed_integer_array(
    sctype: type[np.signedinteger[Any] | _ct.SignedInteger],
) -> None:
    v = sctype(42)
    x = np.array(v)
    x_any: onp.AnySignedIntegerArray = x
    assert np.issubdtype(x.dtype, np.signedinteger)


@pytest.mark.parametrize('sctype', INTEGER)
def test_any_integer_array(
    sctype: type[np.integer[Any] | _ct.Integer],
) -> None:
    v = sctype(42)
    x = np.array(v)
    x_any: onp.AnyIntegerArray = x
    assert np.issubdtype(x.dtype, np.integer)


@pytest.mark.parametrize('sctype', FLOATING)
def test_any_floating_array(
    sctype: type[np.floating[Any] | _ct.Floating],
) -> None:
    v = sctype(42)
    x = np.array(v)
    x_any: onp.AnyFloatingArray = x
    assert np.issubdtype(x.dtype, np.floating)


@pytest.mark.parametrize('sctype', COMPLEX_FLOATING)
def test_any_complex_floating_array(
    sctype: type[np.complexfloating[Any, Any]],
) -> None:
    v = sctype(42 + 42j)
    x = np.array(v)
    x_any: onp.AnyComplexFloatingArray = x
    assert np.issubdtype(x.dtype, np.complexfloating)


@pytest.mark.parametrize('sctype', DATETIME64)
def test_any_datetime64_array(
    sctype: type[dt.datetime | np.datetime64],
) -> None:
    v: dt.datetime | np.datetime64
    v = sctype.now() if issubclass(sctype, dt.datetime) else sctype()
    x = np.datetime64(v)
    x_any: onp.AnyDateTime64Array = x
    assert np.issubdtype(x.dtype, np.datetime64)


@pytest.mark.parametrize('sctype', TIMEDELTA64)
def test_any_timedelta64_array(
    sctype: type[np.timedelta64 | dt.timedelta],
) -> None:
    v = sctype()
    x = np.timedelta64(v)
    x_any: onp.AnyTimeDelta64Array = x
    assert np.issubdtype(x.dtype, np.timedelta64)


@pytest.mark.parametrize('sctype', STR)
def test_any_str_array(sctype: type[str | np.str_]) -> None:
    v = sctype()
    x = np.array(v)
    x_any: onp.AnyStrArray = x
    assert np.issubdtype(x.dtype, np.str_)


@pytest.mark.parametrize('sctype', BYTES)
def test_any_bytes_array(sctype: type[bytes | np.bytes_ | _ct.Bytes]) -> None:
    v = sctype()
    x = np.array(v)
    x_any: onp.AnyBytesArray = x
    assert np.issubdtype(x.dtype, np.bytes_)


@pytest.mark.parametrize('sctype', CHARACTER)
def test_any_character_array(
    sctype: type[str | bytes | np.character | _ct.Bytes],
) -> None:
    v = sctype()
    x = np.array(v)
    x_any: onp.AnyCharacterArray = x
    assert np.issubdtype(x.dtype, np.character)


@pytest.mark.parametrize('sctype', VOID)
def test_any_void_array(sctype: type[memoryview | np.void]) -> None:
    v = sctype(b'')
    x = np.array(v)
    x_any: onp.AnyVoidArray = x
    assert np.issubdtype(x.dtype, np.void)


@pytest.mark.parametrize('sctype', FLEXIBLE)
def test_any_flexible_array(
    sctype: type[bytes | str | np.flexible | _ct.Flexible],
) -> None:
    v = sctype(0)
    x = np.array(v)
    x_any: onp.AnyFlexibleArray = x
    assert np.issubdtype(x.dtype, np.flexible)


@pytest.mark.parametrize('sctype', BOOL)
def test_any_bool_array(sctype: type[bool | np.bool | _ct.Bool]) -> None:
    v = sctype(True)
    x = np.array(v)
    x_any: onp.AnyBoolArray = x
    assert np.issubdtype(x.dtype, np.bool_)


@pytest.mark.parametrize('sctype', OBJECT)
def test_any_object_array(sctype: type[np.object_ | _ct.Object]) -> None:
    v = sctype()
    x = np.array(v)
    x_any: onp.AnyObjectArray = x
    assert np.issubdtype(x.dtype, np.object_)
