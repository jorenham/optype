"""
A collection of `ctypes` type aliases for several numpy scalar types.

NOTE:
    `optype.numpy` assumes a `C99`-compatible C compiler, a 32 or 64-bit
    system, and a `ILP32`, `LLP64` or `LP64` data model; see
    https://en.cppreference.com/w/c/language/arithmetic_types for more info.

    If this is not the case for you, then please open an issue at:
    https://github.com/jorenham/optype/issues
"""

from __future__ import annotations

import ctypes as ct
import sys

# ruff: noqa: N812
from ctypes import (
    c_bool as Bool,
    c_byte as Byte,
    c_char as Bytes,
    c_double as Float64,
    c_float as Float32,
    c_int as IntC,
    c_int8 as Int8,
    c_int16 as Int16,
    c_int32 as Int32,
    c_int64 as Int64,
    c_long as Long,
    # NOTE: `longdouble` only works as type, not as value!
    c_longdouble as LongDouble,
    c_longlong as LongLong,
    c_short as Short,
    c_size_t as UIntP,  # `void_p` on numpy<2, but almost always the same
    c_ssize_t as IntP,
    c_ubyte as UByte,
    c_uint as UIntC,
    c_uint8 as UInt8,
    c_uint16 as UInt16,
    c_uint32 as UInt32,
    c_uint64 as UInt64,
    c_ulong as ULong,
    # NOTE: `ulongdouble` only works as type, not as value!
    c_ulonglong as ULongLong,
    c_ushort as UShort,
)
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast


if sys.version_info >= (3, 13):
    from typing import Never, TypeVar
else:
    from typing_extensions import Never, TypeVar


if sys.version_info >= (3, 14):
    from ctypes import (
        c_double_complex as Complex128,
        c_float_complex as Complex64,
        c_longdouble_complex as CLongDouble,
    )
else:
    Complex64 = Never
    Complex128 = Never
    CLongDouble = Never

from ._ctypeslib import CScalar, CType


# ruff: noqa: RUF022
__all__ = [
    "CType",
    "CScalar",
    "Array",

    "Generic",
    "Number",
    "Integer",
    "Inexact",
    "UnsignedInteger",
    "SignedInteger",
    "Floating",
    "ComplexFloating",
    "Flexible",

    "Bool",

    "UInt8", "Int8",
    "UInt16", "Int16",
    "UInt32", "Int32",
    "UInt64", "Int64",
    "UByte", "Byte",
    "UShort", "Short",
    "UIntC", "IntC",
    "UIntP", "IntP",
    "ULong", "Long",
    "ULongLong", "LongLong",

    "Float32",
    "Float64",
    "LongDouble",

    "Complex64",
    "Complex128",
    "CLongDouble",

    "Bytes",
    "Void",
    "Object",
]  # fmt: skip


SIZE_BYTE: Final = cast(Literal[1], ct.sizeof(ct.c_byte))
SIZE_SHORT: Final = cast(Literal[2], ct.sizeof(ct.c_short))
SIZE_INTC: Final = cast(Literal[4], ct.sizeof(ct.c_int))
SIZE_INTP: Final = cast(Literal[4, 8], ct.sizeof(ct.c_ssize_t))
SIZE_LONG: Final = cast(Literal[4, 8], ct.sizeof(ct.c_long))
SIZE_LONGLONG: Final = cast(Literal[8], ct.sizeof(ct.c_longlong))

assert SIZE_BYTE == 1, f"`sizeof(byte) = {SIZE_BYTE}`, expected 1"
assert SIZE_SHORT == 2, f"`sizeof(short) = {SIZE_SHORT}`, expected 2"
assert SIZE_INTC == 4, f"`sizeof(int) = {SIZE_INTC}`, expected 4"
assert SIZE_INTP in {4, 8}, f"`sizeof(ssize_t) = {SIZE_INTP}`, expected 4 or 8"
assert SIZE_LONG in {4, 8}, f"`sizeof(long int) = {SIZE_LONG}`, expected 4 or 8"

SIZE_SINGLE: Final = cast(Literal[4], ct.sizeof(ct.c_float))
SIZE_DOUBLE: Final = cast(Literal[8], ct.sizeof(ct.c_double))
SIZE_LONGDOUBLE: Final = cast(Literal[8, 12, 16], ct.sizeof(ct.c_longdouble))

assert SIZE_SINGLE == 4, f"`sizeof(float) = {SIZE_SINGLE}`, expected 4"
assert SIZE_DOUBLE == 8, f"`sizeof(double) = {SIZE_DOUBLE}`, expected 8"
assert SIZE_LONGDOUBLE in {8, 12, 16}, (
    f"`sizeof(long double) = {SIZE_LONGDOUBLE}`, expected 12 or 16"
)  # fmt: skip


CT = TypeVar("CT", bound=CType)
Array: TypeAlias = ct.Array[CT] | ct.Array["Array[CT]"]


# `c_(u)byte` is an alias for `c_(u)int8`
UnsignedInteger: TypeAlias = (
    UInt8 | UInt16 | UInt32 | UInt64
    | UShort | UIntC | UIntP | ULong | ULongLong
)  # fmt: skip
SignedInteger: TypeAlias = (
    Int8 | Int16 | Int32 | Int64
    | Short | IntC | IntP | Long | LongLong
)  # fmt: skip

Void: TypeAlias = ct.Structure | ct.Union

# subscripting at runtime will give an error, and no default is defined...
if TYPE_CHECKING:
    Object: TypeAlias = ct.py_object[object]
else:
    Object: TypeAlias = ct.py_object

Flexible: TypeAlias = Bytes | Void
# CScalar is invariant, so this won't match `CScalar[bool]`
Integer: TypeAlias = CScalar[int]
# NOTE: mypy incorrectly ignores the invariance of `CScalar` when using `float`
Floating: TypeAlias = CScalar[float]

if sys.version_info >= (3, 14):
    ComplexFloating: TypeAlias = CScalar[complex]
    Inexact: TypeAlias = Floating | ComplexFloating
    Number: TypeAlias = Integer | Inexact
    Generic: TypeAlias = Bool | Number | Flexible | Object
else:
    ComplexFloating: TypeAlias = Never
    Inexact: TypeAlias = Floating
    Number: TypeAlias = Integer | Floating
    Generic: TypeAlias = Bool | Integer | Floating | Flexible | Object
