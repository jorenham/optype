"""
A collection of `ctypes` type aliases for several numpy scalar types.
"""
from __future__ import annotations

import ctypes as ct
import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, cast

import optype.numpy._compat as _x


if sys.version_info >= (3, 13):
    from typing import (
        Never as CDouble,
        Never as CLongDouble,
        Never as CSingle,
        Never as Complex64,
        Never as Complex128,
        Never as ComplexFloating,
        Never as DateTime64,
        Never as Float16,
        Never as Half,
        Never as Str,
        Never as String,
        Never as TimeDelta64,
        Never as Void,
    )
else:
    from typing_extensions import (
        Never as CDouble,
        Never as CLongDouble,
        Never as CSingle,
        Never as Complex64,
        Never as Complex128,
        Never as ComplexFloating,
        Never as DateTime64,
        Never as Float16,
        Never as Half,
        Never as Str,
        Never as String,
        Never as TimeDelta64,
        Never as Void,
    )


# ruff: noqa: RUF022
__all__: list[str] = []
__all__ += ['Generic', 'Number', 'Integer', 'Inexact']
__all__ += [
    'UnsignedInteger',
    'UInt8', 'UInt16', 'UInt32', 'UInt64', 'UIntP',
    'UByte', 'UShort', 'UIntC', 'ULong', 'ULongLong',
]  # fmt: skip
__all__ += [
    'SignedInteger',
    'Int8', 'Int16', 'Int32', 'Int64', 'IntP',
    'Byte', 'Short', 'IntC', 'Long', 'LongLong',
]  # fmt: skip
__all__ += [
    'Floating',
    'Float16', 'Float32', 'Float64',
    'Half', 'Single', 'Double', 'LongDouble', 'LongDouble',
]  # fmt: skip
__all__ += [
    'ComplexFloating',
    'Complex64', 'Complex128',
    'CSingle', 'CDouble', 'CLongDouble',
]  # fmt: skip
__all__ += ['Character', 'Bytes', 'Str']
__all__ += ['Flexible', 'Void']
__all__ += ['Bool', 'DateTime64', 'TimeDelta64', 'Object']
__all__ += ['String']


# NOTE: `optype.numpy` assumes a `C99`-compatible C compiler, a 32 or 64-bit
# system, and a `ILP32`, `LLP64` or `LP64` data model.
# See https://en.cppreference.com/w/c/language/arithmetic_types for details.
#
# If this is not the case for you, then please open an issue as:
# https://github.com/jorenham/optype/issues


SIZE_BYTE: Final = cast(Literal[1], ct.sizeof(ct.c_byte))
SIZE_SHORT: Final = cast(Literal[2], ct.sizeof(ct.c_short))
SIZE_INTC: Final = cast(Literal[4], ct.sizeof(ct.c_int))
SIZE_INTP: Final = cast(Literal[4, 8], ct.sizeof(ct.c_ssize_t))
SIZE_LONG: Final = cast(Literal[4, 8], ct.sizeof(ct.c_long))
SIZE_LONGLONG: Final = cast(Literal[8], ct.sizeof(ct.c_longlong))

assert SIZE_BYTE == 1, f'`sizeof(byte) = {SIZE_BYTE}`, expected 1'
assert SIZE_SHORT == 2, f'`sizeof(short) = {SIZE_SHORT}`, expected 2'
assert SIZE_INTC == 4, f'`sizeof(int) = {SIZE_INTC}`, expected 4'
assert SIZE_INTP in {4, 8}, f'`sizeof(ssize_t) = {SIZE_INTP}`, expected 4 or 8'
assert ct.sizeof(ct.c_void_p) == SIZE_INTP, (
    'expected `sizeof(void*) == sizeof(ssize_t)`'
)
assert SIZE_LONG in {4, 8}, (
    f'`sizeof(long int) = {SIZE_LONG}`, expected 4 or 8'
)

SIZE_SINGLE: Final = cast(Literal[4], ct.sizeof(ct.c_float))
SIZE_DOUBLE: Final = cast(Literal[8], ct.sizeof(ct.c_double))
SIZE_LONGDOUBLE: Final = cast(Literal[8, 12, 16], ct.sizeof(ct.c_longdouble))

assert SIZE_SINGLE == 4, f'`sizeof(float) = {SIZE_SINGLE}`, expected 4'
assert SIZE_DOUBLE == 8, f'`sizeof(double) = {SIZE_DOUBLE}`, expected 8'
assert SIZE_LONGDOUBLE in {8, 12, 16}, (
    f'`sizeof(long double) = {SIZE_LONGDOUBLE}`, expected 12 or 16'
)


# always 8-bits
UByte: TypeAlias = ct.c_ubyte
Byte: TypeAlias = ct.c_byte
# always be 16-bits
UShort: TypeAlias = ct.c_ushort
Short: TypeAlias = ct.c_short
# always be 32-bits; sometimes a 'long' alias
UIntC: TypeAlias = ct.c_uint
IntC: TypeAlias = ct.c_int
# either 32- or 64-bits (the latter requires a 64-bits Unix-based system)
ULong: TypeAlias = ct.c_ulong
Long: TypeAlias = ct.c_long
# always be 64-bits
ULongLong: TypeAlias = ct.c_ulonglong
LongLong: TypeAlias = ct.c_longlong
# always a byte
UInt8: TypeAlias = ct.c_uint8 | ct.c_ubyte
Int8: TypeAlias = ct.c_int8 | ct.c_byte
# always a short
UInt16: TypeAlias = ct.c_uint16 | ct.c_ushort
Int16: TypeAlias = ct.c_int16 | ct.c_short
# always a C int
UInt32: TypeAlias = ct.c_uint32 | ct.c_uint
Int32: TypeAlias = ct.c_int32 | ct.c_int
# could be a 'long' (if long is 64-bits), otherwise it's a 'long long'
UInt64: TypeAlias = ct.c_uint64
Int64: TypeAlias = ct.c_int64

IntP: TypeAlias = ct.c_ssize_t
SignedInteger: TypeAlias = (
    Int8 | Int16 | Int32 | Int64 | Long | LongLong | IntP
)

# always 32-bits
Single: TypeAlias = ct.c_float
Float32: TypeAlias = Single
# always 64-bits
Double: TypeAlias = ct.c_double
Float64: TypeAlias = Double
# either 64-, 96-, or 128-bits
LongDouble: TypeAlias = ct.c_longdouble
Floating: TypeAlias = Single | Double | LongDouble

# NOTE: The resulting dtype will have typecode 'S1' and name 'bytes8'
Bytes: TypeAlias = ct.c_char
Character: TypeAlias = Bytes
Flexible: TypeAlias = Character

Bool: TypeAlias = ct.c_bool


# subscripting at runtime will give an error
if TYPE_CHECKING:
    Object: TypeAlias = ct.py_object[Any]
else:
    Object: TypeAlias = ct.py_object

Inexact: TypeAlias = Floating

# this is a workaround of `pyright --verifytypes`, which requires all
# type aliases to be within the same branch
if _x.NP2:
    UIntP: TypeAlias = ct.c_size_t
    UnsignedInteger: TypeAlias = (
        UInt8 | UInt16 | UInt32 | UInt64 | ULong | ULongLong | UIntP
    )

    Integer: TypeAlias = UnsignedInteger | SignedInteger
    Number: TypeAlias = Integer | Inexact
    Generic: TypeAlias = Number | Flexible | Bool | Object
else:
    # `c_void_p` can't be used as value for e.g. `np.array`
    UIntP: TypeAlias = ct.c_void_p
    UnsignedInteger: TypeAlias = (
        UInt8 | UInt16 | UInt32 | UInt64 | ULong | ULongLong | UIntP
    )

    Integer: TypeAlias = UnsignedInteger | SignedInteger
    Number: TypeAlias = Integer | Inexact
    Generic: TypeAlias = Number | Flexible | Bool | Object
