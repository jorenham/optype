"""
A collection of `ctypes` type aliases for several numpy scalar types.
"""
from __future__ import annotations

import ctypes as ct
import sys
from typing import TYPE_CHECKING, Any, TypeAlias


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
        Never as LongDouble,
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
        Never as LongDouble,
        Never as Str,
        Never as String,
        Never as TimeDelta64,
        Never as Void,
    )

__all__ = (
    'Bool',
    'Byte',
    'Bytes',
    'CDouble',
    'CLongDouble',
    'CSingle',
    'Character',
    'Complex64',
    'Complex128',
    'ComplexFloating',
    'DateTime64',
    'Double',
    'Flexible',
    'Float16',
    'Float32',
    'Float64',
    'Floating',
    'Generic',
    'Half',
    'Inexact',
    'Int8',
    'Int16',
    'Int32',
    'Int64',
    'IntC',
    'IntP',
    'Integer',
    'Long',
    'LongDouble',
    'LongDouble0',
    'LongLong',
    'Number',
    'Short',
    'SignedInteger',
    'Single',
    'Str',
    'String',
    'TimeDelta64',
    'UByte',
    'UInt0',
    'UInt8',
    'UInt16',
    'UInt32',
    'UInt64',
    'UIntC',
    'UIntP',
    'ULong',
    'ULongLong',
    'UShort',
    'UnsignedInteger',
    'Void',
)


UByte: TypeAlias = ct.c_ubyte
UShort: TypeAlias = ct.c_ushort
UIntC: TypeAlias = ct.c_uint
ULong: TypeAlias = ct.c_ulong
ULongLong: TypeAlias = ct.c_ulonglong
UInt8: TypeAlias = ct.c_uint8
UInt16: TypeAlias = ct.c_uint16
UInt32: TypeAlias = ct.c_uint32
UInt64: TypeAlias = ct.c_uint64
UIntP: TypeAlias = ct.c_size_t
# gives a dtype of uintp, but fails for e.g. `np.array`
UInt0: TypeAlias = ct.c_void_p
UnsignedInteger: TypeAlias = (
    UByte | UShort | UIntC | ULong | ULongLong
    | UInt8 | UInt16 | UInt32 | UInt64 | UIntP
)

Byte: TypeAlias = ct.c_byte
Short: TypeAlias = ct.c_short
IntC: TypeAlias = ct.c_int
Long: TypeAlias = ct.c_long
LongLong: TypeAlias = ct.c_longlong
Int8: TypeAlias = ct.c_int8
Int16: TypeAlias = ct.c_int16
Int32: TypeAlias = ct.c_int32
Int64: TypeAlias = ct.c_int64
IntP: TypeAlias = ct.c_ssize_t
SignedInteger: TypeAlias = (
    Byte | Short | IntC | Long | LongLong
    | Int8 | Int16 | Int32 | Int64 | IntP
)

Single: TypeAlias = ct.c_float
Double: TypeAlias = ct.c_double
# gives a dtype of e.g. float128, but fails for e.g. `np.array`
LongDouble0: TypeAlias = ct.c_longdouble
Float32: TypeAlias = Single
Float64: TypeAlias = Double
Floating: TypeAlias = Single | Double  # | LongDouble

# dtype will have .char='S1' and .name='bytes8'; not 'S' and `bytes`
Bytes: TypeAlias = ct.c_char
Character: TypeAlias = Bytes
Flexible: TypeAlias = Character

Bool: TypeAlias = ct.c_bool

# subscripting at runtime will give an error
if TYPE_CHECKING:
    Object: TypeAlias = ct.py_object[Any]
else:
    Object: TypeAlias = ct.py_object

Integer: TypeAlias = UnsignedInteger | SignedInteger
Inexact: TypeAlias = Floating | ComplexFloating  # noqa: RUF020
Number: TypeAlias = Integer | Inexact
Generic: TypeAlias = Number | Flexible | Bool | Object
