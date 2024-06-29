"""
A collection of `ctypes` type aliases for several numpy scalar types.
"""
from __future__ import annotations

import ctypes as ct
import sys
from typing import TYPE_CHECKING, Any, TypeAlias as _Type


if sys.version_info >= (3, 13):
    from typing import Never
else:
    from typing_extensions import Never

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


UByte: _Type = ct.c_ubyte
UShort: _Type = ct.c_ushort
UIntC: _Type = ct.c_uint
ULong: _Type = ct.c_ulong
ULongLong: _Type = ct.c_ulonglong
UInt8: _Type = ct.c_uint8
UInt16: _Type = ct.c_uint16
UInt32: _Type = ct.c_uint32
UInt64: _Type = ct.c_uint64
UIntP: _Type = ct.c_size_t
# gives a dtype of uintp, but fails for e.g. `np.array`
UInt0: _Type = ct.c_void_p
UnsignedInteger: _Type = (
    UByte | UShort | UIntC | ULong | ULongLong
    | UInt8 | UInt16 | UInt32 | UInt64 | UIntP
)

Byte: _Type = ct.c_byte
Short: _Type = ct.c_short
IntC: _Type = ct.c_int
Long: _Type = ct.c_long
LongLong: _Type = ct.c_longlong
Int8: _Type = ct.c_int8
Int16: _Type = ct.c_int16
Int32: _Type = ct.c_int32
Int64: _Type = ct.c_int64
IntP: _Type = ct.c_ssize_t
SignedInteger: _Type = (
    Byte | Short | IntC | Long | LongLong
    | Int8 | Int16 | Int32 | Int64 | IntP
)

Half: _Type = Never
Single: _Type = ct.c_float
Double: _Type = ct.c_double
# gives a dtype of e.g. float128, but fails for e.g. `np.array`
LongDouble0: _Type = ct.c_longdouble
LongDouble: _Type = Never
Float16: _Type = Half
Float32: _Type = Single
Float64: _Type = Double
Floating: _Type = Single | Double  # | LongDouble

CSingle: _Type = Never
CDouble: _Type = Never
CLongDouble: _Type = Never
Complex64: _Type = Never
Complex128: _Type = Never
ComplexFloating: _Type = Never

DateTime64: _Type = Never
TimeDelta64: _Type = Never

Str: _Type = Never
# dtype will have .char='S1' and .name='bytes8'; not 'S' and `bytes`
Bytes: _Type = ct.c_char
Character: _Type = Bytes
Void: _Type = Never
Flexible: _Type = Character

Bool: _Type = ct.c_bool

# subscripting at runtime will give an error
if TYPE_CHECKING:
    Object: _Type = ct.py_object[Any]
else:
    Object: _Type = ct.py_object

# for the new (mostly broken) numpy.dtypes.StringDType() in numpy>=2
String: _Type = Never

Integer: _Type = UnsignedInteger | SignedInteger
Inexact: _Type = Floating | ComplexFloating
Number: _Type = Integer | Inexact
Generic: _Type = Number | Flexible | Bool | Object
