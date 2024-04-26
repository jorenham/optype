# pyright: reportRedeclaration=false
"""
Type aliases for `numpy.dtype.name`.

See Also:
    - https://numpy.org/devdocs/reference/arrays.scalars.html
    - https://numpy.org/devdocs/reference/generated/numpy.dtype.name.html
"""

from typing import Literal as _L, TypeAlias as _T  # noqa: N814

from numpy.version import short_version as _np_version


_NP_V1 = _np_version < '2'


# generic :> number :> integer :> signedinteger :> _
Int8: _T = _L['int8']
Int16: _T = _L['int16']
Int32: _T = _L['int32']
Int64: _T = _L['int64']
Byte: _T = _L['byte']
Short: _T = _L['short']
IntC: _T = _L['intc']
if _NP_V1:
    IntP: _T = _L['intp', 'int0']
    Long: _T = _L['long', 'int', 'int_']
    Int: _T = Long
else:
    IntP: _T = _L['intp', 'int', 'int_']
    Long: _T = _L['long']
    Int: _T = IntP
LongLong: _T = _L['longlong']

# generic :> number :> integer :> signedinteger
_I_N: _T = Int8 | Int16 | Int32 | Int64
SignedInteger: _T = _I_N | Byte | Short | IntC | IntP | Long | LongLong

# generic :> number :> integer :> unsignedinteger :> _
UInt8: _T = _L['uint8']
UInt16: _T = _L['uint16']
UInt32: _T = _L['uint32']
UInt64: _T = _L['uint64']
UByte: _T = _L['ubyte']
UShort: _T = _L['ushort']
UIntC: _T = _L['uintc']
if _NP_V1:
    UIntP: _T = _L['uintp', 'uint0']
    ULong: _T = _L['ulong', 'uint']
    UInt: _T = ULong
else:
    UIntP: _T = _L['uintp', 'uint']
    ULong: _T = _L['ulong']
    UInt: _T = UIntP
ULongLong: _T = _L['ulonglong']

# generic :> number :> integer :> unsignedinteger
_U_N: _T = UInt8 | UInt16 | UInt32 | UInt64
UnsignedInteger: _T = _U_N | UByte | UShort | UIntC | UIntP | ULong | ULongLong

# generic :> number :> integer
Integer: _T = SignedInteger | UnsignedInteger

# generic :> number :> inexact :> floating :> _
Float16: _T = _L['float16']
Float32: _T = _L['float32']
Float64: _T = _L['float64']
Half: _T = _L['half']
Single: _T = _L['single']
if _NP_V1:
    Double: _T = _L['double', 'float', 'float_']
    LongDouble: _T = _L['longdouble', 'longfloat']
else:
    Double: _T = _L['double', 'float']
    LongDouble: _T = _L['longdouble']
Float: _T = Double

# generic :> number :> inexact :> floating
_F_N: _T = Float16 | Float32 | Float64
Floating: _T = _F_N | Half | Single | Double | LongDouble

# generic :> number :> inexact :> complexfloating :> _
Complex64: _T = _L['complex64']
Complex128: _T = _L['complex128']
if _NP_V1:
    CSingle: _T = _L['csingle', 'singlecomplex']
    CDouble: _T = _L['cdouble', 'complex', 'complex_', 'cfloat']
    CLongDouble: _T = _L['clongdouble', 'clongfloat', 'longcomplex']
else:
    CSingle: _T = _L['csingle']
    CDouble: _T = _L['cdouble', 'complex']
    CLongDouble: _T = _L['clongdouble']
Complex: _T = CDouble

# generic :> number :> inexact :> complexfloating
ComplexFloating: _T = Complex64 | Complex128 | CSingle | CDouble | CLongDouble

# generic :> number :> inexact
Inexact: _T = Floating | ComplexFloating
# generic :> number
Number: _T = Integer | Inexact

# generic :> flexible :> character :> _
if _NP_V1:
    Str: _T = _L['str', 'str_', 'str0', 'unicode', 'unicode_']
    Bytes: _T = _L['bytes', 'bytes_', 'bytes0']
else:
    Str: _T = _L['str', 'str_', 'unicode']
    Bytes: _T = _L['bytes', 'bytes_']

# generic :> flexible :> character
Character: _T = Str | Bytes

# generic :> flexible :> _
if _NP_V1:
    Void: _T = _L['void', 'void0']
else:
    Void: _T = _L['void']

# generic :> flexible
Flexible: _T = Void | Character

# generic :> _
if _NP_V1:
    Bool: _T = _L['bool', 'bool_', 'bool8']
else:
    Bool: _T = _L['bool', 'bool_']
Datetime64: _T = _L['datetime64']
Timedelta64: _T = _L['timedelta64']
Object: _T = _L['object', 'object_']

# generic
Generic: _T = Bool | Datetime64 | Timedelta64 | Object | Number | Flexible
