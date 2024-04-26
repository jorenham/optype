# pyright: reportRedeclaration=false
"""
Type aliases for `numpy.dtype.char`.

See Also:
    - https://numpy.org/devdocs/reference/arrays.scalars.html
    - https://numpy.org/devdocs/reference/generated/numpy.dtype.char.html

Todo:
    - 'T' for `np.dtypes.StringDType` on numpy>=2
"""

from typing import Literal as _L, TypeAlias as _T  # noqa: N814

from numpy.version import short_version as _np_version


_NP_V1 = _np_version < '2'

# generic :> number :> integer :> signedinteger :> _
Int8: _T = _L['i1']
Int16: _T = _L['i2']
Int32: _T = _L['i4']
Int64: _T = _L['i8']
Byte: _T = _L['b']
Short: _T = _L['h']
IntC: _T = _L['i']
Long: _T = _L['l']
LongLong: _T = _L['q']
if _NP_V1:
    IntP: _T = _L['p']
    Int: _T = Long
else:
    IntP: _T = _L['n', 'p']
    Int: _T = IntP

# generic :> number :> integer :> signedinteger
_I_N: _T = Int8 | Int16 | Int32 | Int64
SignedInteger: _T = _I_N | Byte | Short | IntC | IntP | Long | LongLong


# generic :> number :> integer :> unsignedinteger :> _
UInt8: _T = _L['u1']
UInt16: _T = _L['u2']
UInt32: _T = _L['u4']
UInt64: _T = _L['u8']
UByte: _T = _L['B']
UShort: _T = _L['H']
UIntC: _T = _L['I']
ULong: _T = _L['L']
ULongLong: _T = _L['Q']
if _NP_V1:
    UIntP: _T = _L['P']
    UInt: _T = ULong
else:
    UIntP: _T = _L['N', 'P']
    UInt: _T = UIntP

# generic :> number :> integer :> unsignedinteger
_U_N: _T = UInt8 | UInt16 | UInt32 | UInt64
UnsignedInteger: _T = _U_N | UByte | UShort | UIntC | UIntP | ULong | ULongLong

# generic :> number :> integer
Integer: _T = SignedInteger | UnsignedInteger

# generic :> number :> inexact :> floating :> _
Float16: _T = _L['f2']
Float32: _T = _L['f4']
Float64: _T = _L['f8']
Half: _T = _L['e']
Single: _T = _L['f']
Double: _T = _L['d']
LongDouble: _T = _L['g']
Float: _T = Double

# generic :> number :> inexact :> floating
_F_N: _T = Float16 | Float32 | Float64
Floating: _T = _F_N | Half | Single | Double | LongDouble

# generic :> number :> inexact :> complexfloating :> _
Complex64: _T = _L['c8']
Complex128: _T = _L['c16']
CSingle: _T = _L['F']
CDouble: _T = _L['D']
CLongDouble: _T = _L['G']
Complex: _T = CDouble

# generic :> number :> inexact :> complexfloating
ComplexFloating: _T = Complex64 | Complex128 | CSingle | CDouble | CLongDouble

# generic :> number :> inexact
Inexact: _T = Floating | ComplexFloating
# generic :> number
Number: _T = Integer | Inexact

# generic :> flexible :> character :> _
Str: _T = _L['U']
Bytes: _T = _L['S']

# generic :> flexible :> _
Character: _T = Str | Bytes
Void: _T = _L['V']

# generic :> _
Flexible: _T = Void | Character
Bool: _T = _L['?']
Datetime64: _T = _L['M', 'M8']
Timedelta64: _T = _L['m', 'm8']
Object: _T = _L['O']

# generic
Generic: _T = Bool | Datetime64 | Timedelta64 | Object | Number | Flexible
