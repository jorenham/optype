
"""
The allowed `np.dtype` arguments for specific scalar types.
The names are analogous to those in `numpy.dtypes`.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from typing import Any, Final, Literal as _Lit, TypeAlias as _Type

import numpy as np
import numpy.typing as npt

from . import _any_scalar as _s, _compat as _x, _ctype as _ct
from ._dtype import HasDType


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar


_NP_V2: Final[bool] = np.__version__.startswith('2.')


# helper aliases
_ST = TypeVar('_ST', bound=np.generic)
_VT = TypeVar('_VT', bound=object)
_Any1: _Type = np.dtype[_ST] | type[_ST] | HasDType[np.dtype[_ST]]
_Any2: _Type = np.dtype[_ST] | type[_ST | _VT] | HasDType[np.dtype[_ST]]


N = TypeVar('N', bound=npt.NBitBase, default=Any)
N2 = TypeVar('N2', bound=npt.NBitBase, default=N)


#
# integer - unsigned
#

# uint8
_UInt8Name: _Type = _Lit['uint8']
_UInt8Char: _Type = _Lit['u1', '=u1', '<u1', '>u1']
_UInt8Code: _Type = _UInt8Name | _UInt8Char
AnyUInt8DType: _Type = _Any2[np.uint8, _s.AnyUInt8] | _UInt8Code

# uint16
_UInt16Name: _Type = _Lit['uint16']
_UInt16Char: _Type = _Lit['u2', '=u2', '<u2', '>u2']
_UInt16Code: _Type = _UInt16Name | _UInt16Char
AnyUInt16DType: _Type = _Any2[np.uint16, _s.AnyUInt16] | _UInt16Code

# uint32
_UInt32Name: _Type = _Lit['uint32']
_UInt32Char: _Type = _Lit['u4', '=u4', '<u4', '>u4']
_UInt32Code: _Type = _UInt32Name | _UInt32Char
AnyUInt32DType: _Type = _Any2[np.uint32, _s.AnyUInt32] | _UInt32Code

# uint64
_UInt64Name: _Type = _Lit['uint64']
_UInt64Char: _Type = _Lit['u8', '=u8', '<u8', '>u8']
_UInt64Code: _Type = _UInt64Name | _UInt64Char
AnyUInt64DType: _Type = _Any2[np.uint64, _s.AnyUInt64] | _UInt64Code

# uintp (assuming that `uint_ptr_t == size_t`, as done in `numpy.typing`)
if _NP_V2:
    _UIntPName: _Type = _Lit['uintp', 'uint']
    _UIntPChar: _Type = _Lit['N', '=N', '<N', '>N']
else:
    _UIntPName: _Type = _Lit['uintp', 'uint0']
    _UIntPChar: _Type = _Lit['P', '=P', '<P', '>P']
_UIntPCode: _Type = _UIntPName | _UIntPChar
AnyUIntPDType: _Type = _Any2[np.uintp, _s.AnyUIntP | _ct.UInt0] | _UIntPCode

# ubyte
_UByteName: _Type = _Lit['ubyte']
_UByteChar: _Type = _Lit['B', '=B', '<B', '>B']
_UByteCode: _Type = _UByteName | _UByteChar
AnyUByteDType: _Type = _Any2[np.ubyte, _s.AnyUByte] | _UByteCode

# ushort
_UShortName: _Type = _Lit['ushort']
_UShortChar: _Type = _Lit['H', '=H', '<H', '>H']
_UShortCode: _Type = _UShortName | _UShortChar
AnyUShortDType: _Type = _Any2[np.ushort, _s.AnyUShort] | _UShortCode

# uintc
_UIntCName: _Type = _Lit['uintc']
_UIntCChar: _Type = _Lit['I', '=I', '<I', '>I']
_UIntCCode: _Type = _UIntCName | _UIntCChar
AnyUIntCDType: _Type = _Any2[np.uintc, _s.AnyUIntC] | _UIntCCode

# ulong (uint if numpy<2)
if _NP_V2:
    _ULongName: _Type = _Lit['ulong']
else:
    _ULongName: _Type = _Lit['ulong', 'uint']
_ULongChar: _Type = _Lit['L', '=L', '<L', '>L']
_ULongCode: _Type = _ULongName | _ULongChar
AnyULongDType: _Type = _Any2[_x.ULong, _s.AnyULong] | _ULongCode

# ulonglong
_ULongLongName: _Type = _Lit['ulonglong']
_ULongLongChar: _Type = _Lit['Q', '=Q', '<Q', '>Q']
_ULongLongCode: _Type = _ULongLongName | _ULongLongChar
AnyULongLongDType: _Type = (
    _Any2[np.ulonglong, _s.AnyULongLong]
    | _ULongLongCode
)

# unsignedinteger
# fmt: off
_UnsignedIntegerCode: _Type = (
    _UInt8Code | _UInt16Code | _UInt32Code | _UInt64Code | _UIntPCode
    | _UByteCode | _UShortCode | _UIntCCode | _ULongCode | _ULongLongCode
)
# fmt: on
AnyUnsignedIntegerDType: _Type = (
    _Any2[np.unsignedinteger[Any], _s.AnyUnsignedInteger | _ct.UInt0]
    | _UnsignedIntegerCode
)


#
# integer - signed
#

# int8
_Int8Name: _Type = _Lit['int8']
_Int8Char: _Type = _Lit['i1', '=i1', '<i1', '>i1']
_Int8Code: _Type = _Int8Name | _Int8Char
AnyInt8DType: _Type = _Any2[np.int8, _s.AnyInt8] | _Int8Code

# int16
_Int16Name: _Type = _Lit['int16']
_Int16Char: _Type = _Lit['i2', '=i2', '<i2', '>i2']
_Int16Code: _Type = _Int16Name | _Int16Char
AnyInt16DType: _Type = _Any2[np.int16, _s.AnyInt16] | _Int16Code

# int32
_Int32Name: _Type = _Lit['int32']
_Int32Char: _Type = _Lit['i4', '=i4', '<i4', '>i4']
_Int32Code: _Type = _Int32Name | _Int32Char
AnyInt32DType: _Type = _Any2[np.int32, _s.AnyInt32] | _Int32Code

# int64
_Int64Name: _Type = _Lit['int64']
_Int64Char: _Type = _Lit['i8', '=i8', '<i8', '>i8']
_Int64Code: _Type = _Int64Name | _Int64Char
AnyInt64DType: _Type = _Any2[np.int64, _s.AnyInt64] | _Int64Code

# intp
if _NP_V2:
    _IntPName: _Type = _Lit['intp', 'int', 'int_']
    _IntPChar: _Type = _Lit['n', '=n', '<n', '>n']
else:
    _IntPName: _Type = _Lit['intp', 'int0']
    _IntPChar: _Type = _Lit['p', '=p', '<p', '>p']
_IntPCode: _Type = _IntPName | _IntPChar
AnyIntPDType: _Type = _Any2[np.intp, _s.AnyIntP] | _IntPCode

# byte
_ByteName: _Type = _Lit['byte']
_ByteChar: _Type = _Lit['b', '=b', '<b', '>b']
_ByteCode: _Type = _ByteName | _ByteChar
AnyByteDType: _Type = _Any2[np.byte, _s.AnyByte] | _ByteCode

# short
_ShortName: _Type = _Lit['short']
_ShortChar: _Type = _Lit['h', '=h', '<h', '>h']
_ShortCode: _Type = _ShortName | _ShortChar
AnyShortDType: _Type = _Any2[np.short, _s.AnyShort] | _ShortCode

# intc
_IntCName: _Type = _Lit['intc']
_IntCChar: _Type = _Lit['i', '=i', '<i', '>i']
_IntCCode: _Type = _IntCName | _IntCChar
AnyIntCDType: _Type = _Any2[np.intc, _s.AnyIntC] | _IntCCode

# long (or int_ if numpy<2)
if _NP_V2:
    _LongName: _Type = _Lit['long']
else:
    _LongName: _Type = _Lit['long', 'int', 'int_']
_LongChar: _Type = _Lit['l', '=l', '<l', '>l']
_LongCode: _Type = _LongName | _LongChar
AnyLongDType: _Type = _Any2[_x.Long, _s.AnyLong] | _LongCode

# longlong
_LongLongName: _Type = _Lit['longlong']
_LongLongChar: _Type = _Lit['q', '=q', '<q', '>q']
_LongLongCode: _Type = _LongLongName | _LongLongChar
AnyLongLongDType: _Type = _Any2[np.longlong, _s.AnyLongLong] | _LongLongCode

# signedinteger
# fmt: off
_SignedIntegerCode: _Type = (
    _Int8Code | _Int16Code | _Int32Code | _Int64Code | _IntPCode
    | _ByteCode | _ShortCode | _IntCCode | _LongCode | _LongLongCode
)
# fmt: on
AnySignedIntegerDType: _Type = (
    _Any2[np.signedinteger[Any], _s.AnySignedInteger]
    | _SignedIntegerCode
)


#
# floating point - real
#

# float16
_Float16Name: _Type = _Lit['float16']
_Float16Char: _Type = _Lit['f2', '=f2', '<f2', '>f2']
_Float16Code: _Type = _Float16Name | _Float16Char
# there exists no `ct.c_half`
AnyFloat16DType: _Type = _Any1[np.float16] | _Float16Code

# float32
_Float32Name: _Type = _Lit['float32']
_Float32Char: _Type = _Lit['f4', '=f4', '<f4', '>f4']
_Float32Code: _Type = _Float32Name | _Float32Char
AnyFloat32DType: _Type = _Any2[np.float32, _s.AnyFloat32] | _Float32Code

# float64
_Float64Name: _Type = _Lit['float64']
_Float64Char: _Type = _Lit['f8', '=f8', '<f8', '>f8']
_Float64Code: _Type = _Float64Name | _Float64Char
# np.dtype(None) -> np.float64
AnyFloat64DType: _Type = None | _Any2[np.float64, _s.AnyFloat64] | _Float64Code

# half
_HalfName: _Type = _Lit['half']
_HalfChar: _Type = _Lit['e', '=e', '<e', '>e']
_HalfCode: _Type = _HalfName | _HalfChar
AnyHalfDType: _Type = _Any1[np.half] | _HalfCode

# single
_SingleName: _Type = _Lit['single']
_SingleChar: _Type = _Lit['f', '=f', '<f', '>f']
_SingleCode: _Type = _SingleName | _SingleChar
AnySingleDType: _Type = _Any2[np.single, _s.AnySingle] | _SingleCode

# double
if _NP_V2:
    _DoubleName: _Type = _Lit['double', 'float']
else:
    _DoubleName: _Type = _Lit['double', 'float', 'float_']
_DoubleChar: _Type = _Lit['d', '=d', '<d', '>d']
_DoubleCode: _Type = _DoubleName | _DoubleChar
AnyDoubleDType: _Type = _Any2[np.double, _s.AnyDouble] | _DoubleCode

# longdouble
if _NP_V2:
    _LongDoubleName: _Type = _Lit['longdouble']
else:
    _LongDoubleName: _Type = _Lit['longdouble', 'longfloat']
_LongDoubleChar: _Type = _Lit['g', '=g', '<g', '>g']
_LongDoubleCode: _Type = _LongDoubleName | _LongDoubleChar
AnyLongDoubleDType: _Type = (
    _Any2[np.longdouble, _s.AnyLongDouble | _ct.LongDouble0]
    | _LongDoubleCode
)

# floating
# fmt: off
_FloatingCode: _Type = (
    _Float16Code | _Float32Code | _Float64Code
    | _HalfCode | _SingleCode | _DoubleCode | _LongDoubleCode
)
# fmt: on
AnyFloatingDType: _Type = (
    _Any2[np.floating[Any], _s.AnyFloating]
    | _FloatingCode
)


#
# floating point - complex
#

# complex64
_Complex64Name: _Type = _Lit['complex64']
_Complex64Char: _Type = _Lit['c8', '=c8', '<c8', '>c8']
_Complex64Code: _Type = _Complex64Name | _Complex64Char
AnyComplex64DType: _Type = _Any1[np.complex64] | _Complex64Code

# complex128
_Complex128Name: _Type = _Lit['complex128']
_Complex128Char: _Type = _Lit['c16', '=c16', '<c16', '>c16']
_Complex128Code: _Type = _Complex128Name | _Complex128Char
AnyComplex128DType: _Type = (
    _Any2[np.complex128, _s.AnyComplex128]
    | _Complex128Code
)

# csingle
if _NP_V2:
    _CSingleName: _Type = _Lit['csingle']
else:
    _CSingleName: _Type = _Lit['csingle', 'singlecomplex']
_CSingleChar: _Type = _Lit['F', '=F', '<F', '>F']
_CSingleCode: _Type = _CSingleName | _CSingleChar
AnyCSingleDType: _Type = _Any1[np.csingle] | _CSingleCode

# cdouble
if _NP_V2:
    _CDoubleName: _Type = _Lit['cdouble', 'complex']
else:
    _CDoubleName: _Type = _Lit['cdouble', 'cfloat', 'complex', 'complex_']
_CDoubleChar: _Type = _Lit['D', '=D', '<D', '>D']
_CDoubleCode: _Type = _CDoubleName | _CDoubleChar
AnyCDoubleDType: _Type = _Any2[np.cdouble, _s.AnyCDouble] | _CDoubleCode

# clongdouble
if _NP_V2:
    _CLongDoubleName: _Type = _Lit['clongdouble']
else:
    _CLongDoubleName: _Type = _Lit[
        'clongdouble',
        'clongfloat',
        'longcomplex',
    ]
_CLongDoubleChar: _Type = _Lit['G', '=G', '<G', '>G']
_CLongDoubleCode: _Type = _CLongDoubleName | _CLongDoubleChar
AnyCLongDoubleDType: _Type = _Any1[np.clongdouble] | _CLongDoubleCode

# complexfloating
# fmt: off
_ComplexFloatingCode: _Type = (
    _Complex64Code | _Complex128Code
    | _CSingleCode | _CDoubleCode | _CLongDoubleCode
)
# fmt: on
AnyComplexFloatingDType: _Type = (
    _Any2[np.complexfloating[Any, Any], _s.AnyComplexFloating]
    | _ComplexFloatingCode
)


#
# temporal
#

# datetime64
_DateTime64Name: _Type = _Lit['datetime64']
# fmt: off
_DateTime64Char: _Type = _Lit[
    'M', '=M', '<M', '>M',
    'M8', '=M8', '<M8', '>M8',
]
# fmt: on
_DateTime64Code: _Type = _DateTime64Name | _DateTime64Char
AnyDateTime64DType: _Type = _Any1[np.datetime64] | _DateTime64Code

# timedelta64
_TimeDelta64Name: _Type = _Lit['timedelta64']
# fmt: off
_TimeDelta64Char: _Type = _Lit[
    'm', '=m', '<m', '>m',
    'm8', '=m8', '<m8', '>m8',
]
# fmt: on
_TimeDelta64Code: _Type = _TimeDelta64Name | _TimeDelta64Char
AnyTimeDelta64DType: _Type = _Any1[np.timedelta64] | _TimeDelta64Code


#
# flexible - characters
#

# str
if _NP_V2:
    _StrName: _Type = _Lit['str', 'str_', 'unicode']
else:
    _StrName: _Type = _Lit['str', 'str_', 'str0', 'unicode', 'unicode_']
_StrChar: _Type = _Lit['U', '=U', '<U', '>U']
_StrCode: _Type = _StrName | _StrChar
AnyStrDType: _Type = _Any2[np.str_, _s.AnyStr] | _StrCode

# bytes
if _NP_V2:
    _BytesName: _Type = _Lit['bytes', 'bytes_']
else:
    _BytesName: _Type = _Lit['bytes', 'bytes_', 'bytes0']
# TODO: Figure out why `np.dtype(ct.c_xhar) != np.dtype(np.bytes_)`,
# whose `.char` attrs are `'c' != 'S'`, `.itemsize` are `1 != 0`, `.isbuiltin`
# are `0 != 1`, and `.name` are `'bytes8' != 'bytes'`
_BytesChar: _Type = _Lit['S', '=S', '<S', '>S', 'S0', '=S0', '<S0', '>S0']
_BytesCode: _Type = _BytesName | _BytesChar
AnyBytesDType: _Type = _Any2[np.bytes_, _s.AnyBytes] | _BytesCode

# character
_CharacterCode: _Type = _StrCode | _BytesCode
AnyCharacterDType: _Type = (
    _Any2[np.character, _s.AnyCharacter] |
    _CharacterCode
)


#
# flexible
#

# void
if _NP_V2:
    _VoidName: _Type = _Lit['void']
else:
    _VoidName: _Type = _Lit['void', 'void0']
_VoidChar: _Type = _Lit['V', '=V', '<V', '>V']
_VoidCode: _Type = _VoidName | _VoidChar
AnyVoidDType: _Type = _Any1[np.void] | _VoidCode

# flexible
_FlexibleCode: _Type = _CharacterCode | _VoidCode
AnyFlexibleDType: _Type = _Any2[np.flexible, _s.AnyFlexible] | _FlexibleCode


#
# other
#

# bool_
if _NP_V2:
    _BoolName: _Type = _Lit['bool', 'bool_']
else:
    _BoolName: _Type = _Lit['bool', 'bool_', 'bool8']
_BoolChar: _Type = _Lit['?', '=?', '<?', '>?']
_BoolCode: _Type = _BoolName | _BoolChar
AnyBoolDType: _Type = _Any2[_x.Bool, _s.AnyBool] | _BoolCode

# object
_ObjectName: _Type = _Lit['object', 'object_']
_ObjectChar: _Type = _Lit['O', '=O', '<O', '>O']
_ObjectCode: _Type = _ObjectName | _ObjectChar
AnyObjectDType: _Type = _Any2[np.object_, _s.AnyObject] | _ObjectCode


#
# abstract
#

# integer
_IntegerCode: _Type = _UnsignedIntegerCode | _SignedIntegerCode
AnyIntegerDType: _Type = _Any2[np.integer[Any], _s.AnyInteger] | _IntegerCode

# inexact
_InexactCode: _Type = _FloatingCode | _ComplexFloatingCode
AnyInexactDType: _Type = _Any2[np.inexact[Any], _s.AnyInexact] | _InexactCode

# number
_NumberCode: _Type = _IntegerCode | _InexactCode
AnyNumberDType: _Type = _Any2[np.number[Any], _s.AnyNumber] | _NumberCode

# generic
_GenericCode: _Type = (
    _BoolCode
    | _NumberCode
    | _DateTime64Code
    | _TimeDelta64Code
    | _FlexibleCode
    | _ObjectCode
)
AnyGenericDType: _Type = _Any2[np.generic, _s.AnyGeneric] | _GenericCode

ST = TypeVar('ST', bound=np.generic, default=Any)
AnyDType: _Type = _Any2[ST, complex | str | bytes] | _GenericCode | type
