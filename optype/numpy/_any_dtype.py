"""
The allowed `np.dtype` arguments for specific scalar types.
The names are analogous to those in `numpy.dtypes`.
"""
from __future__ import annotations

import sys
from typing import Any, Literal as _Lit, TypeAlias as _Type

import numpy as np

import optype.numpy._compat as _x
import optype.numpy._ctype as _ct
import optype.numpy._dtype as _dt


if sys.version_info >= (3, 13):
    from typing import Never, TypeVar
else:
    from typing_extensions import Never, TypeVar


# ruff: noqa: RUF022
__all__: list[str] = []
__all__ += [
    'AnyGenericDType',
    'AnyNumberDType',
    'AnyIntegerDType',
    'AnyUnsignedIntegerDType',
    'AnySignedIntegerDType',
    'AnyInexactDType',
    'AnyFloatingDType',
    'AnyComplexFloatingDType',
    'AnyFlexibleDType',
    'AnyCharacterDType',
]
__all__ += [
    'AnyUInt8DType',
    'AnyUInt8DType',
    'AnyUInt16DType',
    'AnyUInt32DType',
    'AnyUInt64DType',
    'AnyUIntPDType',
    'AnyUByteDType',
    'AnyUShortDType',
    'AnyUIntCDType',
    'AnyULongDType',
    'AnyULongLongDType',
]
__all__ += [
    'AnyIntegerDType',
    'AnyInt8DType',
    'AnyInt8DType',
    'AnyInt16DType',
    'AnyInt32DType',
    'AnyInt64DType',
    'AnyIntPDType',
    'AnyByteDType',
    'AnyShortDType',
    'AnyIntCDType',
    'AnyLongDType',
    'AnyLongLongDType',
]
__all__ += [
    'AnyFloatingDType',
    'AnyFloat16DType',
    'AnyFloat32DType',
    'AnyFloat64DType',
    'AnyHalfDType',
    'AnySingleDType',
    'AnyLongDoubleDType',
]
__all__ += [
    'AnyComplex64DType',
    'AnyComplex128DType',
    'AnyCSingleDType',
    'AnyCDoubleDType',
    'AnyCLongDoubleDType',
]
__all__ += [
    'AnyBytesDType',
    'AnyStrDType',
    'AnyVoidDType',
]
__all__ += [
    'AnyBoolDType',
    'AnyDateTime64DType',
    'AnyTimeDelta64DType',
    'AnyObjectDType',
]
__all__ += ['AnyStringDType']


# helper aliases
_ST = TypeVar('_ST', bound=np.generic)
_VT = TypeVar('_VT')
_Any1: _Type = np.dtype[_ST] | type[_ST] | _dt.HasDType[np.dtype[_ST]]
_Any2: _Type = np.dtype[_ST] | type[_ST | _VT] | _dt.HasDType[np.dtype[_ST]]


#
# integer - unsigned
# TODO: Merge the aliases
#

_UInt8Name: _Type = _Lit['uint8']
_UInt8Char: _Type = _Lit['u1', '|u1', '=u1', '<u1', '>u1']
_UInt8Code: _Type = _Lit[_UInt8Name, _UInt8Char]
AnyUInt8DType: _Type = _Any2[np.uint8, _ct.UInt8] | _UInt8Code

# uint16
_UInt16Name: _Type = _Lit['uint16']
_UInt16Char: _Type = _Lit['u2', '|u2', '=u2', '<u2', '>u2']
_UInt16Code: _Type = _Lit[_UInt16Name, _UInt16Char]
AnyUInt16DType: _Type = _Any2[np.uint16, _ct.UInt16] | _UInt16Code

# uint32
_UInt32Name: _Type = _Lit['uint32']
_UInt32Char: _Type = _Lit['u4', '|u4', '=u4', '<u4', '>u4']
_UInt32Code: _Type = _Lit[_UInt32Name, _UInt32Char]
AnyUInt32DType: _Type = _Any2[np.uint32, _ct.UInt32] | _UInt32Code

# uint64
_UInt64Name: _Type = _Lit['uint64']
_UInt64Char: _Type = _Lit['u8', '|u8', '=u8', '<u8', '>u8']
_UInt64Code: _Type = _Lit[_UInt64Name, _UInt64Char]
AnyUInt64DType: _Type = _Any2[np.uint64, _ct.UInt64] | _UInt64Code

# uintp (assuming that `uint_ptr_t == size_t`)
# NOTE: Since numpy 2.0, it's guaranteed to be 32 bits on 32-bit systems and
# 64 on 64-bit, but we assume that this also would've held on numpy 1.
_UInt0Char: _Type = _Lit['P', '|P', '=P', '<P', '>P']
if _x.NP2:
    _UIntPName: _Type = _Lit['uintp', 'uint']
    _UIntPChar: _Type = _Lit[_UInt0Char, 'N', '|N', '=N', '<N', '>N']
    _UIntPCode: _Type = _Lit[_UIntPName, _UIntPChar]
    AnyUIntPDType: _Type = _Any2[np.uintp, _ct.UIntP] | _UIntPCode
else:
    _UIntPName: _Type = _Lit['uintp']  # 'uint0' is removed in NumPy 2.0
    _UIntPChar: _Type = _UInt0Char
    _UIntPCode: _Type = _Lit[_UIntPName, _UIntPChar]
    AnyUIntPDType: _Type = _Any2[np.uintp, _ct.UIntP] | _UIntPCode

# ubyte / uint8 (assuming that byte == int8)
_UByteName: _Type = _Lit['ubyte']
_UByteChar: _Type = _Lit['B', '|B', '=B', '<B', '>B']
_UByteCode: _Type = _Lit[_UByteName, _UByteChar]
AnyUByteDType: _Type = _Any2[np.ubyte, _ct.UByte] | _UByteCode

# ushort
_UShortName: _Type = _Lit['ushort']
_UShortChar: _Type = _Lit['H', '|H', '=H', '<H', '>H']
_UShortCode: _Type = _Lit[_UShortName, _UShortChar]
AnyUShortDType: _Type = _Any2[np.ushort, _ct.UShort] | _UShortCode

# uintc
_UIntCName: _Type = _Lit['uintc']
_UIntCChar: _Type = _Lit['I', '|I', '=I', '<I', '>I']
_UIntCCode: _Type = _Lit[_UIntCName, _UIntCChar]
AnyUIntCDType: _Type = _Any2[np.uintc, _ct.UIntC] | _UIntCCode

# ulong (uint if numpy<2)
_ULongChar: _Type = _Lit['L', '|L', '=L', '<L', '>L']
if _x.NP2:
    _ULongName: _Type = _Lit['ulong']
    _ULongCode: _Type = _Lit[_ULongName, _ULongChar]
    AnyULongDType: _Type = _Any2[_x.ULong, _ct.ULong] | _ULongCode
else:
    _ULongName: _Type = _Lit['ulong', 'uint']
    _ULongCode: _Type = _Lit[_ULongName, _ULongChar]
    AnyULongDType: _Type = _Any2[_x.ULong, _ct.ULong] | _ULongCode

# ulonglong
_ULongLongName: _Type = _Lit['ulonglong']
_ULongLongChar: _Type = _Lit['Q', '|Q', '=Q', '<Q', '>Q']
_ULongLongCode: _Type = _Lit[_ULongLongName, _ULongLongChar]
AnyULongLongDType: _Type = _Any2[np.ulonglong, _ct.ULongLong] | _ULongLongCode


#
# integer - signed
# TODO: Merge the aliases
#

# int8
_Int8Name: _Type = _Lit['int8']
_Int8Char: _Type = _Lit['i1', '|i1', '=i1', '<i1', '>i1']
_Int8Code: _Type = _Int8Name | _Int8Char
AnyInt8DType: _Type = _Any2[np.int8, _ct.Int8] | _Int8Code

# int16
_Int16Name: _Type = _Lit['int16']
_Int16Char: _Type = _Lit['i2', '|i2', '=i2', '<i2', '>i2']
_Int16Code: _Type = _Int16Name | _Int16Char
AnyInt16DType: _Type = _Any2[np.int16, _ct.Int16] | _Int16Code

# int32
_Int32Name: _Type = _Lit['int32']
_Int32Char: _Type = _Lit['i4', '|i4', '=i4', '<i4', '>i4']
_Int32Code: _Type = _Int32Name | _Int32Char
AnyInt32DType: _Type = _Any2[np.int32, _ct.Int32] | _Int32Code

# int64
_Int64Name: _Type = _Lit['int64']
_Int64Char: _Type = _Lit['i8', '|i8', '=i8', '<i8', '>i8']
_Int64Code: _Type = _Int64Name | _Int64Char
AnyInt64DType: _Type = _Any2[np.int64, _ct.Int64] | _Int64Code

# intp
# (`AnyIntPDType` must be inside each block, for valid typing)
if _x.NP2:
    _IntPName: _Type = _Lit['intp', 'int', 'int_']
    _IntPChar: _Type = _Lit['n', '|n', '=n', '<n', '>n']
    _IntPCode: _Type = _IntPName | _IntPChar
    AnyIntPDType: _Type = _Any2[np.intp, _ct.IntP] | _IntPCode
else:
    _IntPName: _Type = _Lit['intp']  # 'int0' is removed in NumPy 2.0
    _IntPChar: _Type = _Lit['p', '|p', '=p', '<p', '>p']
    _IntPCode: _Type = _IntPName | _IntPChar
    AnyIntPDType: _Type = _Any2[np.intp, _ct.IntP] | _IntPCode

# byte
_ByteName: _Type = _Lit['byte']
_ByteChar: _Type = _Lit['b', '|b', '=b', '<b', '>b']
_ByteCode: _Type = _ByteName | _ByteChar
AnyByteDType: _Type = _Any2[np.byte, _ct.Byte] | _ByteCode

# short
_ShortName: _Type = _Lit['short']
_ShortChar: _Type = _Lit['h', '|h', '=h', '<h', '>h']
_ShortCode: _Type = _ShortName | _ShortChar
AnyShortDType: _Type = _Any2[np.short, _ct.Short] | _ShortCode

# intc
_IntCName: _Type = _Lit['intc']
_IntCChar: _Type = _Lit['i', '|i', '=i', '<i', '>i']
_IntCCode: _Type = _IntCName | _IntCChar
AnyIntCDType: _Type = _Any2[np.intc, _ct.IntC] | _IntCCode

# long (or int_ if numpy<2)
_LongChar: _Type = _Lit['l', '|l', '=l', '<l', '>l']
if _x.NP2:
    _LongName: _Type = _Lit['long']
    _LongCode: _Type = _LongName | _LongChar
    AnyLongDType: _Type = _Any2[_x.Long, _ct.Long] | _LongCode
else:
    _LongName: _Type = _Lit['long', 'int', 'int_']
    _LongCode: _Type = _LongName | _LongChar
    AnyLongDType: _Type = _Any2[_x.Long, _ct.Long] | _LongCode

# longlong
_LongLongName: _Type = _Lit['longlong']
_LongLongChar: _Type = _Lit['q', '|q', '=q', '<q', '>q']
_LongLongCode: _Type = _LongLongName | _LongLongChar
AnyLongLongDType: _Type = _Any2[np.longlong, _ct.LongLong] | _LongLongCode


#
# floating point - real
# TODO: Merge the aliases
#

# float16
_Float16Name: _Type = _Lit['float16']
_Float16Char: _Type = _Lit['f2', '|f2', '=f2', '<f2', '>f2']
_Float16Code: _Type = _Float16Name | _Float16Char
# there exists no `ct.c_half`
AnyFloat16DType: _Type = _Any1[np.float16] | _Float16Code

# float32
_Float32Name: _Type = _Lit['float32']
_Float32Char: _Type = _Lit['f4', '|f4', '=f4', '<f4', '>f4']
_Float32Code: _Type = _Lit[_Float32Name, _Float32Char]
AnyFloat32DType: _Type = _Any2[np.float32, _ct.Float32] | _Float32Code

# float64
_Float64Name: _Type = _Lit['float64']
_Float64Char: _Type = _Lit['f8', '|f8', '=f8', '<f8', '>f8']
_Float64Code: _Type = _Lit[_Float64Name, _Float64Char]
# np.dtype(None) -> np.float64
AnyFloat64DType: _Type = None | _Any2[np.float64, _ct.Float64] | _Float64Code

# half
_HalfName: _Type = _Lit['half']
_HalfChar: _Type = _Lit['e', '|e', '=e', '<e', '>e']
_HalfCode: _Type = _Lit[_HalfName, _HalfChar]
AnyHalfDType: _Type = _Any1[np.half] | _HalfCode

# single
_SingleName: _Type = _Lit['single']
_SingleChar: _Type = _Lit['f', '|f', '=f', '<f', '>f']
_SingleCode: _Type = _Lit[_SingleName, _SingleChar]
AnySingleDType: _Type = _Any2[np.single, _ct.Single] | _SingleCode

# double
# ('float_' was removed in NumPy 2.0)
_DoubleName: _Type = _Lit['double', 'float']
_DoubleChar: _Type = _Lit['d', '|d', '=d', '<d', '>d']
_DoubleCode: _Type = _Lit[_DoubleName, _DoubleChar]
AnyDoubleDType: _Type = _Any2[np.double, _ct.Double] | _DoubleCode

# longdouble
# ('longfloat' was removed in NumPy 2.0)
_LongDoubleName: _Type = _Lit['longdouble']
_LongDoubleChar: _Type = _Lit['g', '|g', '=g', '<g', '>g']
_LongDoubleCode: _Type = _Lit[_LongDoubleName, _LongDoubleChar]
AnyLongDoubleDType: _Type = (
    _Any2[np.longdouble, _ct.LongDouble] | _LongDoubleCode
)

# floating
_FloatingCode: _Type = (
    _Float16Code | _Float32Code | _Float64Code
    | _HalfCode | _SingleCode | _DoubleCode | _LongDoubleCode
)  # fmt: skip
AnyFloatingDType: _Type = (
    _Any2[np.floating[Any], _ct.Floating]
    | _FloatingCode
)

#
# floating point - complex
# TODO: Merge the aliases
#

# complex64
_Complex64Name: _Type = _Lit['complex64']
_Complex64Char: _Type = _Lit['c8', '|c8', '=c8', '<c8', '>c8']
_Complex64Code: _Type = _Lit[_Complex64Name, _Complex64Char]
AnyComplex64DType: _Type = _Any1[np.complex64] | _Complex64Code

# complex128
_Complex128Name: _Type = _Lit['complex128']
_Complex128Char: _Type = _Lit['c16', '|c16', '=c16', '<c16', '>c16']
_Complex128Code: _Type = _Lit[_Complex128Name, _Complex128Char]
# NOTE: There's no `type[complex]` here, as that would also include `float`,
# `int`, and `bool`!
AnyComplex128DType: _Type = _Any1[np.complex128] | _Complex128Code

# csingle
# ('singlecomplex' was removed in NumPy 2.0)
_CSingleName: _Type = _Lit['csingle']
_CSingleChar: _Type = _Lit['F', '|F', '=F', '<F', '>F']
_CSingleCode: _Type = _Lit[_CSingleName, _CSingleChar]
AnyCSingleDType: _Type = _Any1[np.csingle] | _CSingleCode

# cdouble
# ('complex_' and 'cfloat' were removed in NumPy 2.0)
_CDoubleName: _Type = _Lit['cdouble', 'complex']
_CDoubleChar: _Type = _Lit['D', '|D', '=D', '<D', '>D']
_CDoubleCode: _Type = _Lit[_CDoubleName, _CDoubleChar]
AnyCDoubleDType: _Type = _Any1[np.cdouble] | _CDoubleCode

# clongdouble
# ('clongfloat' and 'longcomplex' were removed in NumPy 2.0)
_CLongDoubleName: _Type = _Lit['clongdouble']
_CLongDoubleChar: _Type = _Lit['G', '|G', '=G', '<G', '>G']
_CLongDoubleCode: _Type = _Lit[_CLongDoubleName, _CLongDoubleChar]
AnyCLongDoubleDType: _Type = _Any1[np.clongdouble] | _CLongDoubleCode

# complexfloating
_ComplexFloatingCode: _Type = _Lit[
    _Complex64Code,
    _Complex128Code,
    _CSingleCode,
    _CDoubleCode,
    _CLongDoubleCode,
]
AnyComplexFloatingDType: _Type = (
    _Any1[np.complexfloating[Any, Any]] | _ComplexFloatingCode
)


#
# temporal
#

# datetime64
_DateTime64Name: _Type = _Lit[
    'datetime64',
    'datetime64[as]',
    'datetime64[fs]',
    'datetime64[ps]',
    'datetime64[ns]',
    'datetime64[us]',
    'datetime64[ms]',
    'datetime64[s]',
    'datetime64[m]',
    'datetime64[h]',
    'datetime64[D]',
    'datetime64[W]',
    'datetime64[M]',
    'datetime64[Y]',
]
_DateTime64Char: _Type = _Lit[
    'M', '|M', '=M', '<M', '>M',
    'M8', '|M8', '=M8', '<M8', '>M8',
    'M8[as]', '|M8[as]', '=M8[as]', '<M8[as]', '>M8[as]',
    'M8[fs]', '|M8[fs]', '=M8[fs]', '<M8[fs]', '>M8[fs]',
    'M8[ps]', '|M8[ps]', '=M8[ps]', '<M8[ps]', '>M8[ps]',
    'M8[ns]', '|M8[ns]', '=M8[ns]', '<M8[ns]', '>M8[ns]',
    'M8[us]', '|M8[us]', '=M8[us]', '<M8[us]', '>M8[us]',
    'M8[s]', '|M8[s]', '=M8[s]', '<M8[s]', '>M8[s]',
    'M8[m]', '|M8[m]', '=M8[m]', '<M8[m]', '>M8[m]',
    'M8[h]', '|M8[h]', '=M8[h]', '<M8[h]', '>M8[h]',
    'M8[D]', '|M8[D]', '=M8[D]', '<M8[D]', '>M8[D]',
    'M8[W]', '|M8[W]', '=M8[W]', '<M8[W]', '>M8[W]',
    'M8[M]', '|M8[M]', '=M8[M]', '<M8[M]', '>M8[M]',
    'M8[Y]', '|M8[Y]', '=M8[Y]', '<M8[Y]', '>M8[Y]',
]  # fmt: skip
_DateTime64Code: _Type = _Lit[_DateTime64Name, _DateTime64Char]
AnyDateTime64DType: _Type = _Any1[np.datetime64] | _DateTime64Code

# timedelta64
_TimeDelta64Name: _Type = _Lit[
    'timedelta64',
    'timedelta64[as]',
    'timedelta64[fs]',
    'timedelta64[ps]',
    'timedelta64[ns]',
    'timedelta64[us]',
    'timedelta64[ms]',
    'timedelta64[s]',
    'timedelta64[m]',
    'timedelta64[h]',
    'timedelta64[D]',
    'timedelta64[W]',
    'timedelta64[M]',
    'timedelta64[Y]',
]
_TimeDelta64Char: _Type = _Lit[
    'm', '|m', '=m', '<m', '>m',
    'm8', '|m8', '=m8', '<m8', '>m8',
    'm8[as]', '|m8[as]', '=m8[as]', '<m8[as]', '>m8[as]',
    'm8[fs]', '|m8[fs]', '=m8[fs]', '<m8[fs]', '>m8[fs]',
    'm8[ps]', '|m8[ps]', '=m8[ps]', '<m8[ps]', '>m8[ps]',
    'm8[ns]', '|m8[ns]', '=m8[ns]', '<m8[ns]', '>m8[ns]',
    'm8[us]', '|m8[us]', '=m8[us]', '<m8[us]', '>m8[us]',
    'm8[s]', '|m8[s]', '=m8[s]', '<m8[s]', '>m8[s]',
    'm8[m]', '|m8[m]', '=m8[m]', '<m8[m]', '>m8[m]',
    'm8[h]', '|m8[h]', '=m8[h]', '<m8[h]', '>m8[h]',
    'm8[D]', '|m8[D]', '=m8[D]', '<m8[D]', '>m8[D]',
    'm8[W]', '|m8[W]', '=m8[W]', '<m8[W]', '>m8[W]',
    'm8[M]', '|m8[M]', '=m8[M]', '<m8[M]', '>m8[M]',
    'm8[Y]', '|m8[Y]', '=m8[Y]', '<m8[Y]', '>m8[Y]',
]  # fmt: skip
_TimeDelta64Code: _Type = _Lit[_TimeDelta64Name, _TimeDelta64Char]
AnyTimeDelta64DType: _Type = _Any1[np.timedelta64] | _TimeDelta64Code


#
# flexible - characters
#

# str
# ('str0' and `unicode_` were removed in NumPy 2.0)
_StrName: _Type = _Lit['str_', 'str', 'unicode']
_StrChar: _Type = _Lit[
    'U', '|U', '=U', '<U', '>U',
    'U0', '|U0', '=U0', '<U0', '>U0',
    'U1', '|U1', '=U1', '<U1', '>U1',
]
_StrCode: _Type = _Lit[_StrName, _StrChar]
# no ctypes `str_` exists
AnyStrDType: _Type = _Any2[np.str_, str] | _StrCode

# bytes
# ('bytes0' was removed in NumPy 2.0)
_BytesName: _Type = _Lit['bytes_', 'bytes']
_BytesChar: _Type = _Lit[
    'S', '|S', '=S', '<S', '>S',
    'S0', '|S0', '=S0', '<S0', '>S0',
    'S1', '|S1', '=S1', '<S1', '>S1',  # named 'bytes8', matches `ct.c_char`
]  # fmt: skip
_BytesCode: _Type = _Lit[_BytesName, _BytesChar]
AnyBytesDType: _Type = _Any2[np.bytes_, bytes | _ct.Bytes] | _BytesCode

# character
_CharacterCode: _Type = _Lit[_StrCode, _BytesCode]
AnyCharacterDType: _Type = (
    _Any2[np.character, str | bytes | _ct.Character] | _CharacterCode
)

# void
# TODO: Structured DTypes are missing!
_VoidName: _Type = _Lit['void']  # 'void0' was removed in NumPy 2.0
_VoidChar: _Type = _Lit[
    'V', '|V', '=V', '<V', '>V',
    'V0', '|V0', '=V0', '<V0', '>V0',
    'V1', '|V1', '=V1', '<V1', '>V1',
]
_VoidCode: _Type = _Lit[_VoidName, _VoidChar]
# no ctypes `void` exists
AnyVoidDType: _Type = _Any2[np.void, memoryview] | _VoidCode

# flexible
_FlexibleCode: _Type = _Lit[_CharacterCode, _VoidCode]
AnyFlexibleDType: _Type = (
    _Any2[np.flexible, bytes | str | memoryview | _ct.Flexible] | _FlexibleCode
)

#
# other
#

# bool_
_BoolName: _Type = _Lit['bool_', 'bool']  # 'bool0' was removed in NumPy 2.0
_BoolChar: _Type = _Lit['?', '|?', '=?', '<?', '>?']
_BoolCode: _Type = _Lit[_BoolName, _BoolChar]
# It's ok to use `bool` here
AnyBoolDType: _Type = _Any2[_x.Bool, bool | _ct.Bool] | _BoolCode

# object
_ObjectName: _Type = _Lit['object_', 'object']
_ObjectChar: _Type = _Lit['O', '|O', '=O', '<O', '>O']
_ObjectCode: _Type = _Lit[_ObjectName, _ObjectChar]
# NOTE: `builtins.object` isn't included, since this could lead to many bugs
# e.g. in NumPy 2.0: `dtype(type[str | float]) -> dtype[object_]`...
AnyObjectDType: _Type = _Any2[np.object_, _ct.Object] | _ObjectCode


#
# abstract
#

_SCT = TypeVar('_SCT', bound=np.generic, default=Any)

_InexactCode: _Type = _Lit[_FloatingCode, _ComplexFloatingCode]
AnyInexactDType: _Type = _Any2[np.inexact[Any], _ct.Inexact] | _InexactCode

_UnsignedIntegerName: _Type = _Lit[
    'uint8', 'uint16', 'uint32', 'uint64', 'uintp', 'uint',
    'ubyte', 'ushort', 'uintc', 'ulong', 'ulonglong',
]  # fmt: skip
_UnsignedIntegerCharCommon: _Type = _Lit[
    _UInt8Char,
    _UInt16Char,
    _UInt32Char,
    _UInt64Char,
    _UByteChar,
    _UShortChar,
    _UIntCChar,
    _ULongChar,
    _ULongLongChar,
    # not associated to any unsigned scalar type in numpy>=2.0
    _Lit['P', '|P', '=P', '<P', '>P'],
]

_SignedIntegerName: _Type = _Lit[
    'int8', 'int16', 'int32', 'int64', 'intp', 'int', 'int_',
    'byte', 'short', 'intc', 'long', 'longlong',
]  # fmt: skip
_SignedIntegerCharCommon: _Type = _Lit[
    _Int8Char,
    _Int16Char,
    _Int32Char,
    _Int64Char,
    _ByteChar,
    _ShortChar,
    _IntCChar,
    _LongChar,
    _LongLongChar,
    # not associated to any particular signed integer type in numpy>=2.0
    _Lit['p', '|p', '=p', '<p', '>p'],
]

# this duplicated mess is needed for valid types and numpy 1/2 compat
if _x.NP2:
    _UnsignedIntegerChar: _Type = _Lit[
        _UnsignedIntegerCharCommon,
        _Lit['N', '|N', '=N', '<N', '>N'],
    ]
    _UnsignedIntegerCode: _Type = _Lit[
        _UnsignedIntegerName,
        _UnsignedIntegerChar,
    ]
    AnyUnsignedIntegerDType: _Type = (
        _Any2[np.unsignedinteger[Any], _ct.UnsignedInteger]
        | _UnsignedIntegerCode
    )

    _SignedIntegerChar: _Type = _Lit[
        _SignedIntegerCharCommon,
        _Lit['n', '|n', '=n', '<n', '>n'],
    ]
    _SignedIntegerCode: _Type = _Lit[_SignedIntegerName, _SignedIntegerChar]
    # NOTE: We can't include `type[int]` here, because that would also
    # `type[bool]`, which will result in many bugs, as it already has done so
    # in numpy itself.
    AnySignedIntegerDType: _Type = (
        _Any2[np.signedinteger[Any], _ct.SignedInteger] | _SignedIntegerCode
    )

    _IntegerCode: _Type = _Lit[_UnsignedIntegerCode, _SignedIntegerCode]
    AnyIntegerDType: _Type = _Any2[np.integer[Any], _ct.Integer] | _IntegerCode

    _NumberCode: _Type = _Lit[_IntegerCode, _InexactCode]
    # NOTE: this doesn't include `int` or `float` or `complex`, since that
    # would autoamtically include `bool`.
    AnyNumberDType: _Type = _Any2[np.number[Any], _ct.Number] | _NumberCode

    _GenericCode: _Type = _Lit[
        _BoolCode,
        _NumberCode,
        _DateTime64Code,
        _TimeDelta64Code,
        _FlexibleCode,
        _ObjectCode,
    ]
    AnyDType: _Type = (
        np.dtype[_SCT]
        | _dt.HasDType[np.dtype[_SCT]]
        | type[Any]
        | _GenericCode
    )
    AnyGenericDType: _Type = AnyDType[np.generic]

else:
    _UnsignedIntegerChar: _Type = _UnsignedIntegerCharCommon
    _UnsignedIntegerCode: _Type = _Lit[
        _UnsignedIntegerName,
        _UnsignedIntegerChar,
    ]
    AnyUnsignedIntegerDType: _Type = (
        _Any2[np.unsignedinteger[Any], _ct.UnsignedInteger]
        | _UnsignedIntegerCode
    )

    _SignedIntegerChar: _Type = _SignedIntegerCharCommon
    _SignedIntegerCode: _Type = _Lit[_SignedIntegerName, _SignedIntegerChar]
    AnySignedIntegerDType: _Type = (
        _Any2[np.signedinteger[Any], _ct.SignedInteger] | _SignedIntegerCode
    )

    _IntegerCode: _Type = _Lit[_UnsignedIntegerCode, _SignedIntegerCode]
    AnyIntegerDType: _Type = _Any2[np.integer[Any], _ct.Integer] | _IntegerCode

    _NumberCode: _Type = _Lit[_IntegerCode, _InexactCode]
    AnyNumberDType: _Type = _Any2[np.number[Any], _ct.Number] | _NumberCode

    _GenericCode: _Type = _Lit[
        _BoolCode,
        _NumberCode,
        _DateTime64Code,
        _TimeDelta64Code,
        _FlexibleCode,
        _ObjectCode,
    ]
    AnyDType: _Type = (
        np.dtype[_SCT]
        | _dt.HasDType[np.dtype[_SCT]]
        | type[Any]
        | _GenericCode
    )
    AnyGenericDType: _Type = AnyDType[np.generic]


# NOTE: At the moment, `np.dtypes.StringDType.type: type[str]`, which is
# impossible (`dtype[str]` isn't valid: `str` isn't a `np.generic`)
_StringName = Never
_StringChar: _Type = _Lit['T', '|T', '=T', '<T', '>T']
_StringCode: _Type = _StringChar
if _x.NP2 and not _x.NP20:  # `numpy>=2.1`
    AnyStringDType: _Type = _dt.HasDType[np.dtypes.StringDType] | _StringCode
elif _x.NP2:  # `numpy>=2.0,<2.1`
    # NOTE: `np.dtypes.StringDType` had no annotations prior to 2.1, so:
    # I (@jorenham) added them :), see:
    # https://github.com/numpy/numpy/pull/27008
    AnyStringDType: _Type = np.dtype[Never] | _StringCode
else:  # `numpy<=2.0`
    AnyStringDType: _Type = Never
