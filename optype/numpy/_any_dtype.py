"""
The allowed `np.dtype` arguments for specific scalar types.
The names are analogous to those in `numpy.dtypes`.
"""
from __future__ import annotations

import sys
from typing import Literal as L, TypeAlias as Alias  # noqa: N817

import numpy as np

import optype.numpy._compat as _x
import optype.numpy._dtype as _dt
import optype.numpy._scalar as _sc
import optype.numpy.ctypeslib as _ct


if sys.version_info >= (3, 13):
    from typing import LiteralString, Never, TypeVar
else:
    from typing_extensions import LiteralString, Never, TypeVar


# ruff: noqa: RUF022
__all__ = [
    'AnyDType',
    'AnyNumberDType',
    'AnyIntegerDType',
    'AnyInexactDType',
    'AnyFlexibleDType',
    'AnyUnsignedIntegerDType',
    'AnySignedIntegerDType',
    'AnyFloatingDType',
    'AnyComplexFloatingDType',
    'AnyCharacterDType',

    'AnyBoolDType',

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

    'AnyFloat16DType',
    'AnyFloat32DType',
    'AnyFloat64DType',
    'AnyLongDoubleDType',

    'AnyComplex64DType',
    'AnyComplex128DType',
    'AnyCLongDoubleDType',

    'AnyDateTime64DType',
    'AnyTimeDelta64DType',

    'AnyBytesDType',
    'AnyStrDType',
    'AnyVoidDType',
    'AnyObjectDType',

    'AnyStringDType',
]


_ST = TypeVar('_ST', bound=np.generic)
_T = TypeVar('_T')
_Any1: Alias = np.dtype[_ST] | type[_ST] | _dt.HasDType[np.dtype[_ST]]
_Any2: Alias = np.dtype[_ST] | type[_ST | _T] | _dt.HasDType[np.dtype[_ST]]

# unsigned integers

# TODO: include ubyte
_UInt8Name: Alias = L['uint8']
_UInt8Char: Alias = L['u1', '|u1', '=u1', '<u1', '>u1']
_UInt8Code: Alias = L[_UInt8Name, _UInt8Char]
AnyUInt8DType: Alias = _Any2[np.uint8, _ct.UInt8] | _UInt8Code

# TODO: include ushort
_UInt16Name: Alias = L['uint16']
_UInt16Char: Alias = L['u2', '|u2', '=u2', '<u2', '>u2']
_UInt16Code: Alias = L[_UInt16Name, _UInt16Char]
AnyUInt16DType: Alias = _Any2[np.uint16, _ct.UInt16] | _UInt16Code

# TODO: include uintc
_UInt32Name: Alias = L['uint32']
_UInt32Char: Alias = L['u4', '|u4', '=u4', '<u4', '>u4']
_UInt32Code: Alias = L[_UInt32Name, _UInt32Char]
AnyUInt32DType: Alias = _Any2[np.uint32, _ct.UInt32] | _UInt32Code

_UInt64Name: Alias = L['uint64']
_UInt64Char: Alias = L['u8', '|u8', '=u8', '<u8', '>u8']
_UInt64Code: Alias = L[_UInt64Name, _UInt64Char]
AnyUInt64DType: Alias = _Any2[np.uint64, _ct.UInt64] | _UInt64Code

_UByteName: Alias = L['ubyte']
_UByteChar: Alias = L['B', '|B', '=B', '<B', '>B']
_UByteCode: Alias = L[_UByteName, _UByteChar]
AnyUByteDType: Alias = _Any2[np.ubyte, _ct.UByte] | _UByteCode

_UShortName: Alias = L['ushort']
_UShortChar: Alias = L['H', '|H', '=H', '<H', '>H']
_UShortCode: Alias = L[_UShortName, _UShortChar]
AnyUShortDType: Alias = _Any2[np.ushort, _ct.UShort] | _UShortCode

_UIntCName: Alias = L['uintc']
_UIntCChar: Alias = L['I', '|I', '=I', '<I', '>I']
_UIntCCode: Alias = L[_UIntCName, _UIntCChar]
AnyUIntCDType: Alias = _Any2[np.uintc, _ct.UIntC] | _UIntCCode

_UIntName: Alias = L['uint']
_ULongChar: Alias = L['L', '|L', '=L', '<L', '>L']

_ULongLongName: Alias = L['ulonglong']
_ULongLongChar: Alias = L['Q', '|Q', '=Q', '<Q', '>Q']
_ULongLongCode: Alias = L[_ULongLongName, _ULongLongChar]
AnyULongLongDType: Alias = _Any2[np.ulonglong, _ct.ULongLong] | _ULongLongCode

# signed integers

# TODO: include byte
_Int8Name: Alias = L['int8']
_Int8Char: Alias = L['i1', '|i1', '=i1', '<i1', '>i1']
_Int8Code: Alias = L[_Int8Name, _Int8Char]
AnyInt8DType: Alias = _Any2[np.int8, _ct.Int8] | _Int8Code

# TODO: include short
_Int16Name: Alias = L['int16']
_Int16Char: Alias = L['i2', '|i2', '=i2', '<i2', '>i2']
_Int16Code: Alias = L[_Int16Name, _Int16Char]
AnyInt16DType: Alias = _Any2[np.int16, _ct.Int16] | _Int16Code

# TODO: include intc
_Int32Name: Alias = L['int32']
_Int32Char: Alias = L['i4', '|i4', '=i4', '<i4', '>i4']
_Int32Code: Alias = L[_Int32Name, _Int32Char]
AnyInt32DType: Alias = _Any2[np.int32, _ct.Int32] | _Int32Code

_Int64Name: Alias = L['int64']
_Int64Char: Alias = L['i8', '|i8', '=i8', '<i8', '>i8']
_Int64Code: Alias = L[_Int64Name, _Int64Char]
AnyInt64DType: Alias = _Any2[np.int64, _ct.Int64] | _Int64Code

_ByteName: Alias = L['byte']
_ByteChar: Alias = L['b', '|b', '=b', '<b', '>b']
_ByteCode: Alias = L[_ByteName, _ByteChar]
AnyByteDType: Alias = _Any2[np.byte, _ct.Byte] | _ByteCode

_ShortName: Alias = L['short']
_ShortChar: Alias = L['h', '|h', '=h', '<h', '>h']
_ShortCode: Alias = L[_ShortName, _ShortChar]
AnyShortDType: Alias = _Any2[np.short, _ct.Short] | _ShortCode

_IntCName: Alias = L['intc']
_IntCChar: Alias = L['i', '|i', '=i', '<i', '>i']
_IntCCode: Alias = L[_IntCName, _IntCChar]
AnyIntCDType: Alias = _Any2[np.intc, _ct.IntC] | _IntCCode

_IntName: Alias = L['int', 'int_']
_LongChar: Alias = L['l', '|l', '=l', '<l', '>l']

_LongLongName: Alias = L['longlong']
_LongLongChar: Alias = L['q', '|q', '=q', '<q', '>q']
_LongLongCode: Alias = L[_LongLongName, _LongLongChar]
AnyLongLongDType: Alias = _Any2[np.longlong, _ct.LongLong] | _LongLongCode

# real floating

_Float16Name: Alias = L['float16', 'half']
_Float16Char: Alias = L['f2', '|f2', '=f2', '<f2', '>f2', 'e', '|e', '=e', '<e', '>e']
_Float16Code: Alias = L[_Float16Name, _Float16Char]
AnyFloat16DType: Alias = _Any1[np.float16] | _Float16Code

_Float32Name: Alias = L['float32', 'single']
_Float32Char: Alias = L['f4', '|f4', '=f4', '<f4', '>f4', 'f', '|f', '=f', '<f', '>f']
_Float32Code: Alias = L[_Float32Name, _Float32Char]
AnyFloat32DType: Alias = _Any2[np.float32, _ct.Float32] | _Float32Code

_Float64Name: Alias = L['float64', 'float', 'double']
_Float64Char: Alias = L['f8', '|f8', '=f8', '<f8', '>f8', 'd', '|d', '=d', '<d', '>d']
_Float64Code: Alias = L[_Float64Name, _Float64Char]
AnyFloat64DType: Alias = _Any2[np.float64, _ct.Float64] | _Float64Code | None

_LongDoubleName: Alias = L['longdouble']
_LongDoubleChar: Alias = L['g', '|g', '=g', '<g', '>g']
_LongDoubleCode: Alias = L[_LongDoubleName, _LongDoubleChar]
AnyLongDoubleDType: Alias = _Any2[np.longdouble, _ct.LongDouble] | _LongDoubleCode

_FloatingCode: Alias = L[_Float16Code, _Float32Code, _Float64Code, _LongDoubleCode]
AnyFloatingDType: Alias = _Any2[_sc.Floating, _ct.Floating] | _FloatingCode

# complex floating

_Complex64Name: Alias = L['complex64', 'csingle']
_Complex64Char: Alias = L['c8', '|c8', '=c8', '<c8', '>c8', 'F', '|F', '=F', '<F', '>F']
_Complex64Code: Alias = L[_Complex64Name, _Complex64Char]
AnyComplex64DType: Alias = _Any1[np.complex64] | _Complex64Code

_Complex128Name: Alias = L['complex128', 'clongdouble']
_Complex128Char: Alias = L[
    'c16', '|c16', '=c16', '<c16', '>c16',
    'D', '|D', '=D', '<D', '>D',
]
_Complex128Code: Alias = L[_Complex128Name, _Complex128Char]
AnyComplex128DType: Alias = _Any1[np.complex128] | _Complex128Code

_CLongDoubleName: Alias = L['clongdouble']
_CLongDoubleChar: Alias = L['G', '|G', '=G', '<G', '>G']
_CLongDoubleCode: Alias = L[_CLongDoubleName, _CLongDoubleChar]
AnyCLongDoubleDType: Alias = _Any1[np.clongdouble] | _CLongDoubleCode

_ComplexFloatingCode: Alias = L[_Complex64Code, _Complex128Code, _CLongDoubleCode]
AnyComplexFloatingDType: Alias = _Any1[_sc.ComplexFloating] | _ComplexFloatingCode

# temporal

_DateTime64Name: Alias = L[
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
_DateTime64Char: Alias = L[
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
_DateTime64Code: Alias = L[_DateTime64Name, _DateTime64Char]
AnyDateTime64DType: Alias = _Any1[np.datetime64] | _DateTime64Code

_TimeDelta64Name: Alias = L[
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
_TimeDelta64Char: Alias = L[
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
_TimeDelta64Code: Alias = L[_TimeDelta64Name, _TimeDelta64Char]
AnyTimeDelta64DType: Alias = _Any1[np.timedelta64] | _TimeDelta64Code

# flexible

_StrName: Alias = L['str_', 'str', 'unicode']
_StrChar: Alias = L[
    'U', '|U', '=U', '<U', '>U',
    'U0', '|U0', '=U0', '<U0', '>U0',
    'U1', '|U1', '=U1', '<U1', '>U1',
]  # fmt: skip
_StrCode: Alias = L[_StrName, _StrChar]
AnyStrDType: Alias = _Any2[np.str_, str] | _StrCode

_BytesName: Alias = L['bytes_', 'bytes']
_BytesChar: Alias = L[
    'S', '|S', '=S', '<S', '>S',
    'S0', '|S0', '=S0', '<S0', '>S0',
    'S1', '|S1', '=S1', '<S1', '>S1',
]  # fmt: skip
_BytesCode: Alias = L[_BytesName, _BytesChar]
AnyBytesDType: Alias = _Any2[np.bytes_, bytes | _ct.Bytes] | _BytesCode

_CharacterCode: Alias = L[_StrCode, _BytesCode]
AnyCharacterDType: Alias = _Any2[np.character, bytes | str | _ct.Bytes] | _CharacterCode

# TODO: Include structured DType values, e.g. `dtype(('u8', 4))`
_VoidName: Alias = L['void']  # 'void0' was removed in NumPy 2.0
_VoidChar: Alias = L[
    'V', '|V', '=V', '<V', '>V',
    'V0', '|V0', '=V0', '<V0', '>V0',
    'V1', '|V1', '=V1', '<V1', '>V1',
]
_VoidCode: Alias = L[_VoidName, _VoidChar]
AnyVoidDType: Alias = _Any2[np.void, memoryview | _ct.Void] | _VoidCode

# flexible
_FlexibleCode: Alias = L[_CharacterCode, _VoidCode]
AnyFlexibleDType: Alias = (
    _Any2[np.flexible, bytes | str | memoryview | _ct.Flexible]
    | _FlexibleCode
)

# bool_
_BoolName: Alias = L['bool', 'bool_']  # 'bool0' was removed in NumPy 2.0
_BoolChar: Alias = L['?', '|?', '=?', '<?', '>?']
_BoolCode: Alias = L[_BoolName, _BoolChar]
# It's ok to use `bool` here
AnyBoolDType: Alias = _Any2[_x.Bool, bool | _ct.Bool] | _BoolCode

# object
_ObjectName: Alias = L['object', 'object_']
_ObjectChar: Alias = L['O', '|O', '=O', '<O', '>O']
_ObjectCode: Alias = L[_ObjectName, _ObjectChar]
# NOTE: `type[object]` isn't included, since this could lead to many bugs
#   e.g. in `numpy<2.1` we have `dtype(type[str | float]) -> dtype[object_]`...
AnyObjectDType: Alias = _Any2[np.object_, _ct.Object] | _ObjectCode


_InexactCode: Alias = L[_FloatingCode, _ComplexFloatingCode]
AnyInexactDType: Alias = _Any2[_sc.Inexact, _ct.Floating] | _InexactCode

_SignedIntegerName: Alias = L[
    'byte', 'int8', 'short', 'int16', 'intc', 'int32', 'int64',
    'intp', 'int', 'int_', 'long', 'longlong',
]  # fmt: skip
_SIntCharCommon: Alias = L[
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
    L['p', '|p', '=p', '<p', '>p'],
]

_UnsignedIntegerName: Alias = L[
    'ubyte', 'uint8', 'ushort', 'uint16', 'uintc', 'uint32', 'uint64',
    'uintp', 'uint', 'ulong', 'ulonglong',
]  # fmt: skip
_UIntCharCommon: Alias = L[
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
    L['P', '|P', '=P', '<P', '>P'],
]

# NOTE: At the moment, `np.dtypes.StringDType.type: type[str]`, which is
# impossible (i.e. `dtype[str]` isn't valid, as `str` isn't a `np.generic`)
_StringName = Never
_StringChar: Alias = L['T', '|T', '=T', '<T', '>T']
_StringCode: Alias = _StringChar

if _x.NP2:
    _UIntPName: Alias = L[_UIntName, 'uintp']
    _UIntPChar: Alias = L['N', '|N', '=N', '<N', '>N']
    _UIntPCode: Alias = L[_UIntPName, _UIntPChar]
    AnyUIntPDType: Alias = _Any2[np.uintp, _ct.UIntP] | _UIntPCode

    _ULongName: Alias = L['ulong']
    _ULongCode: Alias = L[_ULongName, _ULongChar]
    AnyULongDType: Alias = _Any2[_x.ULong, _ct.ULong] | _ULongCode

    _IntPName: Alias = L['intp', _IntName]
    _IntPChar: Alias = L['n', '|n', '=n', '<n', '>n']
    _IntPCode: Alias = L[_IntPName, _IntPChar]
    AnyIntPDType: Alias = _Any2[np.intp, _ct.IntP] | _IntPCode

    _LongName: Alias = L['long']
    _LongCode: Alias = L[_LongName, _LongChar]
    AnyLongDType: Alias = _Any2[_x.Long, _ct.Long] | _LongCode

    _UnsignedIntegerChar: Alias = L[
        _UIntCharCommon,
        _UIntPChar,
        L['N', '|N', '=N', '<N', '>N'],
    ]
    _UnsignedIntegerCode: Alias = L[_UnsignedIntegerName, _UnsignedIntegerChar]
    AnyUnsignedIntegerDType: Alias = (
        _Any2[_sc.UnsignedInteger, _ct.UnsignedInteger]
        | _UnsignedIntegerCode
    )

    _SignedIntegerChar: Alias = L[
        _SIntCharCommon,
        _IntPChar,
        L['n', '|n', '=n', '<n', '>n'],
    ]
    _SignedIntegerCode: Alias = L[_SignedIntegerName, _SignedIntegerChar]
    AnySignedIntegerDType: Alias = (
        _Any2[_sc.SignedInteger, _ct.SignedInteger]
        | _SignedIntegerCode
    )

    _IntegerCode: Alias = L[_UnsignedIntegerCode, _SignedIntegerCode]
    AnyIntegerDType: Alias = _Any2[_sc.Integer, _ct.Integer] | _IntegerCode

    _NumberCode: Alias = L[_IntegerCode, _InexactCode]
    # NOTE: this doesn't include `int` or `float` or `complex`, since that
    # would autoamtically include `bool`.
    AnyNumberDType: Alias = _Any2[_sc.Number, _ct.Number] | _NumberCode

    # NOTE: `np.dtypes.StringDType` didn't exist in the stubs prior to 2.1 (so
    # I (@jorenham) added them, see https://github.com/numpy/numpy/pull/27008).
    if not _x.NP20:
        # `numpy>=2.1`
        _HasStringDType: Alias = _dt.HasDType[np.dtypes.StringDType]  # type: ignore[type-var] # pyright: ignore[reportInvalidTypeArguments]

        AnyStringDType: Alias = _HasStringDType | _StringCode  # type: ignore[type-var]

        AnyDType: Alias = _Any2[np.generic, object] | _HasStringDType | LiteralString
    else:
        AnyStringDType: Alias = np.dtype[Never] | _StringCode

        AnyDType: Alias = _Any2[np.generic, object] | LiteralString
else:
    _UIntPName: Alias = L['uintp']  # 'uint0' is removed in NumPy 2.0
    _UIntPChar: Alias = L['P', '|P', '=P', '<P', '>P']
    _UIntPCode: Alias = L[_UIntPName, _UIntPChar]
    # assuming that `c_void_p == c_size_t`
    AnyUIntPDType: Alias = _Any2[np.uintp, _ct.UIntP] | _UIntPCode

    _ULongName: Alias = L[_UIntName, 'ulong']
    _ULongCode: Alias = L[_ULongName, _ULongChar]
    AnyULongDType: Alias = _Any2[_x.ULong, _ct.ULong] | _ULongCode

    _IntPName: Alias = L['intp']  # 'int0' is removed in NumPy 2.0
    _IntPChar: Alias = L['p', '|p', '=p', '<p', '>p']
    _IntPCode: Alias = L[_IntPName, _IntPChar]
    AnyIntPDType: Alias = _Any2[np.intp, _ct.IntP] | _IntPCode

    _LongName: Alias = L['long', _IntName]
    _LongCode: Alias = L[_LongName, _LongChar]
    AnyLongDType: Alias = _Any2[_x.Long, _ct.Long] | _LongCode

    _UnsignedIntegerChar: Alias = L[_UIntCharCommon, _UIntPChar]
    _UnsignedIntegerCode: Alias = L[_UnsignedIntegerName, _UnsignedIntegerChar]
    AnyUnsignedIntegerDType: Alias = (
        _Any2[_sc.UnsignedInteger, _ct.UnsignedInteger]
        | _UnsignedIntegerCode
    )

    _SignedIntegerChar: Alias = L[_SIntCharCommon, _IntPChar]
    _SignedIntegerCode: Alias = L[_SignedIntegerName, _SignedIntegerChar]
    AnySignedIntegerDType: Alias = (
        _Any2[_sc.SignedInteger, _ct.SignedInteger]
        | _SignedIntegerCode
    )

    _IntegerCode: Alias = L[_UnsignedIntegerCode, _SignedIntegerCode]
    AnyIntegerDType: Alias = _Any2[_sc.Integer, _ct.Integer] | _IntegerCode

    _NumberCode: Alias = L[_IntegerCode, _InexactCode]
    AnyNumberDType: Alias = _Any2[_sc.Number, _ct.Number] | _NumberCode

    AnyStringDType: Alias = Never

    AnyDType: Alias = _Any2[np.generic, object] | LiteralString
