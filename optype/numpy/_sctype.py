from __future__ import annotations

import ctypes as ct
import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias

import numpy as np

from ._dtype import ArgDType as _SoloType


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar


if TYPE_CHECKING:
    import numpy.typing as npt


_NP_V2: Final[bool] = np.__version__.startswith('2.')


_T_np = TypeVar('_T_np', bound=np.generic)
_T_py = TypeVar('_T_py', bound=object)
_DualType: TypeAlias = _SoloType[_T_np] | type[_T_py]


#
# booleans
#

if _NP_V2:
    _AnyBoolNP: TypeAlias = np.bool
    _AnyBoolName: TypeAlias = Literal['bool', 'bool_']
else:
    _AnyBoolNP: TypeAlias = np.bool_
    _AnyBoolName: TypeAlias = Literal['bool', 'bool_', 'bool8']
_AnyBoolChar: TypeAlias = Literal['?', '=?', '<?', '>?']
_AnyBoolCode: TypeAlias = _AnyBoolName | _AnyBoolChar
AnyBool: TypeAlias = _AnyBoolNP | ct.c_bool | bool
AnyBoolType: TypeAlias = _AnyBoolCode | _DualType[np.bool_, AnyBool]

#
# unsigned integers
#

# uint8
_AnyUInt8Name: TypeAlias = Literal['uint8']
_AnyUInt8Char: TypeAlias = Literal['u1', '=u1', '<u1', '>u1']
_AnyUInt8Code: TypeAlias = _AnyUInt8Name | _AnyUInt8Char
AnyUInt8: TypeAlias = np.uint8 | ct.c_uint8
AnyUInt8Type: TypeAlias = _AnyUInt8Code | _DualType[np.uint8, AnyUInt8]

# uint16
_AnyUInt16Name: TypeAlias = Literal['uint16']
_AnyUInt16Char: TypeAlias = Literal['u2', '=u2', '<u2', '>u2']
_AnyUInt16Code: TypeAlias = _AnyUInt16Name | _AnyUInt16Char
AnyUInt16: TypeAlias = np.uint16 | ct.c_uint16
AnyUInt16Type: TypeAlias = _AnyUInt16Code | _DualType[np.uint16, AnyUInt16]

# uint32
_AnyUInt32Name: TypeAlias = Literal['uint32']
_AnyUInt32Char: TypeAlias = Literal['u4', '=u4', '<u4', '>u4']
_AnyUInt32Code: TypeAlias = _AnyUInt32Name | _AnyUInt32Char
AnyUInt32: TypeAlias = np.uint32 | ct.c_uint32
AnyUInt32Type: TypeAlias = _AnyUInt32Code | _DualType[np.uint32, AnyUInt32]

# uint64
_AnyUInt64Name: TypeAlias = Literal['uint64']
_AnyUInt64Char: TypeAlias = Literal['u8', '=u8', '<u8', '>u8']
_AnyUInt64Code: TypeAlias = _AnyUInt64Name | _AnyUInt64Char
AnyUInt64: TypeAlias = np.uint64 | ct.c_uint64
AnyUInt64Type: TypeAlias = _AnyUInt64Code | _DualType[np.uint64, AnyUInt64]

# ubyte
_AnyUByteName: TypeAlias = Literal['ubyte']
_AnyUByteChar: TypeAlias = Literal['B', '=B', '<B', '>B']
_AnyUByteCode: TypeAlias = _AnyUByteName | _AnyUByteChar
AnyUByte: TypeAlias = np.ubyte | ct.c_ubyte
AnyUByteType: TypeAlias = _AnyUByteCode | _DualType[np.ubyte, AnyUByte]

# ushort
_AnyUShortName: TypeAlias = Literal['ushort']
_AnyUShortChar: TypeAlias = Literal['H', '=H', '<H', '>H']
_AnyUShortCode: TypeAlias = _AnyUShortName | _AnyUShortChar
AnyUShort: TypeAlias = np.ushort | ct.c_ushort
AnyUShortType: TypeAlias = _AnyUShortCode | _DualType[np.ushort, AnyUShort]

# uintc
_AnyUIntCName: TypeAlias = Literal['uintc']
_AnyUIntCChar: TypeAlias = Literal['I', '=I', '<I', '>I']
_AnyUIntCCode: TypeAlias = _AnyUIntCName | _AnyUIntCChar
AnyUIntC: TypeAlias = np.uintc | ct.c_uint
AnyUIntCType: TypeAlias = _AnyUIntCCode | _DualType[np.uintc, AnyUIntC]

# uintp (assuming `uint_ptr_t == size_t` like in `numpy.typing`)
if _NP_V2:
    _AnyUIntPName: TypeAlias = Literal['uintp', 'uint']
    _AnyUIntPChar: TypeAlias = Literal['N', '=N', '<N', '>N']
else:
    _AnyUIntPName: TypeAlias = Literal['uintp', 'uint0']
    _AnyUIntPChar: TypeAlias = Literal['P', '=P', '<P', '>P']
_AnyUIntPCode: TypeAlias = _AnyUIntPName | _AnyUIntPChar
AnyUIntP: TypeAlias = np.uintp | ct.c_void_p | ct.c_size_t
AnyUIntPType: TypeAlias = _AnyUIntPCode | _DualType[np.uintp, AnyUIntP]

# ulong (uint on numpy^=1)
if _NP_V2:
    _AnyULongNP: TypeAlias = np.ulong
    _AnyULongName: TypeAlias = Literal['ulong']
else:
    _AnyULongNP: TypeAlias = np.uint
    _AnyULongName: TypeAlias = Literal['ulong', 'uint']
_AnyULongChar: TypeAlias = Literal['L', '=L', '<L', '>L']
_AnyULongCode: TypeAlias = _AnyULongName | _AnyULongChar
AnyULong: TypeAlias = _AnyULongNP | ct.c_ulong
AnyULongType: TypeAlias = _AnyULongCode | _DualType[_AnyULongNP, AnyULong]

# ulonglong
_AnyULongLongName: TypeAlias = Literal['ulonglong']
_AnyULongLongChar: TypeAlias = Literal['Q', '=Q', '<Q', '>Q']
_AnyULongLongCode: TypeAlias = _AnyULongLongName | _AnyULongLongChar
AnyULongLong: TypeAlias = np.ulonglong | ct.c_ulonglong
AnyULongLongType: TypeAlias = (
    _AnyULongLongCode
    | _DualType[np.ulonglong, AnyULongLong]
)

# unsignedinteger
_N_unsignedinteger = TypeVar(
    '_N_unsignedinteger',
    bound='npt.NBitBase',
    default=Any,
)
_AnyUnsignedIntegerNP: TypeAlias = np.unsignedinteger[_N_unsignedinteger]
# fmt: off
_AnyUnsignedIntegerCT: TypeAlias = (
    ct.c_uint8 | ct.c_uint16 | ct.c_uint32 | ct.c_uint64
    | ct.c_ubyte | ct.c_ushort | ct.c_uint | ct.c_ulong | ct.c_ulonglong
    | ct.c_size_t | ct.c_void_p
)
# fmt: on
# `builtins.int` is signed
_AnyUnsignedIntegerCode: TypeAlias = (
    _AnyUInt8Code | _AnyUInt16Code | _AnyUInt32Code | _AnyUInt64Code
    | _AnyUByteCode | _AnyUShortCode | _AnyULongCode | _AnyULongLongCode
    | _AnyUIntCCode | _AnyUIntPCode
)
AnyUnsignedInteger: TypeAlias = (
    _AnyUnsignedIntegerNP[_N_unsignedinteger]
    | _AnyUnsignedIntegerCT
)
AnyUnsignedIntegerType: TypeAlias = (
    _AnyUnsignedIntegerCode
    | _DualType[
        _AnyUnsignedIntegerNP[_N_unsignedinteger],
        AnyUnsignedInteger[_N_unsignedinteger],
    ]
)


#
# signed integers
#

# int8
_AnyInt8Name: TypeAlias = Literal['int8']
_AnyInt8Char: TypeAlias = Literal['i1', '=i1', '<i1', '>i1']
_AnyInt8Code: TypeAlias = _AnyInt8Name | _AnyInt8Char
AnyInt8: TypeAlias = np.int8 | ct.c_int8
AnyInt8Type: TypeAlias = _AnyInt8Code | _DualType[np.int8, AnyInt8]

# int16
_AnyInt16Name: TypeAlias = Literal['int16']
_AnyInt16Char: TypeAlias = Literal['i2', '=i2', '<i2', '>i2']
_AnyInt16Code: TypeAlias = _AnyInt16Name | _AnyInt16Char
AnyInt16: TypeAlias = np.int16 | ct.c_int16
AnyInt16Type: TypeAlias = _AnyInt16Code | _DualType[np.int16, AnyInt16]

# int32
_AnyInt32Name: TypeAlias = Literal['int32']
_AnyInt32Char: TypeAlias = Literal['i4', '=i4', '<i4', '>i4']
_AnyInt32Code: TypeAlias = _AnyInt32Name | _AnyInt32Char
AnyInt32: TypeAlias = np.int32 | ct.c_int32
AnyInt32Type: TypeAlias = _AnyInt32Code | _DualType[np.int32, AnyInt32]

# int64
_AnyInt64Name: TypeAlias = Literal['int64']
_AnyInt64Char: TypeAlias = Literal['i8', '=i8', '<i8', '>i8']
_AnyInt64Code: TypeAlias = _AnyInt64Name | _AnyInt64Char
AnyInt64: TypeAlias = np.int64 | ct.c_int64
AnyInt64Type: TypeAlias = _AnyInt64Code | _DualType[np.int64, AnyInt64]

# byte
_AnyByteName: TypeAlias = Literal['byte']
_AnyByteChar: TypeAlias = Literal['b', '=b', '<b', '>b']
_AnyByteCode: TypeAlias = _AnyByteName | _AnyByteChar
AnyByte: TypeAlias = np.byte | ct.c_byte
AnyByteType: TypeAlias = _AnyByteCode | _DualType[np.byte, AnyByte]

# short
_AnyShortName: TypeAlias = Literal['short']
_AnyShortChar: TypeAlias = Literal['h', '=h', '<h', '>h']
_AnyShortCode: TypeAlias = _AnyShortName | _AnyShortChar
AnyShort: TypeAlias = np.short | ct.c_short
AnyShortType: TypeAlias = _AnyShortCode | _DualType[np.short, AnyShort]

# intc
_AnyIntCName: TypeAlias = Literal['intc']
_AnyIntCChar: TypeAlias = Literal['i', '=i', '<i', '>i']
_AnyIntCCode: TypeAlias = _AnyIntCName | _AnyIntCChar
AnyIntC: TypeAlias = np.intc | ct.c_int
AnyIntCType: TypeAlias = _AnyIntCCode | _DualType[np.intc, AnyIntC]

# intp
if _NP_V2:
    _AnyIntPName: TypeAlias = Literal['intp', 'int', 'int_']
    _AnyIntPChar: TypeAlias = Literal['n', '=n', '<n', '>n']
else:
    _AnyIntPName: TypeAlias = Literal['intp', 'int0']
    _AnyIntPChar: TypeAlias = Literal['p', '=p', '<p', '>p']
_AnyIntPCode: TypeAlias = _AnyIntPName | _AnyIntPChar
AnyIntP: TypeAlias = np.intp | ct.c_ssize_t
AnyIntPType: TypeAlias = _AnyIntPCode | _DualType[np.intp, AnyIntP]

# long (int_ on numpy^=1)
_AnyLongChar: TypeAlias = Literal['l', '=l', '<l', '>l']
if _NP_V2:
    _AnyLongNP: TypeAlias = np.long
    _AnyLongName: TypeAlias = Literal['long']
    AnyLong: TypeAlias = _AnyLongNP | ct.c_long
else:
    _AnyLongNP: TypeAlias = np.int_
    _AnyLongName: TypeAlias = Literal['long', 'int', 'int_']
    AnyLong: TypeAlias = _AnyLongNP | ct.c_long | int
_AnyLongCode: TypeAlias = _AnyLongName | _AnyLongChar
AnyLongType: TypeAlias = _AnyLongCode | _DualType[_AnyLongNP, AnyLong]

# longlong
_AnyLongLongName: TypeAlias = Literal['longlong']
_AnyLongLongChar: TypeAlias = Literal['q', '=q', '<q', '>q']
_AnyLongLongCode: TypeAlias = _AnyLongLongName | _AnyLongLongChar
AnyLongLong: TypeAlias = np.longlong | ct.c_longlong
AnyLongLongType: TypeAlias = (
    _AnyLongLongCode
    | _DualType[np.longlong, AnyLongLong]
)

# signedinteger
_N_signedinteger = TypeVar(
    '_N_signedinteger',
    bound='npt.NBitBase',
    default=Any,
)
_AnySignedIntegerNP = np.signedinteger[_N_signedinteger]
# fmt: off
_AnySignedIntegerCT: TypeAlias = (
    ct.c_int8 | ct.c_int16 | ct.c_int32 | ct.c_int64
    | ct.c_byte | ct.c_short | ct.c_int | ct.c_long | ct.c_longlong
    | ct.c_ssize_t
)
# fmt: on
_AnySignedIntegerPY: TypeAlias = int
# fmt: off
_AnySignedIntegerCode: TypeAlias = (
    _AnyInt8Code | _AnyInt16Code | _AnyInt32Code | _AnyInt64Code
    | _AnyByteCode | _AnyShortCode | _AnyLongCode | _AnyLongLongCode
    | _AnyIntCCode | _AnyIntPCode
)
# fmt: on
AnySignedInteger: TypeAlias = (
    _AnySignedIntegerNP[_N_signedinteger]
    | _AnySignedIntegerCT
    | _AnySignedIntegerPY
)
AnySignedIntegerType: TypeAlias = (
    _AnySignedIntegerCode
    | _DualType[
        _AnySignedIntegerNP[_N_signedinteger],
        AnySignedInteger[_N_signedinteger],
    ]
)


#
# (unsigned | signed) integers
#

# integer
_N_integer = TypeVar('_N_integer', bound='npt.NBitBase', default=Any)
_AnyIntegerNP: TypeAlias = np.integer[_N_integer]
_AnyIntegerCT: TypeAlias = _AnyUnsignedIntegerCT | _AnySignedIntegerCT
_AnyIntegerPY: TypeAlias = _AnySignedIntegerPY
_AnyIntegerCode: TypeAlias = _AnyUnsignedIntegerCode | _AnySignedIntegerCode
AnyInteger: TypeAlias = (
    _AnyIntegerNP[_N_integer]
    | _AnyIntegerCT
    | _AnyIntegerPY
)
AnyIntegerType: TypeAlias = (
    _AnyIntegerCode
    | _DualType[_AnyIntegerNP[_N_integer], AnyInteger[_N_integer]]
)


#
# real floats
#

# float16
_AnyFloat16Name: TypeAlias = Literal['float16']
_AnyFloat16Char: TypeAlias = Literal['f2', '=f2', '<f2', '>f2']
_AnyFloat16Code: TypeAlias = _AnyFloat16Name | _AnyFloat16Char
AnyFloat16: TypeAlias = np.float16
AnyFloat16Type: TypeAlias = _AnyFloat16Code | _SoloType[np.float16]

# float32
_AnyFloat32Name: TypeAlias = Literal['float32']
_AnyFloat32Char: TypeAlias = Literal['f4', '=f4', '<f4', '>f4']
_AnyFloat32Code: TypeAlias = _AnyFloat32Name | _AnyFloat32Char
AnyFloat32: TypeAlias = np.float32
AnyFloat32Type: TypeAlias = _AnyFloat32Code | _SoloType[np.float32]

# float64
_AnyFloat64NP: TypeAlias = np.float64
_AnyFloat64Name: TypeAlias = Literal['float64']
_AnyFloat64Char: TypeAlias = Literal['f8', '=f8', '<f8', '>f8']
_AnyFloat64Code: TypeAlias = _AnyFloat64Name | _AnyFloat64Char
AnyFloat64: TypeAlias = _AnyFloat64NP
AnyFloat64Type: TypeAlias = _AnyFloat64Code | _SoloType[_AnyFloat64NP]

# half
_AnyHalfName: TypeAlias = Literal['half']
_AnyHalfChar: TypeAlias = Literal['e', '=e', '<e', '>e']
_AnyHalfCode: TypeAlias = _AnyHalfName | _AnyHalfChar
AnyHalf: TypeAlias = np.half
AnyHalfType: TypeAlias = _AnyHalfCode | _SoloType[np.half]

# single
_AnySingleName: TypeAlias = Literal['single']
_AnySingleChar: TypeAlias = Literal['f', '=f', '<f', '>f']
_AnySingleCode: TypeAlias = _AnySingleName | _AnySingleChar
AnySingle: TypeAlias = np.single | ct.c_float
AnySingleType: TypeAlias = _AnySingleCode | _DualType[np.single, AnySingle]

# double
_AnyDoubleNP: TypeAlias = np.double
if _NP_V2:
    _AnyDoubleName: TypeAlias = Literal['double', 'float']
else:
    _AnyDoubleName: TypeAlias = Literal['double', 'float', 'float_']
_AnyDoubleChar: TypeAlias = Literal['d', '=d', '<d', '>d']
_AnyDoubleCode: TypeAlias = _AnyDoubleName | _AnyDoubleChar
AnyDouble: TypeAlias = _AnyDoubleNP | ct.c_double
AnyDoubleType: TypeAlias = _AnyDoubleCode | _DualType[_AnyDoubleNP, AnyDouble]

# longdouble
if _NP_V2:
    _AnyLongDoubleName: TypeAlias = Literal['longdouble']
else:
    _AnyLongDoubleName: TypeAlias = Literal['longdouble', 'longfloat']
_AnyLongDoubleChar: TypeAlias = Literal['g', '=g', '<g', '>g']
_AnyLongDoubleCode: TypeAlias = _AnyLongDoubleName | _AnyLongDoubleChar
AnyLongDouble: TypeAlias = np.longdouble | ct.c_longdouble
AnyLongDoubleType: TypeAlias = (
    _AnyLongDoubleCode
    | _DualType[np.longdouble, AnyLongDouble]
)

# floating
_N_floating = TypeVar('_N_floating', bound='npt.NBitBase', default=Any)
_AnyFloatingNP: TypeAlias = np.floating[_N_floating]
_AnyFloatingCT: TypeAlias = ct.c_float | ct.c_double | ct.c_longdouble
_AnyFloatingPY: TypeAlias = float
# fmt: off
_AnyFloatingCode: TypeAlias = (
    _AnyFloat16Code | _AnyFloat32Code | _AnyFloat64Code
    | _AnyHalfCode | _AnySingleCode | _AnyDoubleCode | _AnyLongDoubleCode
)
# fmt: on
AnyFloating: TypeAlias = (
    _AnyFloatingNP[_N_floating]
    | _AnyFloatingCT
    | _AnyFloatingPY
)
AnyFloatingType: TypeAlias = (
    _AnyFloatingCode
    | _DualType[_AnyFloatingNP[_N_floating], AnyFloating[_N_floating]]
)


#
# complex floats
#

# complex64
_AnyComplex64Name: TypeAlias = Literal['complex64']
_AnyComplex64Char: TypeAlias = Literal['c8', '=c8', '<c8', '>c8']
_AnyComplex64Code: TypeAlias = _AnyComplex64Name | _AnyComplex64Char
AnyComplex64: TypeAlias = np.complex64
AnyComplex64Type: TypeAlias = _AnyComplex64Code | _SoloType[np.complex64]

# complex128
_AnyComplex128NP: TypeAlias = np.complex128
_AnyComplex128Name: TypeAlias = Literal['complex128']
_AnyComplex128Char: TypeAlias = Literal['c16', '=c16', '<c16', '>c16']
_AnyComplex128Code: TypeAlias = _AnyComplex128Name | _AnyComplex128Char
AnyComplex128: TypeAlias = _AnyComplex128NP
AnyComplex128Type: TypeAlias = _AnyComplex128Code | _SoloType[_AnyComplex128NP]

# csingle
if _NP_V2:
    _AnyCSingleName: TypeAlias = Literal['csingle']
else:
    _AnyCSingleName: TypeAlias = Literal['csingle', 'singlecomplex']
_AnyCSingleChar: TypeAlias = Literal['F', '=F', '<F', '>F']
_AnyCSingleCode: TypeAlias = _AnyCSingleName | _AnyCSingleChar
AnyCSingle: TypeAlias = np.csingle
AnyCSingleType: TypeAlias = _AnyCSingleCode | _SoloType[np.csingle]

# cdouble
_AnyCDoubleNP: TypeAlias = np.cdouble
if _NP_V2:
    _AnyCDoubleName: TypeAlias = Literal['cdouble', 'complex']
else:
    # fmt: off
    _AnyCDoubleName: TypeAlias = Literal[
        'cdouble', 'cfloat',
        'complex', 'complex_',
    ]
    # fmt: on
_AnyCDoubleChar: TypeAlias = Literal['D', '=D', '<D', '>D']
_AnyCDoubleCode: TypeAlias = _AnyCDoubleName | _AnyCDoubleChar
AnyCDouble: TypeAlias = _AnyCDoubleNP | complex
AnyCDoubleType: TypeAlias = (
    _AnyCDoubleCode
    | _DualType[_AnyCDoubleNP, AnyCDouble]
)

# clongdouble
if _NP_V2:
    _AnyCLongDoubleName: TypeAlias = Literal['clongdouble']
else:
    _AnyCLongDoubleName: TypeAlias = Literal[
        'clongdouble',
        'clongfloat',
        'longcomplex',
    ]
_AnyCLongDoubleChar: TypeAlias = Literal['G', '=G', '<G', '>G']
_AnyCLongDoubleCode: TypeAlias = _AnyCLongDoubleName | _AnyCLongDoubleChar
AnyCLongDouble: TypeAlias = np.clongdouble
AnyCLongDoubleType: TypeAlias = _AnyCLongDoubleCode | _SoloType[np.clongdouble]

# complexfloating
_N_complexfloating_re = TypeVar(
    '_N_complexfloating_re',
    bound='npt.NBitBase',
    default=Any,
)
_N_complexfloating_im = TypeVar(
    '_N_complexfloating_im',
    bound='npt.NBitBase',
    default=_N_complexfloating_re,
)
_AnyComplexFloatingNP: TypeAlias = np.complexfloating[
    _N_complexfloating_re,
    _N_complexfloating_im,
]
_AnyComplexFloatingPY: TypeAlias = complex
# fmt: off
_AnyComplexFloatingCode: TypeAlias = (
    _AnyComplex64Code | _AnyComplex128Code
    | _AnyCSingleCode | _AnyCDoubleCode | _AnyCLongDoubleCode
)
# fmt: on
AnyComplexFloating: TypeAlias = (
    _AnyComplexFloatingNP[_N_complexfloating_re, _N_complexfloating_im]
    | _AnyComplexFloatingPY
)
AnyComplexFloatingType: TypeAlias = (
    _AnyComplexFloatingCode
    | _DualType[
        _AnyComplexFloatingNP[_N_complexfloating_re, _N_complexfloating_im],
        AnyComplexFloating[_N_complexfloating_re, _N_complexfloating_im],
    ]
)


#
# (real | complex) floats
#

# inexact
_N_inexact = TypeVar('_N_inexact', bound='npt.NBitBase', default=Any)
_AnyInexactNP: TypeAlias = np.inexact[_N_inexact]
_AnyInexactCT: TypeAlias = _AnyFloatingCT
_AnyInexactPY: TypeAlias = _AnyFloatingPY | _AnyComplexFloatingPY
_AnyInexactCode: TypeAlias = _AnyFloatingCode | _AnyComplexFloatingCode
AnyInexact: TypeAlias = (
    _AnyInexactNP[_N_inexact]
    | _AnyInexactCT
    | _AnyInexactPY
)
AnyInexactType: TypeAlias = (
    _AnyInexactCode
    | _DualType[_AnyInexactNP[_N_inexact], AnyInexact[_N_inexact]]
)


#
# integers | floats
#

# number
_N_number = TypeVar('_N_number', bound='npt.NBitBase', default=Any)
_AnyNumberNP: TypeAlias = np.number[_N_number]
_AnyNumberCT: TypeAlias = _AnyIntegerCT | _AnyInexactCT
_AnyNumberPY: TypeAlias = _AnyIntegerPY | _AnyInexactPY
_AnyNumberCode: TypeAlias = _AnyIntegerCode | _AnyInexactCode
AnyNumber: TypeAlias = _AnyNumberNP[_N_number] | _AnyNumberCT | _AnyNumberPY
AnyNumberType: TypeAlias = (
    _AnyNumberCode
    | _DualType[_AnyNumberNP[_N_number], AnyNumber[_N_number]]
)


#
# temporal
#

# timedelta64
_AnyTimedelta64Name: TypeAlias = Literal['timedelta64']
# fmt: off
_AnyTimedelta64Char: TypeAlias = Literal[
    'm', '=m', '<m', '>m',
    'm8', '=m8', '<m8', '>m8',
]
# fmt: on
_AnyTimedelta64Code: TypeAlias = _AnyTimedelta64Name | _AnyTimedelta64Char
AnyTimedelta64: TypeAlias = np.timedelta64
AnyTimedelta64Type: TypeAlias = _AnyTimedelta64Code | _SoloType[np.timedelta64]

# datetime64
_AnyDatetime64Name: TypeAlias = Literal['datetime64']
# fmt: off
_AnyDatetime64Char: TypeAlias = Literal[
    'M', '=M', '<M', '>M',
    'M8', '=M8', '<M8', '>M8',
]
# fmt: on
_AnyDatetime64Code: TypeAlias = _AnyDatetime64Name | _AnyDatetime64Char
AnyDatetime64: TypeAlias = np.datetime64
AnyDatetime64Type: TypeAlias = _AnyDatetime64Code | _SoloType[np.datetime64]


#
# character strings
#

# str
if _NP_V2:
    _AnyStrName: TypeAlias = Literal['str', 'str_', 'unicode']
else:
    # fmt: off
    _AnyStrName: TypeAlias = Literal[
        'str', 'str_', 'str0',
        'unicode', 'unicode_',
    ]
    # fmt: on
_AnyStrChar: TypeAlias = Literal['U', '=U', '<U', '>U']
_AnyStrCode: TypeAlias = _AnyStrName | _AnyStrChar
AnyStr: TypeAlias = np.str_ | str
AnyStrType: TypeAlias = _AnyStrCode | _DualType[np.str_, AnyStr]

# bytes
if _NP_V2:
    _AnyBytesName: TypeAlias = Literal['bytes', 'bytes_']
else:
    _AnyBytesName: TypeAlias = Literal['bytes', 'bytes_', 'bytes0']
_AnyBytesChar: TypeAlias = Literal['S', '=S', '<S', '>S']
_AnyBytesCode: TypeAlias = _AnyBytesName | _AnyBytesChar
AnyBytes: TypeAlias = np.bytes_ | bytes
AnyBytesType: TypeAlias = _AnyBytesCode | _DualType[np.bytes_, AnyBytes]

# character
_AnyCharacterNP: TypeAlias = np.character
_AnyCharacterCT: TypeAlias = ct.c_char
_AnyCharacterPY: TypeAlias = str | bytes
_AnyCharacterCode: TypeAlias = _AnyStrCode | _AnyBytesCode
AnyCharacter: TypeAlias = _AnyCharacterNP | _AnyCharacterCT | _AnyCharacterPY
AnyCharacterType: TypeAlias = (
    _AnyCharacterCode
    | _DualType[_AnyCharacterNP, AnyCharacter]
)

# void
if _NP_V2:
    _AnyVoidName: TypeAlias = Literal['void']
else:
    _AnyVoidName: TypeAlias = Literal['void', 'void0']
_AnyVoidChar: TypeAlias = Literal['V', '=V', '<V', '>V']
_AnyVoidCode: TypeAlias = _AnyVoidName | _AnyVoidChar
AnyVoid: TypeAlias = np.void
AnyVoidType: TypeAlias = _AnyVoidCode | _SoloType[np.void]

# flexible
_AnyFlexibleNP: TypeAlias = np.flexible
_AnyFlexibleCT: TypeAlias = _AnyCharacterCT
_AnyFlexiblePY: TypeAlias = _AnyCharacterPY
_AnyFlexibleCode: TypeAlias = _AnyCharacterCode | _AnyVoidCode
AnyFlexible: TypeAlias = _AnyFlexibleNP | _AnyFlexibleCT | _AnyFlexiblePY
AnyFlexibleType: TypeAlias = (
    _AnyFlexibleCode
    | _DualType[_AnyFlexibleNP, AnyFlexible]
)


#
# python objects
#

# object
_AnyObjectNP: TypeAlias = np.object_
if TYPE_CHECKING:
    _AnyObjectCT: TypeAlias = ct.py_object[Any]
else:
    _AnyObjectCT: TypeAlias = ct.py_object
_AnyObjectName: TypeAlias = Literal['object', 'object_']
_AnyObjectChar: TypeAlias = Literal['O', '=O', '<O', '>O']
_AnyObjectCode: TypeAlias = _AnyObjectName | _AnyObjectChar
AnyObject: TypeAlias = _AnyObjectNP | _AnyObjectCT | object
AnyObjectType: TypeAlias = _AnyObjectCode | _DualType[_AnyObjectNP, AnyObject]


#
# any scalar
#

# generic
_N_generic = TypeVar('_N_generic', bound='npt.NBitBase', default=Any)
_AnyGenericNP: TypeAlias = (
    _AnyBoolNP
    | _AnyObjectNP
    | _AnyNumberNP[_N_generic]
    | _AnyFlexibleNP
    | np.generic  # catch-all for any other user-defined scalar types
)
_AnyGenericCode: TypeAlias = (
    _AnyBoolCode
    | _AnyNumberCode
    | _AnyFlexibleCode
    | _AnyObjectCode
)
AnyGeneric: TypeAlias = (
    AnyBool
    | AnyNumber[_N_generic]
    | AnyFlexible
    | AnyObject
    | np.generic  # catch-all for any other user-defined scalar types
)
AnyGenericType: TypeAlias = (
    _AnyGenericCode
    | _DualType[_AnyGenericNP[_N_generic], AnyGeneric[_N_generic]]
)
