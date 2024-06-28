from __future__ import annotations

import ctypes as ct
import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias

import numpy as np

from ._dtype import AnyDType as _SoloType


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar


if TYPE_CHECKING:
    import numpy.typing as npt


_NP_V2: Final[bool] = np.__version__.startswith('2.')


_ST__DualType = TypeVar('_ST__DualType', bound=np.generic)
_PT__DualType = TypeVar('_PT__DualType', bound=object)
_DualType: TypeAlias = _SoloType[_ST__DualType] | type[_PT__DualType]


#
# unsigned integers
#

# uint8
_AnyUInt8NP: TypeAlias = np.uint8
_AnyUInt8Name: TypeAlias = Literal['uint8']
_AnyUInt8Char: TypeAlias = Literal['u1', '=u1', '<u1', '>u1']
_AnyUInt8Code: TypeAlias = _AnyUInt8Name | _AnyUInt8Char
AnyUInt8Value: TypeAlias = _AnyUInt8NP | ct.c_uint8
AnyUInt8Type: TypeAlias = _AnyUInt8Code | _DualType[_AnyUInt8NP, AnyUInt8Value]

# uint16
_AnyUInt16NP: TypeAlias = np.uint16
_AnyUInt16Name: TypeAlias = Literal['uint16']
_AnyUInt16Char: TypeAlias = Literal['u2', '=u2', '<u2', '>u2']
_AnyUInt16Code: TypeAlias = _AnyUInt16Name | _AnyUInt16Char
AnyUInt16Value: TypeAlias = _AnyUInt16NP | ct.c_uint16
AnyUInt16Type: TypeAlias = (
    _AnyUInt16Code
    | _DualType[_AnyUInt16NP, AnyUInt16Value]
)

# uint32
_AnyUInt32NP: TypeAlias = np.uint32
_AnyUInt32Name: TypeAlias = Literal['uint32']
_AnyUInt32Char: TypeAlias = Literal['u4', '=u4', '<u4', '>u4']
_AnyUInt32Code: TypeAlias = _AnyUInt32Name | _AnyUInt32Char
AnyUInt32Value: TypeAlias = _AnyUInt32NP | ct.c_uint32
AnyUInt32Type: TypeAlias = (
    _AnyUInt32Code
    | _DualType[_AnyUInt32NP, AnyUInt32Value]
)

# uint64
_AnyUInt64: TypeAlias = np.uint64
_AnyUInt64Name: TypeAlias = Literal['uint64']
_AnyUInt64Char: TypeAlias = Literal['u8', '=u8', '<u8', '>u8']
_AnyUInt64Code: TypeAlias = _AnyUInt64Name | _AnyUInt64Char
AnyUInt64Value: TypeAlias = _AnyUInt64 | ct.c_uint64
AnyUInt64Type: TypeAlias = (
    _AnyUInt64Code
    | _DualType[_AnyUInt64, AnyUInt64Value]
)

# ubyte
_AnyUByteNP: TypeAlias = np.ubyte
_AnyUByteName: TypeAlias = Literal['ubyte']
_AnyUByteChar: TypeAlias = Literal['B', '=B', '<B', '>B']
_AnyUByteCode: TypeAlias = _AnyUByteName | _AnyUByteChar
AnyUByteValue: TypeAlias = _AnyUByteNP | ct.c_ubyte
AnyUByteType: TypeAlias = _AnyUByteCode | _DualType[_AnyUByteNP, AnyUByteValue]

# ushort
_AnyUShortNP: TypeAlias = np.ushort
_AnyUShortName: TypeAlias = Literal['ushort']
_AnyUShortChar: TypeAlias = Literal['H', '=H', '<H', '>H']
_AnyUShortCode: TypeAlias = _AnyUShortName | _AnyUShortChar
AnyUShortValue: TypeAlias = _AnyUShortNP | ct.c_ushort
AnyUShortType: TypeAlias = (
    _AnyUShortCode
    | _DualType[_AnyUShortNP, AnyUShortValue]
)

# uintc
_AnyUIntCNP: TypeAlias = np.uintc
_AnyUIntCName: TypeAlias = Literal['uintc']
_AnyUIntCChar: TypeAlias = Literal['I', '=I', '<I', '>I']
_AnyUIntCCode: TypeAlias = _AnyUIntCName | _AnyUIntCChar
AnyUIntCValue: TypeAlias = _AnyUIntCNP | ct.c_uint
AnyUIntCType: TypeAlias = _AnyUIntCCode | _DualType[_AnyUIntCNP, AnyUIntCValue]

# uintp (assuming `uint_ptr_t == size_t` like in `numpy.typing`)
_AnyUIntPNP: TypeAlias = np.uintp
if TYPE_CHECKING or _NP_V2:
    _AnyUIntPName: TypeAlias = Literal['uintp', 'uint']
    _AnyUIntPChar: TypeAlias = Literal['N', '=N', '<N', '>N']
else:
    _AnyUIntPName: TypeAlias = Literal['uintp', 'uint0']
    _AnyUIntPChar: TypeAlias = Literal['P', '=P', '<P', '>P']
_AnyUIntPCode: TypeAlias = _AnyUIntPName | _AnyUIntPChar
# Note that `np.array(ct.c_void_p())` will raise a `ValueError`:
# "Unknown PEP 3118 data type specifier 'P'".
AnyUIntPValue: TypeAlias = _AnyUIntPNP | ct.c_size_t  # | ct.c_void_p
# But on the other hand, `np.dtype(ct.c_void_p)` is fine.
AnyUIntPType: TypeAlias = (
    _AnyUIntPCode
    | _DualType[_AnyUIntPNP, AnyUIntPValue | ct.c_void_p]
)

# ulong (uint on numpy^=1)
if TYPE_CHECKING or _NP_V2:
    _AnyULongNP: TypeAlias = np.ulong
    _AnyULongName: TypeAlias = Literal['ulong']
else:
    _AnyULongNP: TypeAlias = np.uint
    _AnyULongName: TypeAlias = Literal['ulong', 'uint']
_AnyULongChar: TypeAlias = Literal['L', '=L', '<L', '>L']
_AnyULongCode: TypeAlias = _AnyULongName | _AnyULongChar
AnyULongValue: TypeAlias = _AnyULongNP | ct.c_ulong
AnyULongType: TypeAlias = _AnyULongCode | _DualType[_AnyULongNP, AnyULongValue]

# ulonglong
_AnyULongLongNP: TypeAlias = np.ulonglong
_AnyULongLongName: TypeAlias = Literal['ulonglong']
_AnyULongLongChar: TypeAlias = Literal['Q', '=Q', '<Q', '>Q']
_AnyULongLongCode: TypeAlias = _AnyULongLongName | _AnyULongLongChar
AnyULongLongValue: TypeAlias = _AnyULongLongNP | ct.c_ulonglong
AnyULongLongType: TypeAlias = (
    _AnyULongLongCode
    | _DualType[_AnyULongLongNP, AnyULongLongValue]
)

# unsignedinteger
_NB_unsignedinteger = TypeVar(
    '_NB_unsignedinteger',
    bound='npt.NBitBase',
    default=Any,
)
_AnyUnsignedIntegerNP: TypeAlias = np.unsignedinteger[_NB_unsignedinteger]
# fmt: off
_AnyUnsignedIntegerCT: TypeAlias = (
    ct.c_uint8 | ct.c_uint16 | ct.c_uint32 | ct.c_uint64
    | ct.c_ubyte | ct.c_ushort | ct.c_uint | ct.c_ulong | ct.c_ulonglong
    | ct.c_size_t  # | ct.c_void_p
)
# fmt: on
# `builtins.int` is signed
_AnyUnsignedIntegerCode: TypeAlias = (
    _AnyUInt8Code | _AnyUInt16Code | _AnyUInt32Code | _AnyUInt64Code
    | _AnyUByteCode | _AnyUShortCode | _AnyULongCode | _AnyULongLongCode
    | _AnyUIntCCode | _AnyUIntPCode
)
AnyUnsignedIntegerValue: TypeAlias = (
    _AnyUnsignedIntegerNP[_NB_unsignedinteger]
    | _AnyUnsignedIntegerCT
)
AnyUnsignedIntegerType: TypeAlias = (
    _AnyUnsignedIntegerCode
    | _DualType[
        _AnyUnsignedIntegerNP[_NB_unsignedinteger],
        AnyUnsignedIntegerValue[_NB_unsignedinteger] | ct.c_void_p,
    ]
)


#
# signed integers
#

# int8
_AnyInt8NP: TypeAlias = np.int8
_AnyInt8Name: TypeAlias = Literal['int8']
_AnyInt8Char: TypeAlias = Literal['i1', '=i1', '<i1', '>i1']
_AnyInt8Code: TypeAlias = _AnyInt8Name | _AnyInt8Char
AnyInt8Value: TypeAlias = _AnyInt8NP | ct.c_int8
AnyInt8Type: TypeAlias = _AnyInt8Code | _DualType[_AnyInt8NP, AnyInt8Value]

# int16
_AnyInt16NP: TypeAlias = np.int16
_AnyInt16Name: TypeAlias = Literal['int16']
_AnyInt16Char: TypeAlias = Literal['i2', '=i2', '<i2', '>i2']
_AnyInt16Code: TypeAlias = _AnyInt16Name | _AnyInt16Char
AnyInt16Value: TypeAlias = _AnyInt16NP | ct.c_int16
AnyInt16Type: TypeAlias = _AnyInt16Code | _DualType[_AnyInt16NP, AnyInt16Value]

# int32
_AnyInt32NP: TypeAlias = np.int32
_AnyInt32Name: TypeAlias = Literal['int32']
_AnyInt32Char: TypeAlias = Literal['i4', '=i4', '<i4', '>i4']
_AnyInt32Code: TypeAlias = _AnyInt32Name | _AnyInt32Char
AnyInt32Value: TypeAlias = _AnyInt32NP | ct.c_int32
AnyInt32Type: TypeAlias = _AnyInt32Code | _DualType[_AnyInt32NP, AnyInt32Value]

# int64
_AnyInt64NP: TypeAlias = np.int64
_AnyInt64Name: TypeAlias = Literal['int64']
_AnyInt64Char: TypeAlias = Literal['i8', '=i8', '<i8', '>i8']
_AnyInt64Code: TypeAlias = _AnyInt64Name | _AnyInt64Char
AnyInt64Value: TypeAlias = _AnyInt64NP | ct.c_int64
AnyInt64Type: TypeAlias = _AnyInt64Code | _DualType[_AnyInt64NP, AnyInt64Value]

# byte
_AnyByteNP: TypeAlias = np.byte
_AnyByteName: TypeAlias = Literal['byte']
_AnyByteChar: TypeAlias = Literal['b', '=b', '<b', '>b']
_AnyByteCode: TypeAlias = _AnyByteName | _AnyByteChar
AnyByteValue: TypeAlias = _AnyByteNP | ct.c_byte
AnyByteType: TypeAlias = _AnyByteCode | _DualType[_AnyByteNP, AnyByteValue]

# short
_AnyShortNP: TypeAlias = np.short
_AnyShortName: TypeAlias = Literal['short']
_AnyShortChar: TypeAlias = Literal['h', '=h', '<h', '>h']
_AnyShortCode: TypeAlias = _AnyShortName | _AnyShortChar
AnyShortValue: TypeAlias = _AnyShortNP | ct.c_short
AnyShortType: TypeAlias = _AnyShortCode | _DualType[_AnyShortNP, AnyShortValue]

# intc
_AnyIntCNP: TypeAlias = np.intc
_AnyIntCName: TypeAlias = Literal['intc']
_AnyIntCChar: TypeAlias = Literal['i', '=i', '<i', '>i']
_AnyIntCCode: TypeAlias = _AnyIntCName | _AnyIntCChar
AnyIntCValue: TypeAlias = _AnyIntCNP | ct.c_int
AnyIntCType: TypeAlias = _AnyIntCCode | _DualType[_AnyIntCNP, AnyIntCValue]

# intp (or int_ if numpy>=2)
_AnyIntPNP: TypeAlias = np.intp
_AnyIntPCT: TypeAlias = ct.c_ssize_t
if TYPE_CHECKING or _NP_V2:
    _AnyIntPName: TypeAlias = Literal['intp', 'int', 'int_']
    _AnyIntPChar: TypeAlias = Literal['n', '=n', '<n', '>n']
else:
    _AnyIntPName: TypeAlias = Literal['intp', 'int0']
    _AnyIntPChar: TypeAlias = Literal['p', '=p', '<p', '>p']
_AnyIntPCode: TypeAlias = _AnyIntPName | _AnyIntPChar
if TYPE_CHECKING or _NP_V2:
    AnyIntPValue: TypeAlias = _AnyIntPNP | _AnyIntPCT | int
else:
    AnyIntPValue: TypeAlias = _AnyIntPNP | _AnyIntPCT
AnyIntPType: TypeAlias = _AnyIntPCode | _DualType[np.intp, AnyIntPValue]

# long (or int_ if numpy<2)
if TYPE_CHECKING or _NP_V2:
    _AnyLongNP: TypeAlias = np.long
else:
    _AnyLongNP: TypeAlias = np.int_
_AnyLongCT: TypeAlias = ct.c_long
if TYPE_CHECKING or _NP_V2:
    _AnyLongName: TypeAlias = Literal['long']
else:
    _AnyLongName: TypeAlias = Literal['long', 'int', 'int_']
_AnyLongChar: TypeAlias = Literal['l', '=l', '<l', '>l']
_AnyLongCode: TypeAlias = _AnyLongName | _AnyLongChar
if TYPE_CHECKING or _NP_V2:
    AnyLongValue: TypeAlias = _AnyLongNP | _AnyLongCT
else:
    AnyLongValue: TypeAlias = _AnyLongNP | _AnyLongCT | int
AnyLongType: TypeAlias = _AnyLongCode | _DualType[_AnyLongNP, AnyLongValue]

# longlong
_AnyLongLongNP: TypeAlias = np.longlong
_AnyLongLongName: TypeAlias = Literal['longlong']
_AnyLongLongChar: TypeAlias = Literal['q', '=q', '<q', '>q']
_AnyLongLongCode: TypeAlias = _AnyLongLongName | _AnyLongLongChar
AnyLongLongValue: TypeAlias = _AnyLongLongNP | ct.c_longlong
AnyLongLongType: TypeAlias = (
    _AnyLongLongCode
    | _DualType[_AnyLongLongNP, AnyLongLongValue]
)

# signedinteger
_NB_signedinteger = TypeVar(
    '_NB_signedinteger',
    bound='npt.NBitBase',
    default=Any,
)
_AnySignedIntegerNP = np.signedinteger[_NB_signedinteger]
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
AnySignedIntegerValue: TypeAlias = (
    _AnySignedIntegerNP[_NB_signedinteger]
    | _AnySignedIntegerCT
    | _AnySignedIntegerPY
)
AnySignedIntegerType: TypeAlias = (
    _AnySignedIntegerCode
    | _DualType[
        _AnySignedIntegerNP[_NB_signedinteger],
        AnySignedIntegerValue[_NB_signedinteger],
    ]
)


#
# (unsigned | signed) integers
#

# integer
_NB_integer = TypeVar('_NB_integer', bound='npt.NBitBase', default=Any)
_AnyIntegerNP: TypeAlias = np.integer[_NB_integer]
_AnyIntegerCT: TypeAlias = _AnyUnsignedIntegerCT | _AnySignedIntegerCT
_AnyIntegerPY: TypeAlias = _AnySignedIntegerPY
_AnyIntegerCode: TypeAlias = _AnyUnsignedIntegerCode | _AnySignedIntegerCode
AnyIntegerValue: TypeAlias = (
    _AnyIntegerNP[_NB_integer]
    | _AnyIntegerCT
    | _AnyIntegerPY
)
AnyIntegerType: TypeAlias = (
    _AnyIntegerCode
    | _DualType[_AnyIntegerNP[_NB_integer], AnyIntegerValue[_NB_integer]]
)


#
# real floats
#

# float16
_AnyFloat16NP: TypeAlias = np.float16
_AnyFloat16Name: TypeAlias = Literal['float16']
_AnyFloat16Char: TypeAlias = Literal['f2', '=f2', '<f2', '>f2']
_AnyFloat16Code: TypeAlias = _AnyFloat16Name | _AnyFloat16Char
AnyFloat16Value: TypeAlias = _AnyFloat16NP
AnyFloat16Type: TypeAlias = _AnyFloat16Code | _SoloType[_AnyFloat16NP]

# float32
_AnyFloat32NP: TypeAlias = np.float32
_AnyFloat32Name: TypeAlias = Literal['float32']
_AnyFloat32Char: TypeAlias = Literal['f4', '=f4', '<f4', '>f4']
_AnyFloat32Code: TypeAlias = _AnyFloat32Name | _AnyFloat32Char
AnyFloat32Value: TypeAlias = _AnyFloat32NP | ct.c_float
AnyFloat32Type: TypeAlias = _AnyFloat32Code | _SoloType[_AnyFloat32NP]

# float64
_AnyFloat64NP: TypeAlias = np.float64
_AnyFloat64Name: TypeAlias = Literal['float64']
_AnyFloat64Char: TypeAlias = Literal['f8', '=f8', '<f8', '>f8']
_AnyFloat64Code: TypeAlias = _AnyFloat64Name | _AnyFloat64Char
AnyFloat64Value: TypeAlias = _AnyFloat64NP | ct.c_double | float
# (np.dtype(None) -> float64)
AnyFloat64Type: TypeAlias = _AnyFloat64Code | _SoloType[_AnyFloat64NP] | None

# half
_AnyHalfNP: TypeAlias = np.half
_AnyHalfName: TypeAlias = Literal['half']
_AnyHalfChar: TypeAlias = Literal['e', '=e', '<e', '>e']
_AnyHalfCode: TypeAlias = _AnyHalfName | _AnyHalfChar
AnyHalfValue: TypeAlias = _AnyHalfNP
AnyHalfType: TypeAlias = _AnyHalfCode | _SoloType[_AnyHalfNP]

# single
_AnySingleNP: TypeAlias = np.single
_AnySingleName: TypeAlias = Literal['single']
_AnySingleChar: TypeAlias = Literal['f', '=f', '<f', '>f']
_AnySingleCode: TypeAlias = _AnySingleName | _AnySingleChar
AnySingleValue: TypeAlias = _AnySingleNP | ct.c_float
AnySingleType: TypeAlias = (
    _AnySingleCode
    | _DualType[_AnySingleNP, AnySingleValue]
)

# double
_AnyDoubleNP: TypeAlias = np.double
if TYPE_CHECKING or _NP_V2:
    _AnyDoubleName: TypeAlias = Literal['double', 'float']
else:
    _AnyDoubleName: TypeAlias = Literal['double', 'float', 'float_']
_AnyDoubleChar: TypeAlias = Literal['d', '=d', '<d', '>d']
_AnyDoubleCode: TypeAlias = _AnyDoubleName | _AnyDoubleChar
AnyDoubleValue: TypeAlias = _AnyDoubleNP | ct.c_double | float
AnyDoubleType: TypeAlias = (
    _AnyDoubleCode
    | _DualType[_AnyDoubleNP, AnyDoubleValue]
)

# longdouble
_AnyLongDoubleNP: TypeAlias = np.longdouble
if TYPE_CHECKING or _NP_V2:
    _AnyLongDoubleName: TypeAlias = Literal['longdouble']
else:
    _AnyLongDoubleName: TypeAlias = Literal['longdouble', 'longfloat']
_AnyLongDoubleChar: TypeAlias = Literal['g', '=g', '<g', '>g']
_AnyLongDoubleCode: TypeAlias = _AnyLongDoubleName | _AnyLongDoubleChar
# Note that `np.array(ct.c_longdouble())` will raise a `ValueError`:
# "Unknown PEP 3118 data type specifier 'g'".
AnyLongDoubleValue: TypeAlias = _AnyLongDoubleNP
# But on the other hand, `np.dtype(ct.c_longdouble)` is fine.
AnyLongDoubleType: TypeAlias = (
    _AnyLongDoubleCode
    | _DualType[_AnyLongDoubleNP, AnyLongDoubleValue | ct.c_longdouble]
)

# floating
_NB_floating = TypeVar('_NB_floating', bound='npt.NBitBase', default=Any)
_AnyFloatingNP: TypeAlias = np.floating[_NB_floating]
_AnyFloatingCT: TypeAlias = ct.c_float | ct.c_double | ct.c_longdouble
_AnyFloatingPY: TypeAlias = float
# fmt: off
_AnyFloatingCode: TypeAlias = (
    _AnyFloat16Code | _AnyFloat32Code | _AnyFloat64Code
    | _AnyHalfCode | _AnySingleCode | _AnyDoubleCode | _AnyLongDoubleCode
)
# fmt: on
AnyFloatingValue: TypeAlias = (
    _AnyFloatingNP[_NB_floating]
    | _AnyFloatingCT
    | _AnyFloatingPY
)
AnyFloatingType: TypeAlias = (
    _AnyFloatingCode
    | _DualType[_AnyFloatingNP[_NB_floating], AnyFloatingValue[_NB_floating]]
)


#
# complex floats
#

# complex64
_AnyComplex64NP: TypeAlias = np.complex64
_AnyComplex64Name: TypeAlias = Literal['complex64']
_AnyComplex64Char: TypeAlias = Literal['c8', '=c8', '<c8', '>c8']
_AnyComplex64Code: TypeAlias = _AnyComplex64Name | _AnyComplex64Char
AnyComplex64Value: TypeAlias = _AnyComplex64NP
AnyComplex64Type: TypeAlias = _AnyComplex64Code | _SoloType[_AnyComplex64NP]

# complex128
_AnyComplex128NP: TypeAlias = np.complex128
_AnyComplex128PY: TypeAlias = complex
_AnyComplex128Name: TypeAlias = Literal['complex128']
_AnyComplex128Char: TypeAlias = Literal['c16', '=c16', '<c16', '>c16']
_AnyComplex128Code: TypeAlias = _AnyComplex128Name | _AnyComplex128Char
AnyComplex128Value: TypeAlias = _AnyComplex128NP | _AnyComplex128PY
AnyComplex128Type: TypeAlias = _AnyComplex128Code | _SoloType[_AnyComplex128NP]

# csingle
_AnyCSingleNP: TypeAlias = np.csingle
if TYPE_CHECKING or _NP_V2:
    _AnyCSingleName: TypeAlias = Literal['csingle']
else:
    _AnyCSingleName: TypeAlias = Literal['csingle', 'singlecomplex']
_AnyCSingleChar: TypeAlias = Literal['F', '=F', '<F', '>F']
_AnyCSingleCode: TypeAlias = _AnyCSingleName | _AnyCSingleChar
AnyCSingleValue: TypeAlias = _AnyCSingleNP
AnyCSingleType: TypeAlias = _AnyCSingleCode | _SoloType[_AnyCSingleNP]

# cdouble
_AnyCDoubleNP: TypeAlias = np.cdouble
_AnyCDoublePY: TypeAlias = complex
if TYPE_CHECKING or _NP_V2:
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
AnyCDoubleValue: TypeAlias = _AnyCDoubleNP | _AnyCDoublePY
AnyCDoubleType: TypeAlias = (
    _AnyCDoubleCode
    | _DualType[_AnyCDoubleNP, AnyCDoubleValue]
)

# clongdouble
AnyCLongDoubleValueNP: TypeAlias = np.clongdouble
if TYPE_CHECKING or _NP_V2:
    _AnyCLongDoubleName: TypeAlias = Literal['clongdouble']
else:
    _AnyCLongDoubleName: TypeAlias = Literal[
        'clongdouble',
        'clongfloat',
        'longcomplex',
    ]
_AnyCLongDoubleChar: TypeAlias = Literal['G', '=G', '<G', '>G']
_AnyCLongDoubleCode: TypeAlias = _AnyCLongDoubleName | _AnyCLongDoubleChar
AnyCLongDoubleValue: TypeAlias = AnyCLongDoubleValueNP
AnyCLongDoubleType: TypeAlias = (
    _AnyCLongDoubleCode
    | _SoloType[AnyCLongDoubleValueNP]
)

# complexfloating
_NB1_complexfloating = TypeVar(
    '_NB1_complexfloating',
    bound='npt.NBitBase',
    default=Any,
)
_NB2_complexfloating = TypeVar(
    '_NB2_complexfloating',
    bound='npt.NBitBase',
    default=_NB1_complexfloating,
)
_AnyComplexFloatingNP: TypeAlias = np.complexfloating[
    _NB1_complexfloating,
    _NB2_complexfloating,
]
_AnyComplexFloatingPY: TypeAlias = complex
# fmt: off
_AnyComplexFloatingCode: TypeAlias = (
    _AnyComplex64Code | _AnyComplex128Code
    | _AnyCSingleCode | _AnyCDoubleCode | _AnyCLongDoubleCode
)
# fmt: on
AnyComplexFloatingValue: TypeAlias = (
    _AnyComplexFloatingNP[_NB1_complexfloating, _NB2_complexfloating]
    | _AnyComplexFloatingPY
)
AnyComplexFloatingType: TypeAlias = (
    _AnyComplexFloatingCode
    | _DualType[
        _AnyComplexFloatingNP[_NB1_complexfloating, _NB2_complexfloating],
        AnyComplexFloatingValue[_NB1_complexfloating, _NB2_complexfloating],
    ]
)


#
# integers | floats
#

# inexact
_NB_inexact = TypeVar('_NB_inexact', bound='npt.NBitBase', default=Any)
_AnyInexactNP: TypeAlias = np.inexact[_NB_inexact]
_AnyInexactCT: TypeAlias = _AnyFloatingCT
_AnyInexactPY: TypeAlias = _AnyFloatingPY | _AnyComplexFloatingPY
_AnyInexactCode: TypeAlias = _AnyFloatingCode | _AnyComplexFloatingCode
AnyInexactValue: TypeAlias = (
    _AnyInexactNP[_NB_inexact]
    | _AnyInexactCT
    | _AnyInexactPY
)
AnyInexactType: TypeAlias = (
    _AnyInexactCode
    | _DualType[_AnyInexactNP[_NB_inexact], AnyInexactValue[_NB_inexact]]
)

# number
_NB_number = TypeVar('_NB_number', bound='npt.NBitBase', default=Any)
_AnyNumberNP: TypeAlias = np.number[_NB_number]
_AnyNumberCT: TypeAlias = _AnyIntegerCT | _AnyInexactCT
_AnyNumberPY: TypeAlias = _AnyIntegerPY | _AnyInexactPY
_AnyNumberCode: TypeAlias = _AnyIntegerCode | _AnyInexactCode
AnyNumberValue: TypeAlias = (
    _AnyNumberNP[_NB_number]
    | _AnyNumberCT
    | _AnyNumberPY
)
AnyNumberType: TypeAlias = (
    _AnyNumberCode
    | _DualType[_AnyNumberNP[_NB_number], AnyNumberValue[_NB_number]]
)


#
# temporal
#

# datetime64
# TODO: Rename to `DateTime64` (for `np.dtypes.DateTime64DType` consistency).
_AnyDatetime64NP: TypeAlias = np.datetime64
_AnyDatetime64Name: TypeAlias = Literal['datetime64']
# fmt: off
_AnyDatetime64Char: TypeAlias = Literal[
    'M', '=M', '<M', '>M',
    'M8', '=M8', '<M8', '>M8',
]
# fmt: on
_AnyDatetime64Code: TypeAlias = _AnyDatetime64Name | _AnyDatetime64Char
AnyDatetime64Value: TypeAlias = _AnyDatetime64NP
AnyDatetime64Type: TypeAlias = _AnyDatetime64Code | _SoloType[_AnyDatetime64NP]


# timedelta64
# TODO: Rename to `TimeDelta64` (for `np.dtypes.TimeDelta64` consistency).
_AnyTimedelta64NP: TypeAlias = np.timedelta64
_AnyTimedelta64Name: TypeAlias = Literal['timedelta64']
# fmt: off
_AnyTimedelta64Char: TypeAlias = Literal[
    'm', '=m', '<m', '>m',
    'm8', '=m8', '<m8', '>m8',
]
# fmt: on
_AnyTimedelta64Code: TypeAlias = _AnyTimedelta64Name | _AnyTimedelta64Char
AnyTimedelta64Value: TypeAlias = _AnyTimedelta64NP
AnyTimedelta64Type: TypeAlias = (
    _AnyTimedelta64Code
    | _SoloType[_AnyTimedelta64NP]
)


#
# character strings
#

# str
_AnyStrNP: TypeAlias = np.str_
_AnyStrPY: TypeAlias = str
if TYPE_CHECKING or _NP_V2:
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
AnyStrValue: TypeAlias = _AnyStrNP | _AnyStrPY
AnyStrType: TypeAlias = _AnyStrCode | _DualType[_AnyStrNP, AnyStrValue]

# bytes
_AnyBytesNP: TypeAlias = np.bytes_
_AnyBytesCT: TypeAlias = ct.c_char
_AnyBytesPY: TypeAlias = bytes
if TYPE_CHECKING or _NP_V2:
    _AnyBytesName: TypeAlias = Literal['bytes', 'bytes_']
else:
    _AnyBytesName: TypeAlias = Literal['bytes', 'bytes_', 'bytes0']
_AnyBytesChar: TypeAlias = Literal['S', '=S', '<S', '>S']
_AnyBytesCode: TypeAlias = _AnyBytesName | _AnyBytesChar
AnyBytesValue: TypeAlias = _AnyBytesNP | _AnyBytesPY
AnyBytesType: TypeAlias = _AnyBytesCode | _DualType[_AnyBytesNP, AnyBytesValue]

# character
_AnyCharacterNP: TypeAlias = np.character
_AnyCharacterCT: TypeAlias = _AnyBytesCT
_AnyCharacterPY: TypeAlias = _AnyStrPY | _AnyBytesPY
_AnyCharacterCode: TypeAlias = _AnyStrCode | _AnyBytesCode
AnyCharacterValue: TypeAlias = (
    _AnyCharacterNP
    | _AnyCharacterCT
    | _AnyCharacterPY
)
AnyCharacterType: TypeAlias = (
    _AnyCharacterCode
    | _DualType[_AnyCharacterNP, AnyCharacterValue]
)

# void
_AnyVoidNP: TypeAlias = np.void
if TYPE_CHECKING or _NP_V2:
    _AnyVoidName: TypeAlias = Literal['void']
else:
    _AnyVoidName: TypeAlias = Literal['void', 'void0']
_AnyVoidChar: TypeAlias = Literal['V', '=V', '<V', '>V']
_AnyVoidCode: TypeAlias = _AnyVoidName | _AnyVoidChar
AnyVoidValue: TypeAlias = _AnyVoidNP
AnyVoidType: TypeAlias = _AnyVoidCode | _SoloType[_AnyVoidNP]

# flexible
_AnyFlexibleNP: TypeAlias = np.flexible
_AnyFlexibleCT: TypeAlias = _AnyCharacterCT
_AnyFlexiblePY: TypeAlias = _AnyCharacterPY
_AnyFlexibleCode: TypeAlias = _AnyCharacterCode | _AnyVoidCode
AnyFlexibleValue: TypeAlias = _AnyFlexibleNP | _AnyFlexibleCT | _AnyFlexiblePY
AnyFlexibleType: TypeAlias = (
    _AnyFlexibleCode
    | _DualType[_AnyFlexibleNP, AnyFlexibleValue]
)

# string
# TODO(jorenham): Add `AnyString{Value,Type}` for `StringDType` on `numpy>=2`
# https://github.com/jorenham/optype/issues/99


#
# other types
#

# bool_
_AnyBoolNP: TypeAlias = np.bool_
_AnyBoolCT: TypeAlias = ct.c_bool
_AnyBoolPY: TypeAlias = bool
if TYPE_CHECKING or _NP_V2:
    _AnyBoolName: TypeAlias = Literal['bool', 'bool_']
else:
    _AnyBoolName: TypeAlias = Literal['bool', 'bool_', 'bool8']
_AnyBoolChar: TypeAlias = Literal['?', '=?', '<?', '>?']
_AnyBoolCode: TypeAlias = _AnyBoolName | _AnyBoolChar
AnyBoolValue: TypeAlias = _AnyBoolNP | _AnyBoolCT | _AnyBoolPY
AnyBoolType: TypeAlias = _AnyBoolCode | _DualType[_AnyBoolNP, AnyBoolValue]

# object
_AnyObjectNP: TypeAlias = np.object_
if TYPE_CHECKING:
    _AnyObjectCT: TypeAlias = ct.py_object[Any]
else:
    _AnyObjectCT: TypeAlias = ct.py_object
_AnyObjectPY: TypeAlias = object
_AnyObjectName: TypeAlias = Literal['object', 'object_']
_AnyObjectChar: TypeAlias = Literal['O', '=O', '<O', '>O']
_AnyObjectCode: TypeAlias = _AnyObjectName | _AnyObjectChar
AnyObjectValue: TypeAlias = _AnyObjectNP | _AnyObjectCT | _AnyObjectPY
AnyObjectType: TypeAlias = (
    _AnyObjectCode
    | _DualType[_AnyObjectNP, AnyObjectValue]
)


#
# any scalar
#

# generic
_NB_generic = TypeVar('_NB_generic', bound='npt.NBitBase', default=Any)
_AnyGenericNP: TypeAlias = (
    _AnyBoolNP
    | _AnyObjectNP
    | _AnyNumberNP[_NB_generic]
    | _AnyFlexibleNP
    | np.generic  # catch-all for any other user-defined scalar types
)
_AnyGenericCT: TypeAlias = (  # noqa: PYI047
    _AnyBoolCT
    | _AnyObjectCT
    | _AnyNumberCT
    | _AnyFlexibleCT
)
_AnyGenericCode: TypeAlias = (
    _AnyBoolCode
    | _AnyNumberCode
    | _AnyFlexibleCode
    | _AnyObjectCode
)
AnyGenericValue: TypeAlias = (
    AnyBoolValue
    | AnyNumberValue[_NB_generic]
    | AnyFlexibleValue
    | AnyObjectValue
    | np.generic  # catch-all for any other user-defined scalar types
)
AnyGenericType: TypeAlias = (
    _AnyGenericCode
    | _DualType[_AnyGenericNP[_NB_generic], AnyGenericValue[_NB_generic]]
)
