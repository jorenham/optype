import ctypes as ct
from typing import Final, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt


_NP_V1: Final[bool] = np.__version__.startswith('1.')


_T = TypeVar('_T', bound=object)
_S = TypeVar('_S', bound=np.generic)
_N = TypeVar('_N', bound=npt.NBitBase)

_SoloType: TypeAlias = np.dtype[_S] | type[_S]
_DualType: TypeAlias = np.dtype[_S] | type[_T]

# bool
if _NP_V1:
    AnyBool: TypeAlias = np.bool_ | ct.c_bool | bool
    AnyBoolName: TypeAlias = Literal['bool', 'bool_', 'bool8']
else:
    AnyBool: TypeAlias = np.bool | ct.c_bool | bool  # noqa: NPY001
    AnyBoolName: TypeAlias = Literal['bool', 'bool_']
AnyBoolChar: TypeAlias = Literal['?', '=?', '<?', '>?']
AnyBoolCode: TypeAlias = AnyBoolName | AnyBoolChar
AnyBoolType: TypeAlias = AnyBoolCode | _DualType[np.bool_, AnyBool]

# uint8
AnyUInt8: TypeAlias = np.uint8 | ct.c_uint8
AnyUInt8Name: TypeAlias = Literal['uint8']
AnyUInt8Char: TypeAlias = Literal['u1', '=u1', '<u1', '>u1']
AnyUInt8Code: TypeAlias = AnyUInt8Name | AnyUInt8Char
AnyUInt8Type: TypeAlias = AnyUInt8Code | _DualType[np.uint8, AnyUInt8]

# uint16
AnyUInt16: TypeAlias = np.uint16 | ct.c_uint16
AnyUInt16Name: TypeAlias = Literal['uint16']
AnyUInt16Char: TypeAlias = Literal['u2', '=u2', '<u2', '>u2']
AnyUInt16Code: TypeAlias = AnyUInt16Name | AnyUInt16Char
AnyUInt16Type: TypeAlias = AnyUInt16Code | _DualType[np.uint16, AnyUInt16]

# uint32
AnyUInt32: TypeAlias = np.uint32 | ct.c_uint32
AnyUInt32Name: TypeAlias = Literal['uint32']
AnyUInt32Char: TypeAlias = Literal['u4', '=u4', '<u4', '>u4']
AnyUInt32Code: TypeAlias = AnyUInt32Name | AnyUInt32Char
AnyUInt32Type: TypeAlias = AnyUInt32Code | _DualType[np.uint32, AnyUInt32]

# uint64
AnyUInt64: TypeAlias = np.uint64 | ct.c_uint64
AnyUInt64Name: TypeAlias = Literal['uint64']
AnyUInt64Char: TypeAlias = Literal['u8', '=u8', '<u8', '>u8']
AnyUInt64Code: TypeAlias = AnyUInt64Name | AnyUInt64Char
AnyUInt64Type: TypeAlias = AnyUInt64Code | _DualType[np.uint64, AnyUInt64]

# ubyte
AnyUByte: TypeAlias = np.ubyte | ct.c_ubyte
AnyUByteName: TypeAlias = Literal['ubyte']
AnyUByteChar: TypeAlias = Literal['B', '=B', '<B', '>B']
AnyUByteCode: TypeAlias = AnyUByteName | AnyUByteChar
AnyUByteType: TypeAlias = AnyUByteCode | _DualType[np.ubyte, AnyUByte]

# ushort
AnyUShort: TypeAlias = np.ushort | ct.c_ushort
AnyUShortName: TypeAlias = Literal['ushort']
AnyUShortChar: TypeAlias = Literal['H', '=H', '<H', '>H']
AnyUShortCode: TypeAlias = AnyUShortName | AnyUShortChar
AnyUShortType: TypeAlias = AnyUShortCode | _DualType[np.ushort, AnyUShort]

# uintc
AnyUIntC: TypeAlias = np.uintc | ct.c_uint
AnyUIntCName: TypeAlias = Literal['uintc']
AnyUIntCChar: TypeAlias = Literal['I', '=I', '<I', '>I']
AnyUIntCCode: TypeAlias = AnyUIntCName | AnyUIntCChar
AnyUIntCType: TypeAlias = AnyUIntCCode | _DualType[np.uintc, AnyUIntC]

# uintp (assuming `uint_ptr_t == size_t` like in `numpy.typing`)
AnyUIntP: TypeAlias = np.uintp | ct.c_void_p | ct.c_size_t
if _NP_V1:
    AnyUIntPName: TypeAlias = Literal['uintp', 'uint0']
    AnyUIntPChar: TypeAlias = Literal['P', '=P', '<P', '>P']
else:
    AnyUIntPName: TypeAlias = Literal['uintp', 'uint']
    AnyUIntPChar: TypeAlias = Literal['N', '=N', '<N', '>N']
AnyUIntPCode: TypeAlias = AnyUIntPName | AnyUIntPChar
AnyUIntPType: TypeAlias = AnyUIntPCode | _DualType[np.uintp, AnyUIntP]

# ulong (uint on numpy^=1)
if _NP_V1:
    AnyULong: TypeAlias = np.uint | ct.c_ulong
    AnyULongName: TypeAlias = Literal['ulong', 'uint']
else:
    AnyULong: TypeAlias = np.ulong | ct.c_ulong
    AnyULongName: TypeAlias = Literal['ulong']
AnyULongChar: TypeAlias = Literal['L', '=L', '<L', '>L']
AnyULongCode: TypeAlias = AnyULongName | AnyULongChar
AnyULongType: TypeAlias = AnyULongCode | _DualType[np.uint, AnyULong]

# uint (ulong on numpy^=1, uintp on numpy^=2)
if _NP_V1:
    AnyUInt: TypeAlias = np.uint | ct.c_ulong
    AnyUIntName: TypeAlias = AnyULongName
    AnyUIntChar: TypeAlias = AnyULongChar
    AnyUIntCode: TypeAlias = AnyULongCode
    AnyUIntType: TypeAlias = AnyULongType
else:
    AnyUInt: TypeAlias = np.uint | ct.c_void_p | ct.c_size_t
    AnyUIntName: TypeAlias = AnyUIntPName
    AnyUIntChar: TypeAlias = AnyUIntPChar
    AnyUIntCode: TypeAlias = AnyUIntPCode
    AnyUIntType: TypeAlias = AnyUIntPType

# ulonglong
AnyULongLong: TypeAlias = np.ulonglong | ct.c_ulonglong
AnyULongLongName: TypeAlias = Literal['ulonglong']
AnyULongLongChar: TypeAlias = Literal['Q', '=Q', '<Q', '>Q']
AnyULongLongCode: TypeAlias = AnyULongLongName | AnyULongLongChar
AnyULongLongType: TypeAlias = (
    AnyULongLongCode | _DualType[np.ulonglong, AnyULongLong]
)

# int8
AnyInt8: TypeAlias = np.int8 | ct.c_int8
AnyInt8Name: TypeAlias = Literal['int8']
AnyInt8Char: TypeAlias = Literal['i1', '=i1', '<i1', '>i1']
AnyInt8Code: TypeAlias = AnyInt8Name | AnyInt8Char
AnyInt8Type: TypeAlias = AnyInt8Code | _DualType[np.int8, AnyInt8]

# int16
AnyInt16: TypeAlias = np.int16 | ct.c_int16
AnyInt16Name: TypeAlias = Literal['int16']
AnyInt16Char: TypeAlias = Literal['i2', '=i2', '<i2', '>i2']
AnyInt16Code: TypeAlias = AnyInt16Name | AnyInt16Char
AnyInt16Type: TypeAlias = AnyInt16Code | _DualType[np.int16, AnyInt16]

# int32
AnyInt32: TypeAlias = np.int32 | ct.c_int32
AnyInt32Name: TypeAlias = Literal['int32']
AnyInt32Char: TypeAlias = Literal['i4', '=i4', '<i4', '>i4']
AnyInt32Code: TypeAlias = AnyInt32Name | AnyInt32Char
AnyInt32Type: TypeAlias = AnyInt32Code | _DualType[np.int32, AnyInt32]

# int64
AnyInt64: TypeAlias = np.int64 | ct.c_int64
AnyInt64Name: TypeAlias = Literal['int64']
AnyInt64Char: TypeAlias = Literal['i8', '=i8', '<i8', '>i8']
AnyInt64Code: TypeAlias = AnyInt64Name | AnyInt64Char
AnyInt64Type: TypeAlias = AnyInt64Code | _DualType[np.int64, AnyInt64]

# byte
AnyByte: TypeAlias = np.byte | ct.c_byte
AnyByteName: TypeAlias = Literal['byte']
AnyByteChar: TypeAlias = Literal['b', '=b', '<b', '>b']
AnyByteCode: TypeAlias = AnyByteName | AnyByteChar
AnyByteType: TypeAlias = AnyByteCode | _DualType[np.byte, AnyByte]

# short
AnyShort: TypeAlias = np.short | ct.c_short
AnyShortName: TypeAlias = Literal['short']
AnyShortChar: TypeAlias = Literal['h', '=h', '<h', '>h']
AnyShortCode: TypeAlias = AnyShortName | AnyShortChar
AnyShortType: TypeAlias = AnyShortCode | _DualType[np.short, AnyShort]

# intc
AnyIntC: TypeAlias = np.intc | ct.c_int
AnyIntCName: TypeAlias = Literal['intc']
AnyIntCChar: TypeAlias = Literal['i', '=i', '<i', '>i']
AnyIntCCode: TypeAlias = AnyIntCName | AnyIntCChar
AnyIntCType: TypeAlias = AnyIntCCode | _DualType[np.intc, AnyIntC]

# intp
AnyIntP: TypeAlias = np.intp | ct.c_ssize_t
if _NP_V1:
    AnyIntPName: TypeAlias = Literal['intp', 'int0']
    AnyIntPChar: TypeAlias = Literal['p', '=p', '<p', '>p']
else:
    AnyIntPName: TypeAlias = Literal['intp', 'int', 'int_']
    AnyIntPChar: TypeAlias = Literal['n', '=n', '<n', '>n']
AnyIntPCode: TypeAlias = AnyIntPName | AnyIntPChar
AnyIntPType: TypeAlias = AnyIntPCode | _DualType[np.intp, AnyIntP]

# long (int_ on numpy^=1)
if _NP_V1:
    AnyLong: TypeAlias = np.int_ | ct.c_long | int
    AnyLongName: TypeAlias = Literal['long', 'int', 'int_']
else:
    AnyLong: TypeAlias = np.long | ct.c_long  # noqa: NPY001
    AnyLongName: TypeAlias = Literal['long']
AnyLongChar: TypeAlias = Literal['l', '=l', '<l', '>l']
AnyLongCode: TypeAlias = AnyLongName | AnyLongChar
AnyLongType: TypeAlias = AnyLongCode | _DualType[np.int_, AnyLong]

# int (long on numpy^=1, intp on numpy^=2)
if _NP_V1:
    AnyInt: TypeAlias = AnyLong
    AnyIntName: TypeAlias = AnyLongName
    AnyIntChar: TypeAlias = AnyLongChar
    AnyIntCode: TypeAlias = AnyLongCode
    AnyIntType: TypeAlias = AnyLongType
else:
    AnyInt: TypeAlias = np.int_ | ct.c_ssize_t | int
    AnyIntName: TypeAlias = AnyIntPName
    AnyIntChar: TypeAlias = AnyIntPChar
    AnyIntCode: TypeAlias = AnyIntPCode
    AnyIntType: TypeAlias = AnyIntPType

# longlong
AnyLongLong: TypeAlias = np.longlong | ct.c_longlong
AnyLongLongName: TypeAlias = Literal['longlong']
AnyLongLongChar: TypeAlias = Literal['q', '=q', '<q', '>q']
AnyLongLongCode: TypeAlias = AnyLongLongName | AnyLongLongChar
AnyLongLongType: TypeAlias = (
    AnyLongLongCode | _DualType[np.longlong, AnyLongLong]
)

# float16
AnyFloat16: TypeAlias = np.float16
AnyFloat16Name: TypeAlias = Literal['float16']
AnyFloat16Char: TypeAlias = Literal['f2', '=f2', '<f2', '>f2']
AnyFloat16Code: TypeAlias = AnyFloat16Name | AnyFloat16Char
AnyFloat16Type: TypeAlias = AnyFloat16Code | _SoloType[np.float16]

# float32
AnyFloat32: TypeAlias = np.float32
AnyFloat32Name: TypeAlias = Literal['float32']
AnyFloat32Char: TypeAlias = Literal['f4', '=f4', '<f4', '>f4']
AnyFloat32Code: TypeAlias = AnyFloat32Name | AnyFloat32Char
AnyFloat32Type: TypeAlias = AnyFloat32Code | _SoloType[np.float32]

# float64
AnyFloat64: TypeAlias = np.float64
AnyFloat64Name: TypeAlias = Literal['float64']
AnyFloat64Char: TypeAlias = Literal['f8', '=f8', '<f8', '>f8']
AnyFloat64Code: TypeAlias = AnyFloat64Name | AnyFloat64Char
AnyFloat64Type: TypeAlias = AnyFloat64Code | _SoloType[np.float64]

# half
AnyHalf: TypeAlias = np.half
AnyHalfName: TypeAlias = Literal['half']
AnyHalfChar: TypeAlias = Literal['e', '=e', '<e', '>e']
AnyHalfCode: TypeAlias = AnyHalfName | AnyHalfChar
AnyHalfType: TypeAlias = AnyHalfCode | _SoloType[np.half]

# single
AnySingle: TypeAlias = np.single | ct.c_float
AnySingleName: TypeAlias = Literal['single']
AnySingleChar: TypeAlias = Literal['f', '=f', '<f', '>f']
AnySingleCode: TypeAlias = AnySingleName | AnySingleChar
AnySingleType: TypeAlias = AnySingleCode | _DualType[np.single, AnySingle]

# double
if _NP_V1:
    AnyDouble: TypeAlias = np.double | ct.c_double | float
    AnyDoubleName: TypeAlias = Literal['double', 'float', 'float_']
else:
    AnyDouble: TypeAlias = np.double | ct.c_double
    AnyDoubleName: TypeAlias = Literal['double', 'float']
AnyDoubleChar: TypeAlias = Literal['d', '=d', '<d', '>d']
AnyDoubleCode: TypeAlias = AnyDoubleName | AnyDoubleChar
AnyDoubleType: TypeAlias = AnyDoubleCode | _DualType[np.double, AnyDouble]

# float (float_ / double on numpy^=1, float64 on numpy^=2)
if _NP_V1:
    AnyFloat: TypeAlias = AnyDouble
    AnyFloatName: TypeAlias = AnyDoubleName
    AnyFloatChar: TypeAlias = AnyDoubleChar
    AnyFloatCode: TypeAlias = AnyDoubleCode
    AnyFloatType: TypeAlias = AnyDoubleType
else:
    AnyFloat: TypeAlias = AnyFloat64
    AnyFloatName: TypeAlias = AnyFloat64Name
    AnyFloatChar: TypeAlias = AnyFloat64Char
    AnyFloatCode: TypeAlias = AnyFloat64Code
    AnyFloatType: TypeAlias = AnyFloat64Type

# longdouble
AnyLongDouble: TypeAlias = np.longdouble | ct.c_longdouble
if _NP_V1:
    AnyLongDoubleName: TypeAlias = Literal['longdouble', 'longfloat']
else:
    AnyLongDoubleName: TypeAlias = Literal['longdouble']
AnyLongDoubleChar: TypeAlias = Literal['g', '=g', '<g', '>g']
AnyLongDoubleCode: TypeAlias = AnyLongDoubleName | AnyLongDoubleChar
AnyLongDoubleType: TypeAlias = (
    AnyLongDoubleCode | _DualType[np.longdouble, AnyLongDouble]
)

# complex64
AnyComplex64: TypeAlias = np.complex64
AnyComplex64Name: TypeAlias = Literal['complex64']
AnyComplex64Char: TypeAlias = Literal['c8', '=c8', '<c8', '>c8']
AnyComplex64Code: TypeAlias = AnyComplex64Name | AnyComplex64Char
AnyComplex64Type: TypeAlias = AnyComplex64Code | _SoloType[np.complex64]

# complex128
AnyComplex128: TypeAlias = np.complex128
AnyComplex128Name: TypeAlias = Literal['complex128']
AnyComplex128Char: TypeAlias = Literal['c16', '=c16', '<c16', '>c16']
AnyComplex128Code: TypeAlias = AnyComplex128Name | AnyComplex128Char
AnyComplex128Type: TypeAlias = AnyComplex128Code | _SoloType[np.complex128]

# csingle
AnyCSingle: TypeAlias = np.csingle
if _NP_V1:
    AnyCSingleName: TypeAlias = Literal['csingle', 'singlecomplex']
else:
    AnyCSingleName: TypeAlias = Literal['csingle']
AnyCSingleChar: TypeAlias = Literal['F', '=F', '<F', '>F']
AnyCSingleCode: TypeAlias = AnyCSingleName | AnyCSingleChar
AnyCSingleType: TypeAlias = AnyCSingleCode | _SoloType[np.csingle]

# cdouble
AnyCDouble: TypeAlias = np.cdouble | complex
_CDoubleNameV2: TypeAlias = Literal['cdouble', 'complex']
if _NP_V1:
    AnyCDoubleName: TypeAlias = _CDoubleNameV2 | Literal['complex_', 'cfloat']
else:
    AnyCDoubleName: TypeAlias = _CDoubleNameV2
AnyCDoubleChar: TypeAlias = Literal['D', '=D', '<D', '>D']
AnyCDoubleCode: TypeAlias = AnyCDoubleName | AnyCDoubleChar
AnyCDoubleType: TypeAlias = AnyCDoubleCode | _DualType[np.cdouble, AnyCDouble]

# complex (complex_ / cdouble on numpy^=1, complex128 on numpy^=2)
if _NP_V1:
    AnyComplex: TypeAlias = AnyCDouble
    AnyComplexName: TypeAlias = AnyCDoubleName
    AnyComplexChar: TypeAlias = AnyCDoubleChar
    AnyComplexCode: TypeAlias = AnyCDoubleCode
    AnyComplexType: TypeAlias = AnyCDoubleType
else:
    AnyComplex: TypeAlias = AnyComplex128
    AnyComplexName: TypeAlias = AnyComplex128Name
    AnyComplexChar: TypeAlias = AnyComplex128Char
    AnyComplexCode: TypeAlias = AnyComplex128Code
    AnyComplexType: TypeAlias = AnyComplex128Type

# clongdouble
AnyCLongDouble: TypeAlias = np.clongdouble
_CLongDoubleNameV2: TypeAlias = Literal['clongdouble']
if _NP_V1:
    AnyCLongDoubleName: TypeAlias = (
        _CLongDoubleNameV2 | Literal['clongfloat', 'longcomplex']
    )
else:
    AnyCLongDoubleName: TypeAlias = _CLongDoubleNameV2
AnyCLongDoubleChar: TypeAlias = Literal['G', '=G', '<G', '>G']
AnyCLongDoubleCode: TypeAlias = AnyCLongDoubleName | AnyCLongDoubleChar
AnyCLongDoubleType: TypeAlias = AnyCLongDoubleCode | _SoloType[np.clongdouble]

# timedelta64
AnyTimedelta64: TypeAlias = np.timedelta64
AnyTimedelta64Name: TypeAlias = Literal['timedelta64']
AnyTimedelta64Char: TypeAlias = Literal[
    'm', '=m', '<m', '>m',
    'm8', '=m8', '<m8', '>m8',
]
AnyTimedelta64Code: TypeAlias = AnyTimedelta64Name | AnyTimedelta64Char
AnyTimedelta64Type: TypeAlias = AnyTimedelta64Code | _SoloType[np.timedelta64]

# datetime64
AnyDatetime64: TypeAlias = np.datetime64
AnyDatetime64Name: TypeAlias = Literal['datetime64']
AnyDatetime64Char: TypeAlias = Literal[
    'M', '=M', '<M', '>M',
    'M8', '=M8', '<M8', '>M8',
]
AnyDatetime64Code: TypeAlias = AnyDatetime64Name | AnyDatetime64Char
AnyDatetime64Type: TypeAlias = AnyDatetime64Code | _SoloType[np.datetime64]

# str
AnyStr: TypeAlias = np.str_ | str
_StrNameV2: TypeAlias = Literal['str', 'str_', 'unicode']
if _NP_V1:
    AnyStrName: TypeAlias = _StrNameV2 | Literal['str0', 'unicode_']
else:
    AnyStrName: TypeAlias = _StrNameV2
AnyStrChar: TypeAlias = Literal['U', '=U', '<U', '>U']
AnyStrCode: TypeAlias = AnyStrName | AnyStrChar
AnyStrType: TypeAlias = AnyStrCode | _DualType[np.str_, AnyStr]

# bytes
AnyBytes: TypeAlias = np.bytes_ | bytes
_BytesNameV2: TypeAlias = Literal['bytes', 'bytes_']
if _NP_V1:
    AnyBytesName: TypeAlias = _BytesNameV2 | Literal['bytes0']
else:
    AnyBytesName: TypeAlias = _BytesNameV2
AnyBytesChar: TypeAlias = Literal['S', '=S', '<S', '>S']
AnyBytesCode: TypeAlias = AnyBytesName | AnyBytesChar
AnyBytesType: TypeAlias = AnyBytesCode | _DualType[np.bytes_, AnyBytes]

# void
AnyVoid: TypeAlias = np.void
_VoidNameV2: TypeAlias = Literal['void']
if _NP_V1:
    AnyVoidName: TypeAlias = _VoidNameV2 | Literal['void0']
else:
    AnyVoidName: TypeAlias = _VoidNameV2
AnyVoidChar: TypeAlias = Literal['V', '=V', '<V', '>V']
AnyVoidCode: TypeAlias = AnyVoidName | AnyVoidChar
AnyVoidType: TypeAlias = AnyVoidCode | _SoloType[np.void]

# object
AnyObject: TypeAlias = np.object_ | ct.py_object  # pyright: ignore[reportMissingTypeArgument]
AnyObjectName: TypeAlias = Literal['object', 'object_']
AnyObjectChar: TypeAlias = Literal['O', '=O', '<O', '>O']
AnyObjectCode: TypeAlias = AnyObjectName | AnyObjectChar
AnyObjectType: TypeAlias = AnyObjectCode | _DualType[np.object_, AnyObject]

#########

# np.unsignedinteger
_CUnsignedInteger: TypeAlias = (
    ct.c_uint8 | ct.c_uint16 | ct.c_uint32 | ct.c_uint64
    | ct.c_ubyte | ct.c_ushort | ct.c_uint | ct.c_ulong | ct.c_ulonglong
    | ct.c_size_t | ct.c_void_p
)
SomeUnsignedInteger: TypeAlias = np.unsignedinteger[_N] | _CUnsignedInteger

# np.signedinteger
_CSignedInteger: TypeAlias = (
    ct.c_int8 | ct.c_int16 | ct.c_int32 | ct.c_int64
    | ct.c_byte | ct.c_short | ct.c_int | ct.c_long | ct.c_longlong
    | ct.c_ssize_t
)
SomeSignedInteger: TypeAlias = np.signedinteger[_N] | int | _CSignedInteger

# np.integer
_CInteger: TypeAlias = _CUnsignedInteger | _CSignedInteger
SomeInteger: TypeAlias = np.integer[_N] | int | _CInteger

# np.floating
_CFloating: TypeAlias = ct.c_float | ct.c_double | ct.c_longdouble
SomeFloating: TypeAlias = np.floating[_N] | float | _CFloating

# np.complexfloating
SomeComplexFloating: TypeAlias = np.complexfloating[_N, _N] | complex

# np.inexact
SomeInexact: TypeAlias = np.inexact[_N] | float | complex | _CFloating

# np.number
_PyNumber: TypeAlias = int | float | complex
_CNumber: TypeAlias = _CInteger | _CFloating
SomeNumber: TypeAlias = np.number[_N] | _PyNumber | _CNumber

# np.character
_PyCharacter: TypeAlias = str | bytes
_CCharacter: TypeAlias = ct.c_char
SomeCharacter: TypeAlias = np.character | _PyCharacter | _CCharacter

# np.flexible
SomeFlexible: TypeAlias = np.flexible | _PyCharacter | _CCharacter

# np.generic
_PyGeneric: TypeAlias = _PyNumber | _PyCharacter
_CGeneric: TypeAlias = _CNumber | _CCharacter | ct.py_object  # pyright: ignore[reportMissingTypeArgument]
SomeGeneric: TypeAlias = np.generic | _CGeneric | _PyGeneric
