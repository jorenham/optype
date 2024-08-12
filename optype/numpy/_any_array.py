from __future__ import annotations

import sys
from typing import TypeAlias as _Type

import numpy as np

import optype.numpy._array as _a
import optype.numpy._compat as _x
import optype.numpy._ctype as _ct


if sys.version_info >= (3, 13):
    from typing import Any, Never, Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import (
        Any,
        Never,
        Protocol,
        TypeVar,
        runtime_checkable,
    )


# ruff: noqa: RUF022
__all__: list[str] = []
__all__ += [
    'AnyGenericArray',
    'AnyNumberArray',
    'AnyIntegerArray',
    'AnyUnsignedIntegerArray',
    'AnySignedIntegerArray',
    'AnyInexactArray',
    'AnyFloatingArray',
    'AnyComplexFloatingArray',
    'AnyFlexibleArray',
    'AnyCharacterArray',
]
__all__ += [
    'AnyUInt8Array',
    'AnyUInt8Array',
    'AnyUInt16Array',
    'AnyUInt32Array',
    'AnyUInt64Array',
    'AnyUIntPArray',
    'AnyUByteArray',
    'AnyUShortArray',
    'AnyUIntCArray',
    'AnyULongArray',
    'AnyULongLongArray',
]
__all__ += [
    'AnyIntegerArray',
    'AnyInt8Array',
    'AnyInt8Array',
    'AnyInt16Array',
    'AnyInt32Array',
    'AnyInt64Array',
    'AnyIntPArray',
    'AnyByteArray',
    'AnyShortArray',
    'AnyIntCArray',
    'AnyLongArray',
    'AnyLongLongArray',
]
__all__ += [
    'AnyFloatingArray',
    'AnyFloat16Array',
    'AnyFloat32Array',
    'AnyFloat64Array',
    'AnyHalfArray',
    'AnySingleArray',
    'AnyLongDoubleArray',
]
__all__ += [
    'AnyComplex64Array',
    'AnyComplex128Array',
    'AnyCSingleArray',
    'AnyCDoubleArray',
    'AnyCLongDoubleArray',
]
__all__ += [
    'AnyBytesArray',
    'AnyStrArray',
    'AnyVoidArray',
]
__all__ += [
    'AnyBoolArray',
    'AnyDateTime64Array',
    'AnyTimeDelta64Array',
    'AnyObjectArray',
]
__all__ += ['AnyStringArray']


T_co = TypeVar('T_co', covariant=True)


@runtime_checkable
class _PyArray(Protocol[T_co]):
    def __len__(self, /) -> int: ...
    def __getitem__(self, i: int, /) -> T_co | _PyArray[T_co]: ...


_ND = TypeVar('_ND', bound=tuple[int, ...])
_ST = TypeVar('_ST', bound=np.generic)
_PT = TypeVar('_PT', bool, int, float, complex, str, bytes, object)
_CT = TypeVar('_CT', bound=_ct.Generic)
_AnyNP: _Type = _a.CanArray[_ND, _ST] | _PyArray[_a.CanArray[Any, _ST]]
_AnyPY: _Type = _PyArray[_PT] | _PT
_Any3: _Type = _AnyNP[_ND, _ST] | _AnyPY[_PT]
_Any4: _Type = _Any3[_ND, _ST, _PT] | _CT

# generic
ND = TypeVar('ND', bound=tuple[int, ...], default=Any)
ST = TypeVar('ST', bound=np.generic, default=Any)
AnyArray: _Type = (
    _AnyNP[ND, ST]
    | _AnyPY[bool]
    | _AnyPY[int]
    | _AnyPY[float]
    | _AnyPY[complex]
    | _AnyPY[str]
    | _AnyPY[bytes]
    | _ct.Generic
    | _AnyPY[object]
)
AnyGenericArray: _Type = AnyArray[ND, np.generic]
__all__ += ['AnyArray', 'AnyGenericArray']

# generic :> number
ST_iufc = TypeVar('ST_iufc', bound=np.number[Any], default=Any)
AnyNumberArray: _Type = (
    _AnyNP[ND, ST_iufc]
    | _AnyPY[int]
    | _AnyPY[float]
    | _AnyPY[complex]
    | _ct.Number
)

# generic :> number :> integer
AnyIntegerArray: _Type = _Any4[ND, np.integer[Any], int, _ct.Integer]

# generic :> number :> integer :> unsignedinteger
AnyUnsignedIntegerArray: _Type = (
    _AnyNP[ND, np.unsignedinteger[Any]]
    | _ct.UnsignedInteger
)
# generic :> number :> integer :> unsignedinteger :> *
AnyUInt8Array: _Type = _AnyNP[ND, np.uint8] | _ct.UInt8
AnyUInt16Array: _Type = _AnyNP[ND, np.uint16] | _ct.UInt16
AnyUInt32Array: _Type = _AnyNP[ND, np.uint32] | _ct.UInt32
AnyUInt64Array: _Type = _AnyNP[ND, np.uint64] | _ct.UInt64
AnyUIntPArray: _Type = _AnyNP[ND, np.uint64] | _ct.UIntP
AnyUByteArray: _Type = _AnyNP[ND, np.ubyte] | _ct.UByte
AnyUShortArray: _Type = _AnyNP[ND, np.ushort] | _ct.UShort
AnyUIntCArray: _Type = _AnyNP[ND, np.uintc] | _ct.UIntC
AnyULongArray: _Type = _AnyNP[ND, np.ulong] | _ct.ULong
AnyULongLongArray: _Type = _AnyNP[ND, np.ulonglong] | _ct.ULongLong

# generic :> number :> integer :> signedinteger
AnySignedIntegerArray: _Type = _Any4[
    ND,
    np.signedinteger[Any],
    int,
    _ct.SignedInteger,
]
# generic :> number :> integer :> signedinteger :> *
AnyInt8Array: _Type = _AnyNP[ND, np.int8] | _ct.Int8
AnyInt16Array: _Type = _AnyNP[ND, np.int16] | _ct.Int16
AnyInt32Array: _Type = _AnyNP[ND, np.int32] | _ct.Int32
AnyInt64Array: _Type = _AnyNP[ND, np.int64] | _ct.Int64
if _x.NP2:
    AnyIntPArray: _Type = _Any4[ND, np.int64, int, _ct.IntP]
else:
    AnyIntPArray: _Type = _AnyNP[ND, np.int64] | _ct.IntP
AnyByteArray: _Type = _AnyNP[ND, np.byte] | _ct.Byte
AnyShortArray: _Type = _AnyNP[ND, np.short] | _ct.Short
AnyIntCArray: _Type = _AnyNP[ND, np.intc] | _ct.IntC
if _x.NP2:
    AnyLongArray: _Type = _AnyNP[ND, _x.Long] | _ct.Long
else:
    AnyLongArray: _Type = _Any4[ND, _x.Long, int, _ct.Long]
AnyLongLongArray: _Type = _AnyNP[ND, np.longlong] | _ct.LongLong

# generic :> number :> inexact
AnyInexactArray: _Type = (
    _AnyNP[ND, np.inexact[Any]]
    | _AnyPY[float]
    | _AnyPY[complex]
    | _ct.Inexact
)

# generic :> number :> inexact :> floating
AnyFloatingArray: _Type = _Any4[ND, np.floating[Any], float, _ct.Floating]
# generic :> number :> inexact :> floating :> *
AnyFloat16Array: _Type = _AnyNP[ND, np.float16] | _ct.Float16
AnyFloat32Array: _Type = _AnyNP[ND, np.float32] | _ct.Float32
AnyFloat64Array: _Type = _AnyNP[ND, np.float64] | _ct.Float64
AnyHalfArray: _Type = _AnyNP[ND, np.half] | _ct.Half
AnySingleArray: _Type = _AnyNP[ND, np.single] | _ct.Single
AnyDoubleArray: _Type = _AnyNP[ND, np.double] | _ct.Double
AnyLongDoubleArray: _Type = _AnyNP[ND, np.longdouble] | _ct.LongDouble

# generic :> number :> inexact :> complexfloating
AnyComplexFloatingArray: _Type = _Any4[
    ND,
    np.complexfloating[Any, Any],
    complex,
    _ct.ComplexFloating,
]
# generic :> number :> inexact :> complexfloating :> *
AnyComplex64Array: _Type = _AnyNP[ND, np.complex64]
AnyComplex128Array: _Type = _Any3[ND, np.complex128, complex]
AnyCSingleArray: _Type = _AnyNP[ND, np.csingle]
AnyCDoubleArray: _Type = _Any3[ND, np.cdouble, complex]
AnyCLongDoubleArray: _Type = _AnyNP[ND, np.clongdouble]

# generic :> flexible :> character
AnyCharacterArray: _Type = (
    _AnyNP[ND, np.character]
    | _AnyPY[str]
    | _AnyPY[bytes]
    | _ct.Character
)
# generic :> flexible :> character :> bytes_
AnyBytesArray: _Type = _Any4[ND, np.bytes_, bytes, _ct.Bytes]
# generic :> flexible :> character :> str_
AnyStrArray: _Type = _Any3[ND, np.str_, str]

# generic :> flexible
AnyFlexibleArray: _Type = (
    _AnyNP[ND, np.flexible]
    | _AnyPY[str]
    | _AnyPY[bytes]
    | _ct.Flexible
)
# generic :> flexible :> void
AnyVoidArray: _Type = _AnyNP[ND, np.void]

# generic :> bool
AnyBoolArray: _Type = _Any4[ND, _x.Bool, bool, _ct.Bool]
# generic :> datetime64
AnyDateTime64Array: _Type = _AnyNP[ND, np.datetime64]
# generic :> timedelta64
AnyTimeDelta64Array: _Type = _AnyNP[ND, np.timedelta64]
# generic :> object_
AnyObjectArray: _Type = _Any4[ND, np.object_, object, _ct.Object]

# generic :> {StringDType.type}
if _x.NP2 and not _x.NP20:
    # `numpy>=2.1`
    AnyStringArray: _Type = np.ndarray[ND, np.dtypes.StringDType]
elif _x.NP2:
    # `numpy>=2.0,<2.1`
    AnyStringArray: _Type = np.ndarray[ND, np.dtype[Never]]
else:
    # `numpy<2.0`
    AnyStringArray: _Type = Never
