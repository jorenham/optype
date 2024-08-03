from __future__ import annotations

import datetime as _dt
from typing import Any, Final, TypeAlias as _Type

import numpy as np

from . import _compat as _x, _ctype as _ct


_NP_V2: Final[bool] = np.__version__.startswith('2.')


# integer - unsigned
AnyUInt8: _Type = np.uint8 | _ct.UInt8
AnyUInt16: _Type = np.uint16 | _ct.UInt16
AnyUInt32: _Type = np.uint32 | _ct.UInt32
AnyUInt64: _Type = np.uint64 | _ct.UInt64
AnyUIntP: _Type = np.uintp | _ct.UIntP
AnyUByte: _Type = np.ubyte | _ct.UByte
AnyUShort: _Type = np.ushort | _ct.UShort
AnyUIntC: _Type = np.uintc | _ct.UIntC
AnyULong: _Type = _x.ULong | _ct.ULong
AnyULongLong: _Type = np.ulonglong | _ct.ULongLong
AnyUnsignedInteger: _Type = np.unsignedinteger[Any] | _ct.UnsignedInteger


# integer - signed
AnyInt8: _Type = np.int8 | _ct.Int8
AnyInt16: _Type = np.int16 | _ct.Int16
AnyInt32: _Type = np.int32 | _ct.Int32
AnyInt64: _Type = np.int64 | _ct.Int64
if _NP_V2:
    AnyIntP: _Type = int | np.intp | _ct.IntP
else:
    AnyIntP: _Type = np.intp | _ct.IntP
AnyByte: _Type = np.byte | _ct.Byte
AnyShort: _Type = np.short | _ct.Short
AnyIntC: _Type = np.intc | _ct.IntC
if _NP_V2:
    AnyLong: _Type = _x.Long | _ct.Long
else:
    AnyLong: _Type = int | _x.Long | _ct.Long
AnyLongLong: _Type = np.longlong | _ct.LongLong
AnySignedInteger: _Type = int | np.signedinteger[Any] | _ct.SignedInteger

# integer
AnyInteger: _Type = int | np.integer[Any] | _ct.Integer

# floating point - real
AnyFloat16: _Type = np.float16
AnyFloat32: _Type = np.float32 | _ct.Float32
AnyFloat64: _Type = float | np.float64 | _ct.Float64
AnyHalf: _Type = np.half
AnySingle: _Type = np.single | _ct.Single
AnyDouble: _Type = float | np.double | _ct.Double
AnyLongDouble: _Type = np.longdouble  # | _ct.LongDouble
AnyFloating: _Type = float | np.floating[Any] | _ct.Floating

# floating point - complex
AnyComplex64: _Type = np.complex64
AnyComplex128: _Type = complex | np.complex128
AnyCSingle: _Type = np.csingle
AnyCDouble: _Type = complex | np.cdouble
# either `np.complex192` or `np.complex256`
AnyCLongDouble: _Type = np.clongdouble
AnyComplexFloating: _Type = complex | np.complexfloating[Any, Any]

# floating point
_InexactPY: _Type = complex  # includes `float`, but also `int` and `bool` :/
AnyInexact: _Type = _InexactPY | np.inexact[Any] | _ct.Inexact

# numeric
_NumberPY: _Type = complex  # includes `float` and `int`, but also `bool` :/
AnyNumber: _Type = _NumberPY | np.number[Any] | _ct.Number

# temporal
# (The `datetime.{datetime,timedelta}` values can only be passed to
# `np.{datetime64,timedelta64}` directly; the sctypes won't be correctly
# inferred if passed as `np.array(...[, dtype=None])`.)
AnyDateTime64: _Type = np.datetime64 | _dt.datetime
AnyTimeDelta64: _Type = np.timedelta64 | _dt.timedelta

# variable-width - character
AnyStr: _Type = str | np.str_
AnyBytes: _Type = bytes | np.bytes_ | _ct.Bytes
_CharacterPY: _Type = str | bytes
AnyCharacter: _Type = _CharacterPY | np.character | _ct.Character

# variable-width
AnyVoid: _Type = np.void  # maybe be structured or unstructured
_FlexiblePY: _Type = _CharacterPY
AnyFlexible: _Type = _FlexiblePY | np.flexible | _ct.Flexible

# other types
AnyBool: _Type = bool | _x.Bool | _ct.Bool
AnyObject: _Type = np.object_ | _ct.Object | object

# generic
_GenericPY: _Type = complex | str | bytes  # `complex <: float <: int <: bool`
AnyGeneric: _Type = np.generic | _GenericPY | _ct.Generic | object
