from __future__ import annotations

import sys
from typing import TypeAlias as Alias

import numpy as np

import optype as opt
import optype.numpy._array as _a
import optype.numpy._compat as _x
import optype.numpy._scalar as _sc
import optype.numpy.ctypeslib as _ct


if sys.version_info >= (3, 13):
    from typing import Never, Protocol, TypeVar
else:
    from typing_extensions import Never, Protocol, TypeVar


# ruff: noqa: RUF022
__all__ = [
    'AnyArray',
    'AnyNumberArray',
    'AnyIntegerArray',
    'AnyUnsignedIntegerArray',
    'AnySignedIntegerArray',
    'AnyInexactArray',
    'AnyFloatingArray',
    'AnyComplexFloatingArray',
    'AnyFlexibleArray',
    'AnyCharacterArray',

    'AnyBoolArray',

    'AnyUInt8Array', 'AnyInt8Array',
    'AnyUInt8Array', 'AnyInt8Array',
    'AnyUInt16Array', 'AnyInt16Array',
    'AnyUInt32Array', 'AnyInt32Array',
    'AnyUInt64Array', 'AnyInt64Array',
    'AnyUByteArray', 'AnyByteArray',
    'AnyUShortArray', 'AnyShortArray',
    'AnyUIntCArray', 'AnyIntCArray',
    'AnyUIntPArray', 'AnyIntPArray',
    'AnyULongArray', 'AnyLongArray',
    'AnyULongLongArray', 'AnyLongLongArray',

    'AnyFloat16Array',
    'AnyFloat32Array', 'AnyComplex64Array',
    'AnyFloat64Array', 'AnyComplex128Array',
    'AnyLongDoubleArray', 'AnyCLongDoubleArray',

    'AnyDateTime64Array',
    'AnyTimeDelta64Array',

    'AnyBytesArray',
    'AnyStrArray',
    'AnyVoidArray',
    'AnyObjectArray',

    'AnyStringArray',
]  # fmt: skip


_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)


class _PyArrray(Protocol[_T_co]):
    def __len__(self, /) -> int: ...
    def __getitem__(self, i: int, /) -> _T_co | _PyArrray[_T_co]: ...


_AnyPyArray: Alias = _T | _PyArrray[_T]
_PyChar: Alias = bytes | str
_PyGeneric: Alias = int | float | complex | _PyChar

_T_np = TypeVar('_T_np', bound=np.generic)
_T_ct = TypeVar('_T_ct', bound=_ct.CType)
_T_py = TypeVar('_T_py', bound=_PyGeneric | opt.CanBuffer)

_Any1: Alias = _AnyPyArray[_a.CanArray[tuple[int, ...], np.dtype[_T_np]]]
_Any2: Alias = _Any1[_T_np] | _AnyPyArray[_T_ct] | _ct.Array[_T_ct]
_Any3: Alias = _Any2[_T_np, _T_ct] | _AnyPyArray[_T_py]


AnyArray: Alias = _Any3[np.generic, _ct.Generic, _PyGeneric]

ST_iufc = TypeVar('ST_iufc', bound=_sc.Number, default=_sc.Number)
# NOTE: `builtins.bool <: int`, so `int` can't be included here
AnyNumberArray: Alias = _Any2[ST_iufc, _ct.Number]
AnyIntegerArray: Alias = _Any2[_sc.Integer, _ct.Integer]
AnyUnsignedIntegerArray: Alias = _Any2[_sc.UnsignedInteger, _ct.UnsignedInteger]
AnySignedIntegerArray: Alias = _Any2[_sc.SignedInteger, _ct.SignedInteger]
AnyInexactArray: Alias = _Any2[_sc.Inexact, _ct.Floating]

AnyBoolArray: Alias = _Any3[_x.Bool, _ct.Bool, bool]

# NOTE: This requires enabling PEP 688 semantics in your type-checker:
#   - mypy: see https://github.com/python/mypy/issues/15313
#   - pyright: set `disableBytesTypePromotions = true` (or `strict = true`)
AnyUInt8Array: Alias = _Any3[np.uint8, _ct.UInt8, opt.CanBuffer]
AnyUInt16Array: Alias = _Any2[np.uint16, _ct.UInt16]
AnyUInt32Array: Alias = _Any2[np.uint32, _ct.UInt32]
AnyUInt64Array: Alias = _Any2[np.uint64, _ct.UInt64]
AnyUByteArray: Alias = _Any2[np.ubyte, _ct.UByte]
AnyUShortArray: Alias = _Any2[np.ushort, _ct.UShort]
AnyUIntCArray: Alias = _Any2[np.uintc, _ct.UIntC]
AnyUIntPArray: Alias = _Any2[np.uintp, _ct.UIntP]
AnyULongArray: Alias = _Any2[_x.ULong, _ct.ULong]
AnyULongLongArray: Alias = _Any2[np.ulonglong, _ct.ULongLong]

AnyInt8Array: Alias = _Any2[np.int8, _ct.Int8]
AnyInt16Array: Alias = _Any2[np.int16, _ct.Int16]
AnyInt32Array: Alias = _Any2[np.int32, _ct.Int32]
AnyInt64Array: Alias = _Any2[np.int64, _ct.Int64]
AnyByteArray: Alias = _Any2[np.byte, _ct.Byte]
AnyShortArray: Alias = _Any2[np.short, _ct.Short]
AnyIntCArray: Alias = _Any2[np.intc, _ct.IntC]
# NOTE: `builtins.bool <: int`, so `int` can't be included here
AnyIntPArray: Alias = _Any2[np.intp, _ct.IntP]  # no int (numpy>=2)
AnyLongArray: Alias = _Any2[_x.Long, _ct.Long]  # no int (numpy<=1)
AnyLongLongArray: Alias = _Any2[np.longlong, _ct.LongLong]

# NOTE: `int <: float` (type-check only), so it can't be included here
AnyFloatingArray: Alias = _Any2[_sc.Floating, _ct.Floating]
AnyFloat16Array: Alias = _Any1[np.float16 | np.half]
AnyFloat32Array: Alias = _Any2[np.float32 | np.single, _ct.Float32]
AnyFloat64Array: Alias = _Any2[np.float64 | np.double, _ct.Float64]
AnyLongDoubleArray: Alias = _Any1[np.longdouble]

# NOTE: `float <: complex` (type-check only), so it can't be included here
AnyComplexFloatingArray: Alias = _Any1[_sc.ComplexFloating]
AnyComplex64Array: Alias = _Any1[np.complex64 | np.csingle]
AnyComplex128Array: Alias = _Any1[np.complex128 | np.cdouble]  # no `complex`
AnyCLongDoubleArray: Alias = _Any1[np.clongdouble]

AnyCharacterArray: Alias = _Any3[np.character, _ct.Bytes, _PyChar]
AnyBytesArray: Alias = _Any3[np.bytes_, _ct.Bytes, bytes]
AnyStrArray: Alias = _Any1[np.str_] | _PyArrray[str]

AnyFlexibleArray: Alias = _Any3[np.flexible, _ct.Flexible, _PyChar]
AnyVoidArray: Alias = _Any2[np.void, _ct.Void]

AnyDateTime64Array: Alias = _Any1[np.datetime64]
AnyTimeDelta64Array: Alias = _Any1[np.timedelta64]
# NOTE: `{everything} <: object`, so it can't be included here
AnyObjectArray: Alias = _Any2[np.object_, _ct.Object]

if _x.NP2 and not _x.NP20:  # `numpy>=2.1`
    AnyStringArray: Alias = _a.CanArray[  # type: ignore[type-var]
        tuple[int, ...],
        np.dtypes.StringDType,  # pyright: ignore[reportInvalidTypeArguments]
    ]
elif _x.NP2:  # `numpy>=2,<2.1`
    AnyStringArray: Alias = _a.CanArray[tuple[int, ...], np.dtype[Never]]
else:  # `numpy<2`
    AnyStringArray: Alias = Never
