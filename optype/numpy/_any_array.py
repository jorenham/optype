# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from typing import Any, TypeAlias, final

import numpy as np

from ._any_scalar import (
    _AnyBoolCT,
    _AnyBytesCT,
    _AnyCharacterCT,
    _AnyFlexibleCT,
    _AnyFloatingCT,
    _AnyGenericCT,
    _AnyInexactCT,
    _AnyIntegerCT,
    _AnyNumberCT,
    _AnyObjectCT,
    _AnySignedIntegerCT,
    _AnyUnsignedIntegerCT,
)
from ._array import CanArray


if sys.version_info >= (3, 13):
    from typing import Never, Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import Never, Protocol, TypeVar, runtime_checkable


__all__ = (
    'AnyArray',
    'AnyArrayBool',
    'AnyArrayBytes',
    'AnyArrayCharacter',
    'AnyArrayComplexFloating',
    'AnyArrayDateTime',
    'AnyArrayFlexible',
    'AnyArrayFloating',
    'AnyArrayInexact',
    'AnyArrayInteger',
    'AnyArrayNumber',
    'AnyArrayObject',
    'AnyArraySignedInteger',
    'AnyArrayStr',
    'AnyArrayTimeDelta',
    'AnyArrayUnsignedInteger',
)


_V_PyArray = TypeVar('_V_PyArray', infer_variance=True)


@final
@runtime_checkable
class _PyArray(Protocol[_V_PyArray]):
    def __len__(self, /) -> int: ...
    def __getitem__(self, i: int, /) -> _V_PyArray | _PyArray[_V_PyArray]: ...


_ND_AnyArray = TypeVar(
    '_ND_AnyArray',
    bound=tuple[int, ...],
    default=tuple[int, ...],
)
_ST_AnyArray = TypeVar('_ST_AnyArray', bound=np.generic, default=Any)
_PT_AnyArray = TypeVar(
    '_PT_AnyArray',
    bound=complex | str | bytes | object,
    default=complex | str | bytes,
)
_CT_AnyArray = TypeVar(
    '_CT_AnyArray',
    bound=_AnyGenericCT,
    default=_AnyGenericCT,
)
_AnyArrayNP: TypeAlias = (
    CanArray[_ND_AnyArray, _ST_AnyArray]
    | _PyArray[CanArray[Any, _ST_AnyArray]]
)
AnyArray: TypeAlias = (
    CanArray[_ND_AnyArray, _ST_AnyArray]
    | _PyArray[CanArray[Any, _ST_AnyArray]]
    | _PT_AnyArray
    | _PyArray[_PT_AnyArray]
    | _CT_AnyArray  # ctypes can only be used to create 0d arrays
)
"""
Generic array-like that can be passed to e.g. `np.array` or `np.asaray`, with
signature `AnyArray[ND: *int, ST: np.generic, PT: complex | str | bytes]`.
"""


#
# integers
#

# unsignedinteger
_ST_AnyArrayUnsignedInteger = TypeVar(
    '_ST_AnyArrayUnsignedInteger',
    bound=np.unsignedinteger[Any],
    default=np.unsignedinteger[Any],
)
AnyArrayUnsignedInteger: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayUnsignedInteger,
    Never,
    _AnyUnsignedIntegerCT,
]

# signedinteger
_ST_AnyArraySignedInteger = TypeVar(
    '_ST_AnyArraySignedInteger',
    bound=np.signedinteger[Any],
    default=np.signedinteger[Any],
)
AnyArraySignedInteger: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArraySignedInteger,
    int,
    _AnySignedIntegerCT,
]

# integer
_ST_AnyArrayInteger = TypeVar(
    '_ST_AnyArrayInteger',
    bound=np.integer[Any],
    default=np.integer[Any],
)
AnyArrayInteger: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayInteger,
    int,
    _AnyIntegerCT,
]


#
# floating point
#

# floating
_ST_AnyArrayFloating = TypeVar(
    '_ST_AnyArrayFloating',
    bound=np.floating[Any],
    default=np.floating[Any],
)
AnyArrayFloating: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayFloating,
    float,
    _AnyFloatingCT,
]

# complexfloating
_ST_AnyArrayComplexFloating = TypeVar(
    '_ST_AnyArrayComplexFloating',
    bound=np.complexfloating[Any, Any],
    default=np.complexfloating[Any, Any],
)
AnyArrayComplexFloating: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayComplexFloating,
    complex,
    Never,
]


# inexact
_ST_AnyArrayInexact = TypeVar(
    '_ST_AnyArrayInexact',
    bound=np.inexact[Any],
    default=np.inexact[Any],
)
AnyArrayInexact: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayInexact,
    float | complex,
    _AnyInexactCT,
]


#
# integers | floats
#

# number
_ST_AnyArrayNumber = TypeVar(
    '_ST_AnyArrayNumber',
    bound=np.number[Any],
    default=np.number[Any],
)
AnyArrayNumber: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayNumber,
    int | float | complex,
    _AnyNumberCT,
]


#
# temporal
#

# datetime64
_ST_AnyArrayDateTime64 = TypeVar(
    '_ST_AnyArrayDateTime64',
    bound=np.datetime64,
    default=np.datetime64,
)
AnyArrayDateTime: TypeAlias = _AnyArrayNP[_ND_AnyArray, _ST_AnyArrayDateTime64]

# timedelta64
_ST_AnyArrayTimeDelta = TypeVar(
    '_ST_AnyArrayTimeDelta',
    bound=np.timedelta64,
    default=np.timedelta64,
)
AnyArrayTimeDelta: TypeAlias = _AnyArrayNP[_ND_AnyArray, _ST_AnyArrayTimeDelta]


#
# character strings
#

# str_
_ST_AnyArrayStr = TypeVar('_ST_AnyArrayStr', bound=np.str_, default=np.str_)
AnyArrayStr: TypeAlias = AnyArray[_ND_AnyArray, _ST_AnyArrayStr, str]
"""This is about `numpy.dtypes.StrDType`; not `numpy.dtypes.StringDType`."""

# bytes_
_ST_AnyArrayBytes = TypeVar(
    '_ST_AnyArrayBytes',
    bound=np.bytes_,
    default=np.bytes_,
)
AnyArrayBytes: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayBytes,
    bytes,
    _AnyBytesCT,
]

# string
# TODO(jorenham): Add `AnyArrayString` with `np.dtypes.StringDType` (np2 only)
# https://github.com/jorenham/optype/issues/99


# character
_ST_AnyArrayCharacter = TypeVar(
    '_ST_AnyArrayCharacter',
    bound=np.character,
    default=np.character,
)
AnyArrayCharacter: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayCharacter,
    str | bytes,
    _AnyCharacterCT,
]

# void
_ST_AnyArrayVoid = TypeVar('_ST_AnyArrayVoid', bound=np.void, default=np.void)
AnyArrayVoid: TypeAlias = _AnyArrayNP[_ND_AnyArray, _ST_AnyArrayVoid]

# flexible
_ST_AnyArrayFlexible = TypeVar(
    '_ST_AnyArrayFlexible',
    bound=np.flexible,
    default=np.flexible,
)
AnyArrayFlexible: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayFlexible,
    str | bytes,
    _AnyFlexibleCT,
]


#
# other types
#

# bool_
_ST_AnyArrayBool = TypeVar(
    '_ST_AnyArrayBool',
    bound=np.bool_,
    default=np.bool_,
)
AnyArrayBool: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayBool,
    bool,
    _AnyBoolCT,
]

# object_
_ST_AnyArrayObject = TypeVar(
    '_ST_AnyArrayObject',
    bound=np.object_,
    default=np.object_,
)
AnyArrayObject: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyArrayObject,
    object,
    _AnyObjectCT,
]
