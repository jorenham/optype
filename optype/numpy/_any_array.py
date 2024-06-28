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
    'AnyBoolArray',
    'AnyBytesArray',
    'AnyCharacterArray',
    'AnyComplexFloatingArray',
    'AnyDateTime64Array',
    'AnyFlexibleArray',
    'AnyFloatingArray',
    'AnyInexactArray',
    'AnyIntegerArray',
    'AnyNumberArray',
    'AnyObjectArray',
    'AnySignedIntegerArray',
    'AnyStrArray',
    'AnyTimeDelta64Array',
    'AnyUnsignedIntegerArray',
    'AnyVoidArray',
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
_AnyNPArray: TypeAlias = (
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
_ST_AnyUnsignedIntegerArray = TypeVar(
    '_ST_AnyUnsignedIntegerArray',
    bound=np.unsignedinteger[Any],
    default=np.unsignedinteger[Any],
)
AnyUnsignedIntegerArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyUnsignedIntegerArray,
    Never,
    _AnyUnsignedIntegerCT,
]

# signedinteger
_ST_AnySignedIntegerArray = TypeVar(
    '_ST_AnySignedIntegerArray',
    bound=np.signedinteger[Any],
    default=np.signedinteger[Any],
)
AnySignedIntegerArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnySignedIntegerArray,
    int,
    _AnySignedIntegerCT,
]

# integer
_ST_AnyIntegerArray = TypeVar(
    '_ST_AnyIntegerArray',
    bound=np.integer[Any],
    default=np.integer[Any],
)
AnyIntegerArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyIntegerArray,
    int,
    _AnyIntegerCT,
]


#
# floating point
#

# floating
_ST_AnyFloatingArray = TypeVar(
    '_ST_AnyFloatingArray',
    bound=np.floating[Any],
    default=np.floating[Any],
)
AnyFloatingArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyFloatingArray,
    float,
    _AnyFloatingCT,
]

# complexfloating
_ST_AnyComplexFloatingArray = TypeVar(
    '_ST_AnyComplexFloatingArray',
    bound=np.complexfloating[Any, Any],
    default=np.complexfloating[Any, Any],
)
AnyComplexFloatingArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyComplexFloatingArray,
    complex,
    Never,
]


# inexact
_ST_AnyInexactArray = TypeVar(
    '_ST_AnyInexactArray',
    bound=np.inexact[Any],
    default=np.inexact[Any],
)
AnyInexactArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyInexactArray,
    float | complex,
    _AnyInexactCT,
]


#
# integers | floats
#

# number
_ST_AnyNumberArray = TypeVar(
    '_ST_AnyNumberArray',
    bound=np.number[Any],
    default=np.number[Any],
)
AnyNumberArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyNumberArray,
    int | float | complex,
    _AnyNumberCT,
]


#
# temporal
#

# datetime64
_ST_AnyDateTime64Array = TypeVar(
    '_ST_AnyDateTime64Array',
    bound=np.datetime64,
    default=np.datetime64,
)
AnyDateTime64Array: TypeAlias = _AnyNPArray[
    _ND_AnyArray,
    _ST_AnyDateTime64Array,
]

# timedelta64
_ST_AnyTimeDelta64Array = TypeVar(
    '_ST_AnyTimeDelta64Array',
    bound=np.timedelta64,
    default=np.timedelta64,
)
AnyTimeDelta64Array: TypeAlias = _AnyNPArray[
    _ND_AnyArray,
    _ST_AnyTimeDelta64Array,
]


#
# character strings
#

# str_
_ST_AnyStrArray = TypeVar('_ST_AnyStrArray', bound=np.str_, default=np.str_)
AnyStrArray: TypeAlias = AnyArray[_ND_AnyArray, _ST_AnyStrArray, str]
"""This is about `numpy.dtypes.StrDType`; not `numpy.dtypes.StringDType`."""

# bytes_
_ST_AnyBytesArray = TypeVar(
    '_ST_AnyBytesArray',
    bound=np.bytes_,
    default=np.bytes_,
)
AnyBytesArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyBytesArray,
    bytes,
    _AnyBytesCT,
]

# string
# TODO(jorenham): Add `AnyStringArray` with `np.dtypes.StringDType` (np2 only)
# https://github.com/jorenham/optype/issues/99


# character
_ST_AnyCharacterArray = TypeVar(
    '_ST_AnyCharacterArray',
    bound=np.character,
    default=np.character,
)
AnyCharacterArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyCharacterArray,
    str | bytes,
    _AnyCharacterCT,
]

# void
_ST_AnyVoidArray = TypeVar('_ST_AnyVoidArray', bound=np.void, default=np.void)
AnyVoidArray: TypeAlias = _AnyNPArray[_ND_AnyArray, _ST_AnyVoidArray]

# flexible
_ST_AnyFlexibleArray = TypeVar(
    '_ST_AnyFlexibleArray',
    bound=np.flexible,
    default=np.flexible,
)
AnyFlexibleArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyFlexibleArray,
    str | bytes,
    _AnyFlexibleCT,
]


#
# other types
#

# bool_
_ST_AnyBoolArray = TypeVar(
    '_ST_AnyBoolArray',
    bound=np.bool_,
    default=np.bool_,
)
AnyBoolArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyBoolArray,
    bool,
    _AnyBoolCT,
]

# object_
_ST_AnyObjectArray = TypeVar(
    '_ST_AnyObjectArray',
    bound=np.object_,
    default=np.object_,
)
AnyObjectArray: TypeAlias = AnyArray[
    _ND_AnyArray,
    _ST_AnyObjectArray,
    object,
    _AnyObjectCT,
]
