# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from typing import Any, TypeAlias, final

import numpy as np

from ._array import CanArray
from ._ctypes import (
    Bool as _BoolCT,
    Bytes as _BytesCT,
    Character as _CharacterCT,
    ComplexFloating as _ComplexFloatingCT,
    Flexible as _FlexibleCT,
    Floating as _FloatingCT,
    Generic as _GenericCT,
    Inexact as _InexactCT,
    Integer as _IntegerCT,
    Number as _NumberCT,
    Object as _ObjectCT,
    SignedInteger as _SignedIntegerCT,
    UnsignedInteger as _UnsignedIntegerCT,
)


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


_ND__AnyNPArray = TypeVar(
    '_ND__AnyNPArray',
    bound=tuple[int, ...],
    default=tuple[int, ...],
)
_ST__AnyNPArray = TypeVar(
    '_ST__AnyNPArray',
    bound=np.generic,
    default=np.generic,
)
_AnyNPArray: TypeAlias = (
    CanArray[_ND__AnyNPArray, _ST__AnyNPArray]
    | _PyArray[CanArray[Any, _ST__AnyNPArray]]
)

_ND_AnyArray = TypeVar(
    '_ND_AnyArray',
    bound=tuple[int, ...],
    default=tuple[int, ...],
)
_ST_AnyArray = TypeVar(
    '_ST_AnyArray',
    bound=np.generic,
    default=np.generic,
)
_PT_AnyArray = TypeVar(
    '_PT_AnyArray',
    bound=complex | str | bytes | object,
    default=complex | str | bytes,
)
_CT_AnyArray = TypeVar(
    '_CT_AnyArray',
    bound=_GenericCT,
    default=_GenericCT,
)
AnyArray: TypeAlias = (
    CanArray[_ND_AnyArray, _ST_AnyArray]
    | _PT_AnyArray
    | _CT_AnyArray  # ctypes can only be used to create 0d arrays
    | _PyArray[CanArray[Any, _ST_AnyArray]]
    | _PyArray[_PT_AnyArray]
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
    _UnsignedIntegerCT,
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
    _SignedIntegerCT,
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
    _IntegerCT,
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
    _FloatingCT,
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
    _ComplexFloatingCT,
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
    _InexactCT,
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
    _NumberCT,
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
AnyStrArray: TypeAlias = AnyArray[_ND_AnyArray, _ST_AnyStrArray, str, Never]
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
    _BytesCT,
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
    _CharacterCT,
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
    _FlexibleCT,
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
    _BoolCT,
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
    _ObjectCT,
]
