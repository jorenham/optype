# mypy: disable-error-code="no-any-explicit"
# pyright: reportExplicitAny=false

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np

import optype.numpy._compat as _x
import optype.numpy._scalar as _sc
import optype.typing as opt
from optype._core._can import CanBuffer
from optype._core._utils import set_module


if sys.version_info >= (3, 13):
    from typing import Never, TypeAliasType, TypeVar
else:
    from typing_extensions import Never, TypeAliasType, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator


# ruff: noqa: RUF022
__all__ = [
    "AnyArray",
    "AnyNumberArray",
    "AnyIntegerArray",
    "AnyUnsignedIntegerArray",
    "AnySignedIntegerArray",
    "AnyInexactArray",
    "AnyFloatingArray",
    "AnyComplexFloatingArray",
    "AnyFlexibleArray",
    "AnyCharacterArray",

    "AnyBoolArray",

    "AnyUIntArray", "AnyIntArray",
    "AnyUInt8Array", "AnyInt8Array",
    "AnyUInt8Array", "AnyInt8Array",
    "AnyUInt16Array", "AnyInt16Array",
    "AnyUInt32Array", "AnyInt32Array",
    "AnyUInt64Array", "AnyInt64Array",
    "AnyUByteArray", "AnyByteArray",
    "AnyUShortArray", "AnyShortArray",
    "AnyUIntCArray", "AnyIntCArray",
    "AnyUIntPArray", "AnyIntPArray",
    "AnyULongArray", "AnyLongArray",
    "AnyULongLongArray", "AnyLongLongArray",

    "AnyFloat16Array",
    "AnyFloat32Array", "AnyComplex64Array",
    "AnyFloat64Array", "AnyComplex128Array",
    "AnyLongDoubleArray", "AnyCLongDoubleArray",

    "AnyDateTime64Array",
    "AnyTimeDelta64Array",

    "AnyBytesArray",
    "AnyStrArray",
    "AnyVoidArray",
    "AnyObjectArray",
    "AnyStringArray",
]  # fmt: skip


_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")
_ST_co = TypeVar("_ST_co", bound=np.generic, covariant=True)
_ST = TypeVar("_ST", bound=np.generic, default=np.generic)
_VT = TypeVar("_VT", default=_ST)


# NOTE: Does not include scalar types
class _AnyArrayNP(Protocol[_ST_co]):
    def __len__(self, /) -> int: ...

    if _x.NP21:

        def __array__(self, /) -> np.ndarray[tuple[int, ...], np.dtype[_ST_co]]: ...

    else:

        def __array__(self, /) -> np.ndarray[Any, np.dtype[_ST_co]]: ...


# NOTE: does not include tuple
class _AnyArrayPY0(Protocol[_T_co]):
    def __len__(self, /) -> int: ...
    def __getitem__(self, i: int, /) -> _T_co | _AnyArrayPY0[_T_co]: ...
    def __reversed__(self, /) -> Iterator[_T_co | _AnyArrayPY0[_T_co]]: ...
    def index(self, x: Any, /) -> int: ...  # pyright: ignore[reportAny]


_AnyArrayPY: TypeAlias = tuple[_T, ...] | _AnyArrayPY0[_T]
_AnyArray = TypeAliasType(
    "_AnyArray",
    _AnyArrayNP[_ST] | _AnyArrayPY[_VT] | _AnyArrayPY[_AnyArrayNP[_ST]],
    type_params=(_ST, _VT),
)

_JustFloat: TypeAlias = opt.Just[float]
_JustComplex: TypeAlias = opt.Just[complex]

###

AnyArray: TypeAlias = _AnyArray[_ST, object] | CanBuffer

AnyNumberArray: TypeAlias = _AnyArray[
    _sc.number,
    _sc.number | opt.JustInt | _JustFloat | _JustComplex,
]
AnyIntegerArray: TypeAlias = _AnyArray[_sc.integer, _sc.integer | opt.JustInt]
AnySignedIntegerArray: TypeAlias = _AnyArray[_sc.sinteger, _sc.sinteger | opt.JustInt]
AnyUnsignedIntegerArray: TypeAlias = _AnyArray[_sc.uinteger]
AnyInexactArray: TypeAlias = _AnyArray[
    _sc.inexact,
    _sc.inexact | _JustFloat | _JustComplex,
]

AnyBoolArray: TypeAlias = _AnyArray[_x.Bool, _x.Bool | bool]

AnyUInt8Array: TypeAlias = _AnyArray[np.uint8, np.uint8 | CanBuffer] | CanBuffer
AnyUByteArray = AnyUInt8Array
AnyUInt16Array: TypeAlias = _AnyArray[np.uint16]
AnyUShortArray = AnyUInt16Array
AnyUInt32Array: TypeAlias = _AnyArray[np.uint32]
AnyUInt64Array: TypeAlias = _AnyArray[np.uint64]
AnyUIntCArray: TypeAlias = _AnyArray[np.uintc]
AnyULongLongArray: TypeAlias = _AnyArray[np.ulonglong]
AnyULongArray: TypeAlias = _AnyArray[_x.ULong]
AnyUIntPArray: TypeAlias = _AnyArray[np.uintp]
AnyUIntArray: TypeAlias = _AnyArray[np.uint]

AnyInt8Array: TypeAlias = _AnyArray[np.int8]
AnyByteArray = AnyInt8Array
AnyInt16Array: TypeAlias = _AnyArray[np.int16]
AnyShortArray = AnyInt16Array
AnyInt32Array: TypeAlias = _AnyArray[np.int32]
AnyInt64Array: TypeAlias = _AnyArray[np.int64]
AnyIntCArray: TypeAlias = _AnyArray[np.intc]
AnyLongLongArray: TypeAlias = _AnyArray[np.longlong]
AnyLongArray: TypeAlias = _AnyArray[_x.Long]  # no int (numpy<=1)
AnyIntPArray: TypeAlias = _AnyArray[np.intp]  # no int (numpy>=2)
AnyIntArray: TypeAlias = _AnyArray[np.int_, np.int_ | opt.JustInt]

AnyFloatingArray: TypeAlias = _AnyArray[_sc.floating, _sc.floating | _JustFloat]
AnyFloat16Array: TypeAlias = _AnyArray[np.float16]
AnyFloat32Array: TypeAlias = _AnyArray[np.float32]
AnyFloat64Array: TypeAlias = _AnyArray[np.float64, np.float64 | _JustFloat]
AnyLongDoubleArray: TypeAlias = _AnyArray[np.longdouble]

AnyComplexFloatingArray: TypeAlias = _AnyArray[
    _sc.cfloating,
    _sc.cfloating | _JustComplex,
]
AnyComplex64Array: TypeAlias = _AnyArray[np.complex64]
AnyComplex128Array: TypeAlias = _AnyArray[np.complex128, np.complex128 | _JustComplex]
AnyCLongDoubleArray: TypeAlias = _AnyArray[np.clongdouble]

AnyCharacterArray: TypeAlias = _AnyArray[np.character, np.character | bytes | str]
AnyBytesArray: TypeAlias = _AnyArray[np.bytes_, np.bytes_ | bytes]
AnyStrArray: TypeAlias = _AnyArray[np.str_, np.str_ | str]

AnyFlexibleArray: TypeAlias = _AnyArray[np.flexible, np.flexible | bytes | str]
AnyVoidArray: TypeAlias = _AnyArray[np.void]  # TODO: structural types

AnyDateTime64Array: TypeAlias = _AnyArray[np.datetime64]
AnyTimeDelta64Array: TypeAlias = _AnyArray[np.timedelta64]
AnyObjectArray: TypeAlias = _AnyArray[np.object_, np.object_ | opt.Just[object]]


if _x.NP20:

    @set_module("optype.numpy")
    class AnyStringArray(Protocol):
        def __len__(self, /) -> int: ...

        if _x.NP21:
            # `numpy>=2.1`
            def __array__(
                self, /
            ) -> np.ndarray[tuple[int, ...], np.dtypes.StringDType]: ...

        elif _x.NP20:
            # `numpy>=2,<2.1`
            def __array__(self, /) -> np.ndarray[Any, np.dtype[Never]]: ...

else:  # `numpy<2`
    AnyStringArray: TypeAlias = Never
