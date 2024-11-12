# mypy: disable-error-code="no-any-explicit"
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

import optype.numpy._compat as _x
import optype.numpy._scalar as _sc
from optype._core._can import CanBuffer
from optype._core._utils import set_module


if sys.version_info >= (3, 13):
    from typing import Never, Protocol, TypeAliasType, TypeVar
else:
    from typing_extensions import Never, Protocol, TypeAliasType, TypeVar

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


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_ST = TypeVar("_ST", bound=np.generic, default=np.generic)
_ST_co = TypeVar("_ST_co", covariant=True, bound=np.generic)
_VT = TypeVar("_VT", default=_ST)


# NOTE: Does not include scalar types
class _AnyArrayNP(Protocol[_ST_co]):
    def __len__(self, /) -> int: ...
    def __array__(self, /) -> np.ndarray[tuple[int, ...], np.dtype[_ST_co]]: ...


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

AnyArray: TypeAlias = _AnyArray[_ST, object] | CanBuffer

_ST_iufc = TypeVar("_ST_iufc", bound=_sc.number, default=_sc.number)
# TODO: also allow `Just[int] | Just[float] | Just[complex]` values
AnyNumberArray: TypeAlias = _AnyArray[_ST_iufc]
# TODO: also allow `Just[int]` values
AnyIntegerArray: TypeAlias = _AnyArray[_sc.integer]
# TODO: also allow `Just[int]` values
AnySignedIntegerArray: TypeAlias = _AnyArray[_sc.sinteger]
AnyUnsignedIntegerArray: TypeAlias = _AnyArray[_sc.uinteger]
# TODO: also allow `Just[float] | Just[complex]` values
AnyInexactArray: TypeAlias = _AnyArray[_sc.inexact]

AnyBoolArray: TypeAlias = _AnyArray[_x.Bool, bool | _x.Bool]

AnyUInt8Array: TypeAlias = _AnyArray[np.uint8, np.uint8 | CanBuffer] | CanBuffer
AnyUByteArray = AnyUInt8Array
AnyUInt16Array: TypeAlias = _AnyArray[np.uint16]
AnyUShortArray = AnyUInt16Array
AnyUInt32Array: TypeAlias = _AnyArray[np.uint32]
AnyUInt64Array: TypeAlias = _AnyArray[np.uint64]
AnyUIntCArray: TypeAlias = _AnyArray[np.uintc]
AnyUIntPArray: TypeAlias = _AnyArray[np.uintp]
AnyULongArray: TypeAlias = _AnyArray[_x.ULong]
AnyULongLongArray: TypeAlias = _AnyArray[np.ulonglong]

AnyInt8Array: TypeAlias = _AnyArray[np.int8]
AnyByteArray = AnyInt8Array
AnyInt16Array: TypeAlias = _AnyArray[np.int16]
AnyShortArray = AnyInt16Array
AnyInt32Array: TypeAlias = _AnyArray[np.int32]
AnyInt64Array: TypeAlias = _AnyArray[np.int64]
AnyIntCArray: TypeAlias = _AnyArray[np.intc]
# TODO: also allow `Just[int]` values
AnyIntPArray: TypeAlias = _AnyArray[np.intp]  # no int (numpy>=2)
AnyLongArray: TypeAlias = _AnyArray[_x.Long]  # no int (numpy<=1)
AnyLongLongArray: TypeAlias = _AnyArray[np.longlong]

# TODO: also allow `Just[float]` values
AnyFloatingArray: TypeAlias = _AnyArray[_sc.floating]
AnyFloat16Array: TypeAlias = _AnyArray[np.float16]
AnyFloat32Array: TypeAlias = _AnyArray[np.float32]
AnyFloat64Array: TypeAlias = _AnyArray[np.float64]
AnyLongDoubleArray: TypeAlias = _AnyArray[np.longdouble]

# TODO: also allow `Just[complex]` values
AnyComplexFloatingArray: TypeAlias = _AnyArray[_sc.cfloating]
AnyComplex64Array: TypeAlias = _AnyArray[np.complex64]
AnyComplex128Array: TypeAlias = _AnyArray[np.complex128]  # no `complex`
AnyCLongDoubleArray: TypeAlias = _AnyArray[np.clongdouble]

AnyCharacterArray: TypeAlias = _AnyArray[np.character, bytes | str | np.character]
AnyBytesArray: TypeAlias = _AnyArray[np.bytes_, bytes | np.bytes_]
AnyStrArray: TypeAlias = _AnyArray[np.str_, str | np.str_]

AnyFlexibleArray: TypeAlias = _AnyArray[np.flexible, bytes | str | np.flexible]
AnyVoidArray: TypeAlias = _AnyArray[np.void]

AnyDateTime64Array: TypeAlias = _AnyArray[np.datetime64]
AnyTimeDelta64Array: TypeAlias = _AnyArray[np.timedelta64]
# NOTE: `{everything} <: object`, so it can't be included here
AnyObjectArray: TypeAlias = _AnyArray[np.object_]


if _x.NP2:

    @set_module("optype.numpy")
    class AnyStringArray(Protocol):
        def __len__(self, /) -> int: ...

        if _x.NP2 and not _x.NP20:
            # `numpy>=2.1`
            def __array__(
                self, /
            ) -> np.ndarray[tuple[int, ...], np.dtypes.StringDType]: ...

        elif _x.NP2:
            # `numpy>=2,<2.1`
            def __array__(self, /) -> np.ndarray[tuple[int, ...], np.dtype[Never]]: ...

else:  # `numpy<2`
    AnyStringArray: TypeAlias = Never
