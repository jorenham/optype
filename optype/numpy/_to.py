# mypy: disable-error-code="no-any-explicit"

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
from typing_extensions import TypeAliasType


__all__ = [
    "ToBool",
    "ToBool1D",
    "ToBool2D",
    "ToBoolND",
    "ToComplex",
    "ToComplex1D",
    "ToComplex2D",
    "ToComplexND",
    "ToFloat",
    "ToFloat1D",
    "ToFloat2D",
    "ToFloatND",
    "ToInt",
    "ToInt1D",
    "ToInt2D",
    "ToIntND",
    "ToMatrix",
    "ToScalar",
    "ToTensor",
    "ToVector",
]

_Integer: TypeAlias = np.integer[Any]
_Floating: TypeAlias = np.floating[Any]


_T = TypeVar("_T")
_ST = TypeVar("_ST", bound=np.generic)
_ST_co = TypeVar("_ST_co", bound=np.generic, covariant=True)


class _CanNDArray(Protocol[_ST_co]):
    """
    Similar to `optype.numpy.CanArray`, but must be sized (i.e. excludes scalars),
    and is parameterized by only the scalar type (instead of the shape and dtype).
    """

    def __len__(self, /) -> int: ...
    def __array__(self, /) -> np.ndarray[tuple[int, ...], np.dtype[_ST_co]]: ...


_To1D: TypeAlias = _CanNDArray[_ST] | Sequence[_T | _ST]
_To2D: TypeAlias = _CanNDArray[_ST] | Sequence[_To1D[_ST, _T]]

ToScalar: TypeAlias = complex | bytes | str | np.generic
ToVector: TypeAlias = _To1D[np.generic, complex | bytes | str]
ToMatrix: TypeAlias = _To2D[np.generic, complex | bytes | str]
ToTensor: TypeAlias = _To1D[np.generic, complex | bytes | str] | Sequence["ToTensor"]

ToBool: TypeAlias = bool | np.bool_
ToBool1D: TypeAlias = _To1D[np.bool_, bool]
ToBool2D: TypeAlias = _To2D[np.bool_, bool]
ToBoolND: TypeAlias = _To1D[np.bool_, bool] | Sequence["ToBoolND"]

_i_co = TypeAliasType("_i_co", _Integer | np.bool_)
ToInt: TypeAlias = int | _i_co
ToInt1D: TypeAlias = _To1D[_i_co, int]
ToInt2D: TypeAlias = _To2D[_i_co, int]
ToIntND: TypeAlias = _To1D[_i_co, int] | Sequence["ToIntND"]

_f_co = TypeAliasType("_f_co", _Floating | _Integer | np.bool_)
ToFloat: TypeAlias = float | _f_co
ToFloat1D: TypeAlias = _To1D[_f_co, float]
ToFloat2D: TypeAlias = _To2D[_f_co, float]
ToFloatND: TypeAlias = _To1D[_f_co, float] | Sequence["ToFloatND"]

_c_co = TypeAliasType("_c_co", np.number[Any] | np.bool_)
ToComplex: TypeAlias = complex | _c_co
ToComplex1D: TypeAlias = _To1D[_c_co, complex]
ToComplex2D: TypeAlias = _To2D[_c_co, complex]
ToComplexND: TypeAlias = _To1D[_c_co, complex] | Sequence["ToComplexND"]
