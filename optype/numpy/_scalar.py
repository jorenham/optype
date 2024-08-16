from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeVar, overload, override, runtime_checkable
else:
    from typing_extensions import (
        Protocol,
        Self,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )

if TYPE_CHECKING:
    from numpy._core.multiarray import flagsobj
    from typing_extensions import CapsuleType


__all__ = ['Scalar']


_PT_co = TypeVar('_PT_co', covariant=True)
_NB_co = TypeVar('_NB_co', bound=int, covariant=True, default=int)
_DT = TypeVar('_DT', bound=np.dtype[np.generic], default=np.dtype[np.generic])

_L0: TypeAlias = Literal[0]
_L1: TypeAlias = Literal[1]
_Array0D: TypeAlias = np.ndarray[tuple[()], _DT]


@runtime_checkable
class Scalar(Protocol[_PT_co, _NB_co]):
    """
    A lightweight `numpy.generic` interface that's actually generic, and
    doesn't require all that nasty `numpy.typing.NBitBase` stuff.
    """

    def item(self, k: _L0 | tuple[()] | tuple[_L0] = ..., /) -> _PT_co: ...
    # unfortunately `| int` is required for compat with `numpy.__init__.pyi`
    @property
    def itemsize(self, /) -> _NB_co | int: ...

    @property
    def base(self, /) -> None: ...
    @property
    def data(self, /) -> memoryview: ...
    @property
    def dtype(self, /) -> np.dtype[Self]: ...  # type: ignore[type-var]  # pyright: ignore[reportInvalidTypeArguments]
    @property
    def flags(self, /) -> flagsobj: ...
    @property
    def nbytes(self, /) -> int: ...
    @property
    def ndim(self, /) -> _L0: ...
    @property
    def shape(self, /) -> tuple[()]: ...
    @property
    def size(self, /) -> _L1: ...
    @property
    def strides(self, /) -> tuple[()]: ...

    @property
    def __array_priority__(self, /) -> float: ...  # -1000000.0
    @property
    def __array_interface__(self, /) -> dict[str, object]: ...  # TypedDict?
    @property
    def __array_struct__(self, /) -> CapsuleType: ...

    @override
    def __hash__(self, /) -> int: ...
    @override
    def __eq__(self, other: object, /) -> np.bool_: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ne__(self, other: object, /) -> np.bool_: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    def __bool__(self, /) -> bool: ...
    # Unlike `numpy/__init__.pyi` suggests, there exists no `__bytes__` method
    # in `np.generic`. Instead, it implements the (C) buffer protocol.

    if sys.version_info >= (3, 12):
        def __buffer__(self, flags: int, /) -> memoryview: ...

    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, memo: dict[int, object] | None, /) -> Self: ...

    @overload
    def __array__(self, /) -> _Array0D: ...
    @overload
    def __array__(self, dtype: _DT, /) -> _Array0D[_DT]: ...


_N_re = TypeVar('_N_re', bound=npt.NBitBase, default=npt.NBitBase)
_N_im = TypeVar('_N_im', bound=npt.NBitBase, default=_N_re)

Generic: TypeAlias = np.generic
Number: TypeAlias = np.number[_N_re]

Integer: TypeAlias = np.integer[_N_re]
UnsignedInteger: TypeAlias = np.unsignedinteger[_N_re]
SignedInteger: TypeAlias = np.signedinteger[_N_re]

Inexact: TypeAlias = np.inexact[_N_re]
Floating: TypeAlias = np.floating[_N_re]
ComplexFloating: TypeAlias = np.complexfloating[_N_re, _N_im]

Flexible: TypeAlias = np.flexible
Character: TypeAlias = np.character
