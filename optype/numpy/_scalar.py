# ruff: noqa: PLW1641
# mypy: disable-error-code="no-any-explicit, no-any-decorated"

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from optype._core._utils import set_module

from ._compat import NP20


if sys.version_info >= (3, 13):
    from types import CapsuleType
    from typing import (
        Protocol,
        Self,
        TypeAliasType,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        CapsuleType,
        Protocol,
        Self,
        TypeAliasType,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )

if TYPE_CHECKING:
    if NP20:
        from numpy._core.multiarray import flagsobj
    else:
        from numpy.core.multiarray import flagsobj


__all__ = ["Scalar"]


_PT_co = TypeVar("_PT_co", covariant=True)
_NB_co = TypeVar("_NB_co", bound=int, covariant=True, default=int)
_DT = TypeVar("_DT", bound=np.dtype[np.generic], default=np.dtype[np.generic])

_L0: TypeAlias = Literal[0]
_L1: TypeAlias = Literal[1]
_Array0D: TypeAlias = np.ndarray[tuple[()], _DT]


@runtime_checkable
@set_module("optype.numpy")
class Scalar(Protocol[_PT_co, _NB_co]):
    """
    A lightweight `numpy.generic` interface that's actually generic, and
    doesn't require all that nasty `numpy.typing.NBitBase` stuff.
    """

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

    def item(self, k: _L0 | tuple[()] | tuple[_L0] = ..., /) -> _PT_co: ...

    @property
    def __array_priority__(self, /) -> float: ...  # -1000000.0
    @property
    def __array_interface__(self, /) -> dict[str, Any]: ...  # pyright: ignore[reportExplicitAny]
    @property
    def __array_struct__(self, /) -> CapsuleType: ...

    if NP20:

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


# `NBitBase` invariant and doesn't actually do anything, so the default should be `Any`
_N = TypeVar("_N", bound=npt.NBitBase, default=Any)  # pyright: ignore[reportExplicitAny]
_N1 = TypeVar("_N1", bound=npt.NBitBase, default=Any)  # pyright: ignore[reportExplicitAny]
_N2 = TypeVar("_N2", bound=npt.NBitBase, default=_N1)

generic = np.generic
flexible = np.flexible
character = np.character

number = TypeAliasType("number", np.number[_N], type_params=(_N,))
integer = TypeAliasType("integer", np.integer[_N], type_params=(_N,))
uinteger = TypeAliasType("uinteger", np.unsignedinteger[_N], type_params=(_N,))
sinteger = TypeAliasType("sinteger", np.signedinteger[_N], type_params=(_N,))
inexact = TypeAliasType("inexact", np.inexact[_N], type_params=(_N,))
floating = TypeAliasType("floating", np.floating[_N], type_params=(_N,))
cfloating = TypeAliasType(
    "cfloating",
    np.complexfloating[_N1, _N2],
    type_params=(_N1, _N2),
)
