from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias


if TYPE_CHECKING:
    import numpy as np
    from numpy._core.multiarray import flagsobj


if sys.version_info >= (3, 13):
    from types import CapsuleType
    from typing import (
        Protocol,
        Self,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        CapsuleType,  # noqa: TCH002
        Protocol,
        Self,  # noqa: TCH002
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )


_L0: TypeAlias = Literal[0]
_L1: TypeAlias = Literal[1]


_T_Scalar = TypeVar('_T_Scalar', infer_variance=True, bound=object)
_N_Scalar = TypeVar('_N_Scalar', infer_variance=True, bound=int)
_ST_Scalar = TypeVar('_ST_Scalar', bound='np.generic')
_DT_Scalar = TypeVar('_DT_Scalar', bound='np.dtype[Any]')


@runtime_checkable  # noqa: PLR0904
class Scalar(Protocol[_T_Scalar, _N_Scalar]):
    """
    A lightweight `numpy.generic` inface, that's actually generic
    and doesn't require that nasty `numpy.typing.NBitBase` stuff.

    TODO: implement a subset of the following overlapping methods/attrs
    TODO: test
    """
    @property
    def T(self, /) -> Self: ...  # noqa: N802
    @property
    def mT(self, /) -> Self: ...  # noqa: N802
    @property
    def flags(self, /) -> flagsobj: ...  # T0DO: use Protocol instead
    @property
    def data(self, /) -> memoryview: ...
    @property
    def base(self, /) -> None: ...
    @property
    def dtype(self: _ST_Scalar, /) -> np.dtype[_ST_Scalar]: ...
    @property
    def ndim(self, /) -> _L0: ...
    @property
    def size(self, /) -> _L1: ...
    @property
    def shape(self, /) -> tuple[()]: ...
    @property
    def strides(self, /) -> tuple[()]: ...
    @property
    def itemsize(self, /) -> _N_Scalar: ...
    @property
    def nbytes(self, /) -> _N_Scalar: ...  # always same as itemsize?

    @property
    def __array_priority__(self, /) -> float: ...  # -1000000.0
    @property
    def __array_interface__(self, /) -> dict[str, Any]: ...  # TypedDict?
    @property
    def __array_struct__(self, /) -> CapsuleType: ...
    @overload
    def __array__(
        self: _ST_Scalar,
        dtype: None = ...,
        /,
    ) -> np.ndarray[tuple[()], np.dtype[_ST_Scalar]]: ...
    @overload
    def __array__(
        self,
        dtype: _DT_Scalar,
        /,
    ) -> np.ndarray[tuple[()], _DT_Scalar]: ...

    if sys.version_info >= (3, 12):
        def __buffer__(self, flags: int, /) -> memoryview: ...

    def __bool__(self, /) -> bool: ...
    def __bytes__(self, /) -> bytes: ...
    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any] | None, /) -> Self: ...

    # never a builtins.bool?
    @override
    def __eq__(self, other: object, /) -> np.bool_: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ne__(self, other: object, /) -> np.bool_: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __hash__(self, /) -> int: ...

    def item(self, k: _L0 | tuple[()] | tuple[_L0] = ..., /) -> _T_Scalar: ...
