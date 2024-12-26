# mypy: disable-error-code="no-any-explicit"
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

import optype.numpy._compat as _x
from optype._core._utils import set_module


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeAliasType, TypeVar, runtime_checkable
else:
    from typing_extensions import (
        Protocol,
        Self,
        TypeAliasType,
        TypeVar,
        runtime_checkable,
    )

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "Array",
    "Array0D",
    "Array1D",
    "Array1D",
    "Array2D",
    "Array3D",
    "ArrayND",
    "CanArray",
    "CanArray0D",
    "CanArray1D",
    "CanArray2D",
    "CanArray3D",
    "CanArrayFinalize",
    "CanArrayND",
    "CanArrayWrap",
    "HasArrayInterface",
    "HasArrayPriority",
    "MArray",
    "MArray0D",
    "MArray1D",
    "MArray1D",
    "MArray2D",
    "MArray3D",
    "Matrix",
]


_NDT = TypeVar("_NDT", bound=tuple[int, ...], default=tuple[int, ...])
_NDT_any = TypeVar(  # for numpy < 2.1
    "_NDT_any",
    bound=tuple[int, ...],
    default=Any,  # pyright: ignore[reportExplicitAny]
)
_NDT_co = TypeVar(
    "_NDT_co",
    bound=tuple[int, ...],
    default=tuple[int, ...],
    covariant=True,
)
_DTT = TypeVar("_DTT", bound=np.dtype[np.generic], default=np.dtype[np.generic])
_DTT_co = TypeVar(
    "_DTT_co",
    bound=np.dtype[np.generic],
    default=np.dtype[np.generic],
    covariant=True,
)
_SCT = TypeVar("_SCT", bound=np.generic, default=np.generic)
_SCT_co = TypeVar("_SCT_co", bound=np.generic, default=np.generic, covariant=True)

_MT = TypeVar("_MT", bound=int, default=int)
_NT = TypeVar("_NT", bound=int, default=_MT)


Matrix = TypeAliasType(
    "Matrix",
    np.matrix[tuple[_MT, _NT], np.dtype[_SCT]],
    type_params=(_SCT, _MT, _NT),
)
"""
Alias of `np.matrix` that is similar to `ArrayND`:

```py
type Matrix[
    SCT: np.generic = np.generic,
    M: int = int,
    N: int = M,
] = np.matrix[tuple[M, N], np.dtype[SCT]]
```

Only a "base" `int` type, or `Literal` or positive integers should be used as type
arguments to `MT` and `NT`. Be careful not to pass it a `bool` or any other `int`
subtype such as `Never`. There's also no need to use `Any` for `MT` or `NT`, as
the (variadic) type parameters of `tuple` are covariant (even though that's
supposed to be illegal for variadic type params, which makes no fucking sense).
"""

if _x.NP21:
    # numpy >= 2.1: shape is covariant

    Array = TypeAliasType(
        "Array",
        np.ndarray[_NDT, np.dtype[_SCT]],
        type_params=(_NDT, _SCT),
    )
    """
    Shape-typed array alias, defined as:

    ```py
    type Array[
        NDT: (int, ...) = (int, ...),
        SCT: np.generic = np.generic,
    ] = np.ndarray[NDT, np.dtype[SCT]]
    ```
    """

    ArrayND = TypeAliasType(
        "ArrayND",
        np.ndarray[_NDT, np.dtype[_SCT]],
        type_params=(_SCT, _NDT),
    )
    """
    Like `Array`, but with flipped type-parameters, i.e.:

    type ArrayND[
        SCT: np.generic = np.generic,
        NDT: (int, ...) = (int, ...),
    ] = np.ndarray[NDT, np.dtype[SCT]]

    Because the optional shape-type parameter comes *after* the scalar-type, `ArrayND`
    can be seen as a flexible generalization of `npt.NDArray`.
    """

    MArray = TypeAliasType(
        "MArray",
        np.ma.MaskedArray[_NDT, np.dtype[_SCT]],
        type_params=(_SCT, _NDT),
    )
    """
    Just like `ArrayND`, but for `np.ma.MaskedArray` instead of `np.ndarray`.

    type MArray[
        SCT: np.generic = np.generic,
        NDT: (int, ...) = (int, ...),
    ] = np.ma.MaskedArray[NDT, np.dtype[SCT]]
    """

    @runtime_checkable
    @set_module("optype.numpy")
    class CanArray(Protocol[_NDT_co, _DTT_co]):
        def __array__(self, /) -> np.ndarray[_NDT_co, _DTT_co]: ...

    @runtime_checkable
    @set_module("optype.numpy")
    class CanArrayND(Protocol[_SCT_co, _NDT_co]):
        """
        Similar to `onp.CanArray`, but must be sized (i.e. excludes scalars), and is
        parameterized by only the scalar type (instead of the shape and dtype).
        """

        def __len__(self, /) -> int: ...
        def __array__(self, /) -> np.ndarray[_NDT_co, np.dtype[_SCT_co]]: ...

else:
    # numpy < 2.1: shape is invariant

    Array = TypeAliasType(
        "Array",
        np.ndarray[_NDT_any, np.dtype[_SCT]],
        type_params=(_NDT_any, _SCT),
    )
    ArrayND = TypeAliasType(
        "ArrayND",
        np.ndarray[_NDT_any, np.dtype[_SCT]],
        type_params=(_SCT, _NDT_any),
    )
    MArray = TypeAliasType(
        "MArray",
        np.ma.MaskedArray[_NDT_any, np.dtype[_SCT]],
        type_params=(_SCT, _NDT_any),
    )

    @runtime_checkable
    @set_module("optype.numpy")
    class CanArray(Protocol[_NDT_any, _DTT_co]):
        def __array__(self, /) -> np.ndarray[_NDT_any, _DTT_co]: ...

    @runtime_checkable
    @set_module("optype.numpy")
    class CanArrayND(Protocol[_SCT_co, _NDT_any]):
        def __len__(self, /) -> int: ...
        def __array__(self, /) -> np.ndarray[_NDT_any, np.dtype[_SCT_co]]: ...


Array0D = TypeAliasType(
    "Array0D",
    np.ndarray[tuple[()], np.dtype[_SCT]],
    type_params=(_SCT,),
)
Array1D = TypeAliasType(
    "Array1D",
    np.ndarray[tuple[int], np.dtype[_SCT]],
    type_params=(_SCT,),
)
Array2D = TypeAliasType(
    "Array2D",
    np.ndarray[tuple[int, int], np.dtype[_SCT]],
    type_params=(_SCT,),
)
Array3D = TypeAliasType(
    "Array3D",
    np.ndarray[tuple[int, int, int], np.dtype[_SCT]],
    type_params=(_SCT,),
)

MArray0D = TypeAliasType(
    "MArray0D",
    np.ma.MaskedArray[tuple[()], np.dtype[_SCT]],
    type_params=(_SCT,),
)
MArray1D = TypeAliasType(
    "MArray1D",
    np.ma.MaskedArray[tuple[int], np.dtype[_SCT]],
    type_params=(_SCT,),
)
MArray2D = TypeAliasType(
    "MArray2D",
    np.ma.MaskedArray[tuple[int, int], np.dtype[_SCT]],
    type_params=(_SCT,),
)
MArray3D = TypeAliasType(
    "MArray3D",
    np.ma.MaskedArray[tuple[int, int, int], np.dtype[_SCT]],
    type_params=(_SCT,),
)


###########################


@runtime_checkable
@set_module("optype.numpy")
class CanArray0D(Protocol[_SCT_co]):
    """
    The 0-d variant of `optype.numpy.CanArrayND`.

    This accepts e.g. `np.asarray(3.14)`, but rejects `np.float64(3.14)`.
    """

    def __len__(self, /) -> int: ...  # always 0
    def __array__(self, /) -> np.ndarray[tuple[()], np.dtype[_SCT_co]]: ...


@runtime_checkable
@set_module("optype.numpy")
class CanArray1D(Protocol[_SCT_co]):
    """The 1-d variant of `optype.numpy.CanArrayND`."""

    def __len__(self, /) -> int: ...
    def __array__(self, /) -> np.ndarray[tuple[int], np.dtype[_SCT_co]]: ...


@runtime_checkable
@set_module("optype.numpy")
class CanArray2D(Protocol[_SCT_co]):
    """The 2-d variant of `optype.numpy.CanArrayND`."""

    def __len__(self, /) -> int: ...
    def __array__(self, /) -> np.ndarray[tuple[int, int], np.dtype[_SCT_co]]: ...


@runtime_checkable
@set_module("optype.numpy")
class CanArray3D(Protocol[_SCT_co]):
    """The 2-d variant of `optype.numpy.CanArrayND`."""

    def __len__(self, /) -> int: ...
    def __array__(self, /) -> np.ndarray[tuple[int, int, int], np.dtype[_SCT_co]]: ...


# this is almost always a `ndarray`, but setting a `bound` might break in some
# edge cases
_T_contra = TypeVar("_T_contra", contravariant=True, default=object)


@runtime_checkable
@set_module("optype.numpy")
class CanArrayFinalize(Protocol[_T_contra]):
    def __array_finalize__(self, obj: _T_contra, /) -> None: ...


@runtime_checkable
@set_module("optype.numpy")
class CanArrayWrap(Protocol):
    if _x.NP20:

        def __array_wrap__(
            self,
            array: np.ndarray[_NDT, _DTT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            return_scalar: bool = ...,
            /,
        ) -> np.ndarray[_NDT, _DTT] | Self: ...

    else:

        def __array_wrap__(
            self,
            array: np.ndarray[_NDT, _DTT],
            context: tuple[np.ufunc, tuple[object, ...], int] | None = ...,
            /,
        ) -> np.ndarray[_NDT, _DTT] | Self: ...


_ArrayInterfaceT_co = TypeVar(
    "_ArrayInterfaceT_co",
    covariant=True,
    bound="Mapping[str, object]",
    default=dict[str, object],
)


@runtime_checkable
@set_module("optype.numpy")
class HasArrayInterface(Protocol[_ArrayInterfaceT_co]):
    @property
    def __array_interface__(self, /) -> _ArrayInterfaceT_co: ...


@runtime_checkable
@set_module("optype.numpy")
class HasArrayPriority(Protocol):
    @property
    def __array_priority__(self, /) -> float: ...
