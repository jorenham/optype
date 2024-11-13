"""
Runtime-protocols for the `pickle` standard library.
https://docs.python.org/3/library/pickle.html
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Concatenate, Literal, Protocol, TypeAlias


if sys.version_info >= (3, 13):
    from typing import Self, TypeVar, override, runtime_checkable
else:
    from typing_extensions import Self, TypeVar, override, runtime_checkable

from ._core._can import CanIndex, CanIterSelf


__all__ = (
    "CanGetnewargs",
    "CanGetnewargsEx",
    "CanGetstate",
    "CanReduce",
    "CanReduceEx",
    "CanSetstate",
)


def __dir__() -> tuple[str, ...]:
    return __all__


# 5 is the highest since 3.8, and will become the new default in 3.14
_ProtocolVersion: TypeAlias = Literal[0, 1, 2, 3, 4, 5]

if sys.version_info >= (3, 11):
    _AnyCallable: TypeAlias = Callable[Concatenate[object, ...], object]
else:
    _AnyCallable: TypeAlias = Callable[..., object]  # type: ignore[no-any-explicit]
_AnyReduceValue: TypeAlias = (
    str
    | tuple[_AnyCallable, tuple[object, ...]]
    | tuple[_AnyCallable, tuple[object, ...], object]
    | tuple[_AnyCallable, tuple[object, ...], object, CanIterSelf[object] | None]
    | tuple[
        _AnyCallable,
        tuple[object, ...],
        object,
        CanIterSelf[object] | None,
        CanIterSelf[tuple[object, object]] | None,
    ]
    | tuple[
        _AnyCallable,
        tuple[object, ...],
        object,
        CanIterSelf[object] | None,
        CanIterSelf[tuple[object, object]] | None,
        Callable[[object, object], object] | None,
    ]
)
_ReduceT_co = TypeVar(
    "_ReduceT_co",
    bound=_AnyReduceValue,
    covariant=True,
    default=_AnyReduceValue,
)


@runtime_checkable
class CanReduce(Protocol[_ReduceT_co]):
    """
    https://docs.python.org/3/library/pickle.html#object.__reduce__
    """

    @override
    def __reduce__(self, /) -> _ReduceT_co: ...


@runtime_checkable
class CanReduceEx(Protocol[_ReduceT_co]):
    """
    https://docs.python.org/3/library/pickle.html#object.__reduce_ex__
    """

    @override
    def __reduce_ex__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        protocol: CanIndex[_ProtocolVersion],  # type: ignore[override]
        /,
    ) -> _ReduceT_co: ...


_StateT_co = TypeVar("_StateT_co", covariant=True)
_StateT_contra = TypeVar("_StateT_contra", contravariant=True)


@runtime_checkable
class CanGetstate(Protocol[_StateT_co]):
    """
    https://docs.python.org/3/library/pickle.html#object.__getstate__
    """

    def __getstate__(self, /) -> _StateT_co: ...


@runtime_checkable
class CanSetstate(Protocol[_StateT_contra]):
    """
    https://docs.python.org/3/library/pickle.html#object.__setstate__
    """

    def __setstate__(self, state: _StateT_contra, /) -> None: ...


_ArgT_co = TypeVar("_ArgT_co", covariant=True, default=object)
_KwargT = TypeVar("_KwargT", default=object)


@runtime_checkable
class CanGetnewargs(Protocol[_ArgT_co]):
    """
    https://docs.python.org/3/library/pickle.html#object.__getnewargs__
    """

    def __new__(cls, /, *args: _ArgT_co) -> Self: ...
    def __getnewargs__(self, /) -> tuple[_ArgT_co, ...]: ...


@runtime_checkable
class CanGetnewargsEx(Protocol[_ArgT_co, _KwargT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__getnewargs_ex__
    """

    def __new__(cls, /, *args: _ArgT_co, **kwargs: _KwargT) -> Self: ...
    def __getnewargs_ex__(
        self,
        /,
    ) -> tuple[tuple[_ArgT_co, ...], dict[str, _KwargT]]: ...
