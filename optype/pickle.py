"""
Runtime-protocols for the `pickle` standard library.
https://docs.python.org/3/library/pickle.html
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from optype._can import CanIterSelf


if TYPE_CHECKING:
    from ._can import CanIndex


if sys.version_info >= (3, 13):
    from typing import (
        Protocol,
        Self,
        TypeVar,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        Protocol,
        Self,
        TypeVar,
        override,
        runtime_checkable,
    )


__all__ = (
    'CanGetnewargs',
    'CanGetnewargsEx',
    'CanGetstate',
    'CanReduce',
    'CanReduceEx',
    'CanSetstate',
)


# 5 is the highest since 3.8, and will become the new default in 3.14
_ProtocolVersion: TypeAlias = Literal[0, 1, 2, 3, 4, 5]

# _AnyReduceValue: TypeAlias = str | tuple[Any, ...]  # noqa: ERA001
_AnyReduceValue: TypeAlias = (
    str
    | tuple[Callable[..., Any], tuple[Any, ...]]
    | tuple[Callable[..., Any], tuple[Any, ...], Any]
    | tuple[Callable[..., Any], tuple[Any, ...], Any, CanIterSelf[Any] | None]
    | tuple[
        Callable[..., Any],
        tuple[Any, ...],
        Any,
        CanIterSelf[Any] | None,
        CanIterSelf[tuple[Any, Any]] | None,
    ]
    | tuple[
        Callable[..., Any],
        tuple[Any, ...],
        Any,
        CanIterSelf[Any] | None,
        CanIterSelf[tuple[Any, Any]] | None,
        Callable[[Any, Any], Any] | None,
    ]
)
_ReduceValueT = TypeVar(
    '_ReduceValueT',
    infer_variance=True,
    bound=_AnyReduceValue,
    default=_AnyReduceValue,
)


@runtime_checkable
class CanReduce(Protocol[_ReduceValueT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__reduce__
    """
    @override
    def __reduce__(self, /) -> _ReduceValueT: ...


@runtime_checkable
class CanReduceEx(Protocol[_ReduceValueT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__reduce_ex__
    """
    @override
    def __reduce_ex__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        protocol: CanIndex[_ProtocolVersion],
        /,
    ) -> _ReduceValueT: ...


_StateT = TypeVar('_StateT', infer_variance=True)


@runtime_checkable
class CanGetstate(Protocol[_StateT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__getstate__
    """
    def __getstate__(self, /) -> _StateT: ...


@runtime_checkable
class CanSetstate(Protocol[_StateT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__setstate__
    """
    def __setstate__(self, state: _StateT, /) -> None: ...


_ArgT = TypeVar('_ArgT', infer_variance=True, default=Any)
_KwargT = TypeVar('_KwargT', infer_variance=True, default=Any)


@runtime_checkable
class CanGetnewargs(Protocol[_ArgT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__getnewargs__
    """
    def __new__(cls, /, *args: _ArgT) -> Self: ...
    def __getnewargs__(self, /) -> tuple[_ArgT, ...]: ...


@runtime_checkable
class CanGetnewargsEx(Protocol[_ArgT, _KwargT]):
    """
    https://docs.python.org/3/library/pickle.html#object.__getnewargs_ex__
    """
    def __new__(cls, /, *args: _ArgT, **kwargs: _KwargT) -> Self: ...
    def __getnewargs_ex__(
        self, /,
    ) -> tuple[tuple[_ArgT, ...], dict[str, _KwargT]]: ...
