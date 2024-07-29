from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, final


if TYPE_CHECKING:
    from ._can import CanIndex


if sys.version_info >= (3, 13):
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
        Protocol,
        Self,  # noqa: TCH002
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )


__all__ = ('Slice',)


_Just0: TypeAlias = Literal[0]
_Just1: TypeAlias = Literal[1]

_StartT = TypeVar('_StartT', infer_variance=True, default=None)
_StopT = TypeVar('_StopT', infer_variance=True, default=Any)
_StepT = TypeVar('_StepT', infer_variance=True, default=None)


@final
@runtime_checkable
class Slice(Protocol[_StartT, _StopT, _StepT]):
    @property
    def start(self, /) -> _StartT: ...
    @property
    def stop(self, /) -> _StopT: ...
    @property
    def step(self, /) -> _StepT: ...

    @overload
    def __new__(cls, b: _StopT, /) -> Self: ...
    @overload
    def __new__(cls, a: _StartT, b: _StopT, /) -> Self: ...
    @overload
    def __new__(cls, a: _StartT, b: _StopT, s: _StepT, /) -> Self: ...

    if sys.version_info >= (3, 12):
        def __hash__(self, /) -> int: ...
    else:
        @override
        def __eq__(self, value: object, /) -> bool: ...
        __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]

    @overload
    def indices(
        self: Slice[None, CanIndex | None, None],
        n: CanIndex,
        /,
    ) -> tuple[_Just0, int, _Just1]: ...
    @overload
    def indices(
        self: Slice[None, CanIndex | None, CanIndex],
        n: CanIndex,
        /,
    ) -> tuple[_Just0, int, int]: ...
    @overload
    def indices(
        self: Slice[CanIndex, CanIndex | None, None],
        n: CanIndex,
        /,
    ) -> tuple[int, int, _Just1]: ...
    @overload
    def indices(
        self: Slice[CanIndex, CanIndex | None, CanIndex],
        n: CanIndex,
        /,
    ) -> tuple[int, int, int]: ...
