from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, final


if TYPE_CHECKING:
    from ._can import CanIndex as _Ix


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


_Zero: TypeAlias = Literal[0]
_One: TypeAlias = Literal[1]

_A_Slice = TypeVar('_A_Slice', infer_variance=True, default=None)
_B_Slice = TypeVar('_B_Slice', infer_variance=True, default=Any)
_S_Slice = TypeVar('_S_Slice', infer_variance=True, default=None)


@final
@runtime_checkable
class Slice(Protocol[_A_Slice, _B_Slice, _S_Slice]):
    @property
    def start(self, /) -> _A_Slice: ...
    @property
    def stop(self, /) -> _B_Slice: ...
    @property
    def step(self, /) -> _S_Slice: ...

    @overload
    def __new__(cls, b: _B_Slice, /) -> Self: ...
    @overload
    def __new__(cls, a: _A_Slice, b: _B_Slice, /) -> Self: ...
    @overload
    def __new__(cls, a: _A_Slice, b: _B_Slice, s: _S_Slice, /) -> Self: ...

    if sys.version_info >= (3, 12):
        def __hash__(self, /) -> int: ...
    else:
        @override
        def __eq__(self, value: object, /) -> bool: ...
        __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]

    @overload
    def indices(
        self: Slice[None, _Ix | None, None],
        n: _Ix,
        /,
    ) -> tuple[_Zero, int, _One]: ...
    @overload
    def indices(
        self: Slice[None, _Ix | None, _Ix],
        n: _Ix,
        /,
    ) -> tuple[_Zero, int, int]: ...
    @overload
    def indices(
        self: Slice[_Ix, _Ix | None, None],
        n: _Ix,
        /,
    ) -> tuple[int, int, _One]: ...
    @overload
    def indices(
        self: Slice[_Ix, _Ix | None, _Ix],
        n: _Ix,
        /,
    ) -> tuple[int, int, int]: ...
