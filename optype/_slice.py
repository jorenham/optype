"""But a dream of a generic `slice`."""
from __future__ import annotations

import typing as _tp

if _tp.TYPE_CHECKING:
    import types as _ts

    from ._nullops import CanIndex as _CanIndex


@_tp.final
@_tp.runtime_checkable
class Slice[A, B, S](_tp.Protocol):
    start: _tp.Final[A]
    stop: _tp.Final[B]
    step: _tp.Final[S]

    @_tp.overload
    def __new__(cls, __stop: B) -> Slice[None, B, None]: ...
    @_tp.overload
    def __new__(cls, __start: A, __stop: B) -> Slice[A, B, None]: ...
    @_tp.overload
    def __new__(cls, __start: A, __stop: B, __step: S) -> Slice[A, B, S]: ...

    def __eq__(self, __other: object) -> bool: ...

    def __ne__(self, __other: object) -> bool: ...

    def __lt__(self, __other: object) -> bool | _ts.NotImplementedType: ...

    def __le__(self, __other: object) -> bool | _ts.NotImplementedType: ...

    def __ge__(self, __other: object) -> bool | _ts.NotImplementedType: ...

    def __gt__(self, __other: object) -> bool | _ts.NotImplementedType: ...

    @_tp.overload
    def indices(
        self: Slice[_CanIndex | None, _CanIndex | None, _CanIndex | None],
        __len: _CanIndex,
    ) -> tuple[int, int, int]: ...
    @_tp.overload
    def indices(self: Slice[A, B, S], __len: _CanIndex) -> _tp.NoReturn: ...
