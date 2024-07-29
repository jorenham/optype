from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, final

from ._can import CanGetitem, CanIter, CanLen, CanNext


if sys.version_info >= (3, 13):
    from typing import (
        Protocol,
        Self,
        TypeIs,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        Protocol,
        Self,  # noqa: TCH002
        TypeIs,  # noqa: TCH002
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )

if TYPE_CHECKING:
    from ._can import CanIndex


__all__ = (
    'AnyIterable',
    'Slice',
    'is_iterable',
)


_ValueT = TypeVar('_ValueT', default=Any)
# Anything that can be iterated over, e.g. in a `for` loop,`builtins.iter`,
# `builtins.enumerate`, or `numpy.array`.
AnyIterable: TypeAlias = CanIter[CanNext[_ValueT]] | CanGetitem[int, _ValueT]


def is_iterable(obj: object, /) -> TypeIs[AnyIterable]:
    """
    Check whether the object can be iterated over, i.e. if it can be used in
    a `for` loop, or if it can be passed to `builtins.iter`.

    Note:
        Contrary to popular *belief*, this isn't limited to objects that
        implement `__iter___`, as suggested by the name of
        `collections.abc.Iterable`.

        Sequence-like objects that implement `__getitem__` for consecutive
        `int` keys that start at `0` (or raise `IndexEeror` if out of bounds),
        can also be used in a `for` loop.
        In fact, all builtins that accept "iterables" also accept these
        sequence-likes at runtime.

    See also:
        - [`optype.types.Iterable`][optype.types.Iterable]
    """
    if isinstance(obj, type):
        # workaround the false-positive bug in `@runtime_checkable` for types
        return False

    if isinstance(obj, CanIter):
        return True
    if isinstance(obj, CanGetitem):
        # check if obj is a sequence-like
        if isinstance(obj, CanLen):
            return True

        # not all sequence-likes implement __len__, e.g. `ctypes.pointer`
        try:
            obj[0]
        except (IndexError, StopIteration):
            pass
        except (KeyError, ValueError, TypeError):
            return False

        return True

    return False


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
