# ruff: noqa: A005
"""
Runtime-protocols for the `copy` standard library.
https://docs.python.org/3/library/copy.html
"""
from __future__ import annotations

import sys


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
        Self,  # noqa: TCH002
        TypeVar,
        override,
        runtime_checkable,
    )


# fmt: off
__all__ = (
    'CanCopy', 'CanCopySelf',
    'CanDeepcopy', 'CanDeepcopySelf',
    'CanReplace', 'CanReplaceSelf',
)
# fmt: on

_CopyT = TypeVar('_CopyT', infer_variance=True)
_ValueT = TypeVar('_ValueT', infer_variance=True)


@runtime_checkable
class CanCopy(Protocol[_CopyT]):
    """Support for creating shallow copies through `copy.copy`."""
    def __copy__(self, /) -> _CopyT: ...


@runtime_checkable
class CanCopySelf(CanCopy['CanCopySelf'], Protocol):
    """Variant of `CanCopy` that returns `Self` (as it should)."""
    @override
    def __copy__(self, /) -> Self: ...


@runtime_checkable
class CanDeepcopy(Protocol[_CopyT]):
    """Support for creating deep copies through `copy.deepcopy`."""
    def __deepcopy__(self, memo: dict[int, object], /) -> _CopyT: ...


@runtime_checkable
class CanDeepcopySelf(CanDeepcopy['CanDeepcopySelf'], Protocol):
    """Variant of `CanDeepcopy` that returns `Self` (as it should)."""
    @override
    def __deepcopy__(self, memo: dict[int, object], /) -> Self: ...


@runtime_checkable
class CanReplace(Protocol[_ValueT, _CopyT]):
    """Support for `copy.replace` in Python 3.13+."""
    def __replace__(self, /, **changes: _ValueT) -> _CopyT: ...


@runtime_checkable
class CanReplaceSelf(
    CanReplace[_ValueT, 'CanReplaceSelf[_ValueT]'],
    Protocol[_ValueT],
):
    """Variant of `CanReplace[V, Self]`."""
    @override
    def __replace__(self, /, **changes: _ValueT) -> Self: ...
