"""
Runtime-protocols for the `copy` standard library.
https://docs.python.org/3/library/copy.html
"""
from __future__ import annotations

import sys
from typing import Any


if sys.version_info >= (3, 13):
    from typing import Protocol, Self, TypeVar, override, runtime_checkable
else:
    from typing_extensions import Protocol, Self, TypeVar, override, runtime_checkable


__all__ = (
    'CanCopy', 'CanCopySelf',
    'CanDeepcopy', 'CanDeepcopySelf',
    'CanReplace', 'CanReplaceSelf',
)  # fmt: skip

_T_co = TypeVar('_T_co', covariant=True)
_V_contra = TypeVar('_V_contra', contravariant=True)
_AnyV_contra = TypeVar('_AnyV_contra', contravariant=True, default=Any)


@runtime_checkable
class CanCopy(Protocol[_T_co]):
    """Support for creating shallow copies through `copy.copy`."""
    def __copy__(self, /) -> _T_co: ...


@runtime_checkable
class CanCopySelf(CanCopy['CanCopySelf'], Protocol):
    """Variant of `CanCopy` that returns `Self` (as it should)."""
    @override
    def __copy__(self, /) -> Self: ...


@runtime_checkable
class CanDeepcopy(Protocol[_T_co]):
    """Support for creating deep copies through `copy.deepcopy`."""
    def __deepcopy__(self, memo: dict[int, object], /) -> _T_co: ...


@runtime_checkable
class CanDeepcopySelf(CanDeepcopy['CanDeepcopySelf'], Protocol):
    """Variant of `CanDeepcopy` that returns `Self` (as it should)."""
    @override
    def __deepcopy__(self, memo: dict[int, object], /) -> Self: ...


@runtime_checkable
class CanReplace(Protocol[_V_contra, _T_co]):
    """Support for `copy.replace` in Python 3.13+."""
    def __replace__(self, /, **changes: _V_contra) -> _T_co: ...


@runtime_checkable
class CanReplaceSelf(
    CanReplace[_AnyV_contra, 'CanReplaceSelf[_AnyV_contra]'],
    Protocol[_AnyV_contra],
):
    """Variant of `CanReplace[V, Self]`."""
    @override
    def __replace__(self, /, **changes: _AnyV_contra) -> Self: ...
