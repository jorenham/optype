"""Interfaces for the "rich comparison" methods (auto-reflective)."""
import typing as _tp


@_tp.runtime_checkable
class CanLt[X, Y](_tp.Protocol):
    """
    - `self < other`
    - `other > self`
    """
    def __lt__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanGt[X, Y](_tp.Protocol):
    """
    - `self > other`
    - `other < self`
    """
    def __gt__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanLe[X, Y](_tp.Protocol):
    """
    - `self <= other`
    - `other >= self`
    """
    def __le__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanGe[X, Y](_tp.Protocol):
    """
    - `self >= other`
    - `other <= self`
    """
    def __ge__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanEq[X, Y](_tp.Protocol):
    """
    - `self == other`
    - `other == self`
    """
    def __eq__(self, __other: X, /) -> Y: ...  # type: ignore[override]

@_tp.runtime_checkable
class CanNe[X, Y](_tp.Protocol):
    """
    - `self != other`
    - `other != self`
    """
    def __ne__(self, __other: X, /) -> Y: ...  # type: ignore[override]
