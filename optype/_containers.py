"""Generic interfaces for elementary container operations."""
import typing as _tp


@_tp.runtime_checkable
class CanContains[X](_tp.Protocol):
    # vibrantly generic
    """
    `other in self`
    """
    def __contains__(self, __other: X) -> bool: ...

@_tp.runtime_checkable
class CanGetitem[K, V](_tp.Protocol):
    """
    `self[key]`
    """
    def __getitem__(self, __key: K) -> V: ...

@_tp.runtime_checkable
class CanSetitem[K, V](_tp.Protocol):
    """
    `self[key] = value`
    """
    def __setitem__(self, __key: K, __value: V) -> None: ...

@_tp.runtime_checkable
class CanDelitem[K](_tp.Protocol):
    """
    `self[key] = value`
    """
    def __delitem__(self, __key: K) -> None: ...

@_tp.runtime_checkable
class CanMissing[K, V](_tp.Protocol):
    """
    fallback for `self[key]`
    """
    def __missing__(self, __key: K) -> V: ...
