"""Interfaces for the (generic) unary operations."""

import collections.abc as _abc
import typing as _tp

# arithmetic operations


@_tp.runtime_checkable
class CanNeg[Y](_tp.Protocol):
    """
    `-self`
    """
    def __neg__(self) -> Y: ...

@_tp.runtime_checkable
class CanPos[Y](_tp.Protocol):
    """
    `+self`
    """
    def __pos__(self) -> Y: ...

@_tp.runtime_checkable
class CanInvert[Y](_tp.Protocol):
    """
    `~self`
    """
    def __invert__(self) -> Y: ...

@_tp.runtime_checkable
class CanAbs[Y](_tp.Protocol):
    """
    `abs(self)`
    """
    def __abs__(self) -> Y: ...


# rounding

@_tp.runtime_checkable
class CanRound0[Y](_tp.Protocol):
    """
    `round(self) -> Y0`
    """
    def __round__(self) -> Y: ...

@_tp.runtime_checkable
class CanRound[N, Y, YN](_tp.Protocol):
    """
    Implements `round(self[, ndigits])` through `__pow__` with overloaded
    signatures:

    - `(Self) -> Y`
    - `(Self, N) -> YN`
    """
    @_tp.overload
    def __round__(self) -> Y: ...
    @_tp.overload
    def __round__(self, __ndigits: N) -> YN: ...

@_tp.runtime_checkable
class CanTrunc[Y](_tp.Protocol):
    """
    `math.trunc(self)`
    """
    def __trunc__(self) -> Y: ...

@_tp.runtime_checkable
class CanFloor[Y](_tp.Protocol):
    """
    `math.floor(self)`
    """
    def __floor__(self) -> Y: ...


@_tp.runtime_checkable
class CanCeil[Y](_tp.Protocol):
    """
    `math.ceil(self)`
    """
    def __ceil__(self) -> Y: ...

# iteration

@_tp.runtime_checkable
class CanReversed[Y](_tp.Protocol):
    """
    `reversed(self)`
    """
    def __reversed__(self) -> Y: ...


@_tp.runtime_checkable
class CanNext[Y](_tp.Protocol):
    """
    `next(self)`
    """
    def __next__(self) -> Y: ...


@_tp.runtime_checkable
class CanIter[Y: CanNext[_tp.Any]](_tp.Protocol):
    """
    `iter(self)`
    """
    def __iter__(self) -> Y: ...


# async iteration

@_tp.runtime_checkable
class CanANext[Y](_tp.Protocol):
    """
    `anext(self)`
    """
    def __anext__(self) -> Y: ...


@_tp.runtime_checkable
class CanAIter[Y: CanANext[_tp.Any]](_tp.Protocol):
    """
    `aiter(self)`
    """
    def __aiter__(self) -> Y: ...


# introspection

@_tp.runtime_checkable
class CanDir[T: _abc.Iterable[_tp.Any]](_tp.Protocol):
    def __dir__(self) -> T: ...
