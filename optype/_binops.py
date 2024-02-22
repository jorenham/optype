"""Interfaces for individual binary operations."""
import typing as _tp

# arithmetic operations


@_tp.runtime_checkable
class CanAdd[X, Y](_tp.Protocol):
    """
    `self + other`
    """
    def __add__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanSub[X, Y](_tp.Protocol):
    """
    `self - other`
    """
    def __sub__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanMul[X, Y](_tp.Protocol):
    """
    `self * other`
    """
    def __mul__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanMatmul[X, Y](_tp.Protocol):
    """
    `self @ other`
    """
    def __matmul__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanTruediv[X, Y](_tp.Protocol):
    """
    `self / other`
    """
    def __truediv__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanFloordiv[X, Y](_tp.Protocol):
    """
    `self // other`
    """
    def __floordiv__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanMod[X, Y](_tp.Protocol):
    """
    `self % other`
    """
    def __mod__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanDivmod[X, Y](_tp.Protocol):
    """
    `divmod(self, other)`
    """
    def __pow__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanPow0[X, Y](_tp.Protocol):
    """
    - `self ** other` or
    - `pow(self, other)`
    """
    def __pow__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanPow[X, M, Y, YM](_tp.Protocol):
    """
    Implements

    - `self ** other` and
    - `pow(self, other[, modulo])`,

    via `__pow__` with overloaded signatures

    - `(Self, X) -> Y` and
    - `(Self, X, M) -> YM`.

    Note that there is no `__rpow__` with modulo (the official docs are wrong).
    """
    @_tp.overload
    def __pow__(self, __other: X) -> Y: ...
    @_tp.overload
    def __pow__(self, __other: X, __modulo: M) -> YM: ...


# bitwise operations

@_tp.runtime_checkable
class CanLshift[X, Y](_tp.Protocol):
    """
    `self << other`
    """
    def __lshift__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanRshift[X, Y](_tp.Protocol):
    """
    `self >> other`
    """
    def __rshift__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanAnd[X, Y](_tp.Protocol):
    """
    `self & other`
    """
    def __and__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanXor[X, Y](_tp.Protocol):
    """
    `self ^ other`
    """
    def __xor__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanOr[X, Y](_tp.Protocol):
    """
    `self | other`
    """
    def __or__(self, __other: X, /) -> Y: ...
