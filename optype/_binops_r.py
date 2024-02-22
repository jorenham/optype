"""Interfaces for the reflected variants of thebinary operation."""
import typing as _tp

# arithmetic operations


@_tp.runtime_checkable
class CanRAdd[X, Y](_tp.Protocol):
    """
    `other + self`
    """
    def __radd__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanRSub[X, Y](_tp.Protocol):
    """
    `other - self`
    """
    def __rsub__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanRMul[X, Y](_tp.Protocol):
    """
    `other * self`
    """
    def __rmul__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanRMatmul[X, Y](_tp.Protocol):
    """
    `other @ self`
    """
    def __rmatmul__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanRTruediv[X, Y](_tp.Protocol):
    """
    `other / self`
    """
    def __rtruediv__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanRFloordiv[X, Y](_tp.Protocol):
    """
    `other // self`
    """
    def __rfloordiv__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanRMod[X, Y](_tp.Protocol):
    """
    `other % self`
    """
    def __rmod__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanRDivmod[X, Y](_tp.Protocol):
    """
    `divmod(other, self)`
    """
    def __rdivmod__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanRPow[X, Y](_tp.Protocol):
    """
    `other ** self` or
    `pow(other, self)`

    note that `pow(a, b, modulo)` will not be reflected
    """
    def __rpow__(self, __other: X) -> Y: ...


# bitwise operations

@_tp.runtime_checkable
class CanRLshift[X, Y](_tp.Protocol):
    """
    `other << self`
    """
    def __rlshift__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanRRshift[X, Y](_tp.Protocol):
    """
    `other >> self`
    """
    def __rrshift__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanRAnd[X, Y](_tp.Protocol):
    """
    `other & self `
    """
    def __rand__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanRXor[X, Y](_tp.Protocol):
    """
    `other ^ self`
    """
    def __rxor__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanROr[X, Y](_tp.Protocol):
    """
    `other | self`
    """
    def __ror__(self, __other: X, /) -> Y: ...
