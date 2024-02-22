# ruff: noqa: PYI034
"""Interfaces for the augmented / in-place binary operation variants."""
import typing as _tp

# arithmetic operations


@_tp.runtime_checkable
class CanIAdd[X, Y](_tp.Protocol):
    """
    `_ += other`
    """
    def __iadd__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanISub[X, Y](_tp.Protocol):
    """
    `_ -= other`
    """
    def __isub__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanIMul[X, Y](_tp.Protocol):
    """
    `_ *= other`
    """
    def __imul__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanIMatmul[X, Y](_tp.Protocol):
    """
    `_ @= other`
    """
    def __imatmul__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanITruediv[X, Y](_tp.Protocol):
    """
    `_ /= other`
    """
    def __itruediv__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanIFloordiv[X, Y](_tp.Protocol):
    """
    `_ //= other`
    """
    def __ifloordiv__(self, __other: X) -> Y: ...

@_tp.runtime_checkable
class CanIMod[X, Y](_tp.Protocol):
    """
    `_ %= other`
    """
    def __imod__(self, __other: X) -> Y: ...

# no __idivmod__

@_tp.runtime_checkable
class CanIPow[X, Y](_tp.Protocol):
    """
    `_ **= other`
    """
    def __ipow__(self, __other: X) -> Y: ...


# bitwise operations (augmented)

@_tp.runtime_checkable
class CanILshift[X, Y](_tp.Protocol):
    """
    `_ <<= other`
    """
    def __ilshift__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanIRshift[X, Y](_tp.Protocol):
    """
    `_ >>= other`
    """
    def __irshift__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanIAnd[X, Y](_tp.Protocol):
    """
    `_ &= other`
    """
    def __iand__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanIXor[X, Y](_tp.Protocol):
    """
    `_ ^= other`
    """
    def __ixor__(self, __other: X, /) -> Y: ...

@_tp.runtime_checkable
class CanIOr[X, Y](_tp.Protocol):
    """
    `_ |= other`
    """
    def __ior__(self, __other: X, /) -> Y: ...


# formatting

@_tp.runtime_checkable
class CanFormat[X: str, Y: str](_tp.Protocol):
    """
    `format(self[, format_spec])`

    note that both the `format_spec` and the returned value can be subclasses
    of `str`.
    """
    def __format__(self, __format_spec: X) -> Y: ...  # type: ignore[override]
