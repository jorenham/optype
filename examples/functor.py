# ruff: noqa: INP001
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    final,
    override,
)

import optype as opt


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import NotImplementedType


"""
Automatic type-variance inference doesn't work here. That's no surprise, since
it is impossible to do so in all cases. So that's yet another theoretically
incorrect (and therefore broken) python-typing "feature"...
"""
T_co = TypeVar('T_co', covariant=True)


@final  # noqa: PLR0904
class Functor(Generic[T_co]):
    __match_args__ = __slots__ = ('value',)

    def __init__(self, value: T_co, /) -> None:
        self.value = value

    def map1[Y](self, f: Callable[[T_co], Y]) -> Functor[Y]:
        """
        Applies a unary operator `f` over the value of `self`,
        and return a new `Functor`.
        """
        return Functor(f(self.value))

    def map2[X, Y](
        self,
        f: Callable[[T_co, X], Y],
        other: Functor[X] | Any,
    ) -> Functor[Y] | NotImplementedType:
        """
        Maps the binary operator `f: (T, X) -> Y` over `self: Functor[T]` and
        `other: Functor[X]`, and returns `Functor[Y]`. A `NotImplemented`
        is returned if `f` is not supported for the types, or if other is not
        a `Functor`.
        """
        if isinstance(other, Functor):
            y = f(self.value, other.value)
            if y is not NotImplemented:
                return Functor(y)

        return NotImplemented

    @override
    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.value!r})'

    @override
    def __hash__(self: Functor[opt.CanHash]) -> int:
        return opt.do_hash(self.value)

    # unary prefix ops

    def __neg__[Y](self: Functor[opt.CanNeg[Y]]) -> Functor[Y]:
        """
        >>> -Functor(3.14)
        Functor(-3.14)
        """
        return self.map1(opt.do_neg)

    def __pos__[Y](self: Functor[opt.CanPos[Y]]) -> Functor[Y]:
        """
        >>> +Functor(True)
        Functor(1)
        """
        return self.map1(opt.do_pos)

    def __invert__[Y](self: Functor[opt.CanInvert[Y]]) -> Functor[Y]:
        """
        >>> ~Functor(0)
        Functor(-1)
        """
        return self.map1(opt.do_invert)

    # rich comparison ops

    def __lt__[X, Y](
        self: Functor[opt.CanLt[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0}) < Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) < Functor((1, -1))
        Functor(True)
        """
        return self.map2(opt.do_lt, x)

    def __le__[X, Y](
        self: Functor[opt.CanLe[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0}) <= Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) <= Functor((1, -1))
        Functor(True)
        """
        return self.map2(opt.do_le, x)

    @override
    def __eq__[X, Y](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Functor[opt.CanEq[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(object()) == Functor(object())
        Functor(False)
        >>> Functor(0) == Functor(0)
        Functor(True)
        >>> Functor(0) == 0
        False
        """
        return self.map2(opt.do_eq, x)

    @override
    def __ne__[X, Y](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Functor[opt.CanNe[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(object()) != Functor(object())
        Functor(True)
        >>> Functor(0) != Functor(0)
        Functor(False)
        >>> Functor(0) != 0
        True
        """
        return self.map2(opt.do_ne, x)

    def __gt__[X, Y](
        self: Functor[opt.CanGt[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0, 1}) > Functor({0})
        Functor(True)
        >>> Functor((0, 1)) > Functor((1, -1))
        Functor(False)
        """
        return self.map2(opt.do_gt, x)

    def __ge__[X, Y](
        self: Functor[opt.CanGe[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0, 1}) >= Functor({0})
        Functor(True)
        >>> Functor((0, 1)) >= Functor((1, -1))
        Functor(False)
        """
        return self.map2(opt.do_ge, x)

    # binary infix ops

    def __add__[X, Y](
        self: Functor[opt.CanAdd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(0) + Functor(1)
        Functor(1)
        >>> Functor(('spam',)) + Functor(('ham',)) + Functor(('eggs',))
        Functor(('spam', 'ham', 'eggs'))
        """
        return self.map2(opt.do_add, x)

    def __sub__[X, Y](
        self: Functor[opt.CanSub[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(0) - Functor(1)
        Functor(-1)
        """
        return self.map2(opt.do_sub, x)

    def __mul__[X, Y](
        self: Functor[opt.CanMul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(('Developers!',)) * Functor(4)
        Functor(('Developers!', 'Developers!', 'Developers!', 'Developers!'))
        """
        return self.map2(opt.do_mul, x)

    def __matmul__[X, Y](
        self: Functor[opt.CanMatmul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_matmul, x)

    def __truediv__[X, Y](
        self: Functor[opt.CanTruediv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1) / Functor(2)
        Functor(0.5)
        """
        return self.map2(opt.do_truediv, x)

    def __floordiv__[X, Y](
        self: Functor[opt.CanFloordiv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1) // Functor(2)
        Functor(0)
        """
        return self.map2(opt.do_floordiv, x)

    def __mod__[X, Y](
        self: Functor[opt.CanMod[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(10) % Functor(7)
        Functor(3)
        """
        return self.map2(opt.do_mod, x)

    def __pow__[X, Y](
        self: Functor[opt.CanPow2[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(2) ** Functor(3)
        Functor(8)
        """
        return self.map2(opt.do_pow, x)

    def __lshift__[X, Y](
        self: Functor[opt.CanLshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1) << Functor(10)
        Functor(1024)
        """
        return self.map2(opt.do_lshift, x)

    def __rshift__[X, Y](
        self: Functor[opt.CanRshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1024) >> Functor(4)
        Functor(64)
        """
        return self.map2(opt.do_rshift, x)

    def __and__[X, Y](
        self: Functor[opt.CanAnd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(True) & Functor(False)
        Functor(False)
        >>> Functor(3) & Functor(7)
        Functor(3)
        """
        return self.map2(opt.do_and, x)

    def __xor__[X, Y](
        self: Functor[opt.CanXor[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(True) ^ Functor(False)
        Functor(True)
        >>> Functor(3) ^ Functor(7)
        Functor(4)
        """
        return self.map2(opt.do_xor, x)

    def __or__[X, Y](
        self: Functor[opt.CanOr[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(True) | Functor(False)
        Functor(True)
        >>> Functor(3) | Functor(7)
        Functor(7)
        """
        return self.map2(opt.do_or, x)

    # binary reflected infix ops

    def __radd__[X, Y](
        self: Functor[opt.CanRAdd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_radd, x)

    def __rsub__[X, Y](
        self: Functor[opt.CanRSub[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rsub, x)

    def __rmul__[X, Y](
        self: Functor[opt.CanRMul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rmul, x)

    def __rmatmul__[X, Y](
        self: Functor[opt.CanRMatmul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rmatmul, x)

    def __rtruediv__[X, Y](
        self: Functor[opt.CanRTruediv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rtruediv, x)

    def __rfloordiv__[X, Y](
        self: Functor[opt.CanRFloordiv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rfloordiv, x)

    def __rmod__[X, Y](
        self: Functor[opt.CanRMod[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rmod, x)

    def __rpow__[X, Y](
        self: Functor[opt.CanRPow[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rpow, x)

    def __rlshift__[X, Y](
        self: Functor[opt.CanRLshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rlshift, x)

    def __rrshift__[X, Y](
        self: Functor[opt.CanRRshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rrshift, x)

    def __rand__[X, Y](
        self: Functor[opt.CanRAnd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rand, x)

    def __rxor__[X, Y](
        self: Functor[opt.CanRXor[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_rxor, x)

    def __ror__[X, Y](
        self: Functor[opt.CanROr[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(opt.do_ror, x)
