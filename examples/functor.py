from __future__ import annotations

from typing import TYPE_CHECKING, Any, final, override
import optype

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import NotImplementedType


@final  # noqa: PLR0904
class Functor[T]:
    __match_args__ = __slots__ = ('value',)

    def __init__(self, value: T, /) -> None:
        self.value = value

    def map1[Y](self, f: Callable[[T], Y]) -> Functor[Y]:
        """
        Applies a unary operator `f` over the value of `self`,
        and return a new `Functor`.
        """
        return Functor(f(self.value))

    def map2[X, Y](
        self,
        f: Callable[[T, X], Y],
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
    def __hash__(self: Functor[optype.CanHash]) -> int:
        return optype.do_hash(self.value)

    # unary prefix ops

    def __neg__[Y](self: Functor[optype.CanNeg[Y]]) -> Functor[Y]:
        """
        >>> -Functor(3.14)
        Functor(-3.14)
        """
        return self.map1(optype.do_neg)

    def __pos__[Y](self: Functor[optype.CanPos[Y]]) -> Functor[Y]:
        """
        >>> +Functor(True)
        Functor(1)
        """
        return self.map1(optype.do_pos)

    def __invert__[Y](self: Functor[optype.CanInvert[Y]]) -> Functor[Y]:
        """
        >>> ~Functor(0)
        Functor(-1)
        """
        return self.map1(optype.do_invert)

    # rich comparison ops

    def __lt__[X, Y](
        self: Functor[optype.CanLt[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0}) < Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) < Functor((1, -1))
        Functor(True)
        """
        return self.map2(optype.do_lt, x)

    def __le__[X, Y](
        self: Functor[optype.CanLe[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0}) <= Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) <= Functor((1, -1))
        Functor(True)
        """
        return self.map2(optype.do_le, x)

    @override
    def __eq__[X, Y](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Functor[optype.CanEq[X, Y]],
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
        return self.map2(optype.do_eq, x)

    @override
    def __ne__[X, Y](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Functor[optype.CanNe[X, Y]],
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
        return self.map2(optype.do_ne, x)

    def __gt__[X, Y](
        self: Functor[optype.CanGt[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0, 1}) > Functor({0})
        Functor(True)
        >>> Functor((0, 1)) > Functor((1, -1))
        Functor(False)
        """
        return self.map2(optype.do_gt, x)

    def __ge__[X, Y](
        self: Functor[optype.CanGe[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor({0, 1}) >= Functor({0})
        Functor(True)
        >>> Functor((0, 1)) >= Functor((1, -1))
        Functor(False)
        """
        return self.map2(optype.do_ge, x)

    # binary infix ops

    def __add__[X, Y](
        self: Functor[optype.CanAdd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(0) + Functor(1)
        Functor(1)
        >>> Functor(('spam',)) + Functor(('ham',)) + Functor(('eggs',))
        Functor(('spam', 'ham', 'eggs'))
        """
        return self.map2(optype.do_add, x)

    def __sub__[X, Y](
        self: Functor[optype.CanSub[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(0) - Functor(1)
        Functor(-1)
        """
        return self.map2(optype.do_sub, x)

    def __mul__[X, Y](
        self: Functor[optype.CanMul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(('Developers!',)) * Functor(4)
        Functor(('Developers!', 'Developers!', 'Developers!', 'Developers!'))
        """
        return self.map2(optype.do_mul, x)

    def __matmul__[X, Y](
        self: Functor[optype.CanMatmul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_matmul, x)

    def __truediv__[X, Y](
        self: Functor[optype.CanTruediv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1) / Functor(2)
        Functor(0.5)
        """
        return self.map2(optype.do_truediv, x)

    def __floordiv__[X, Y](
        self: Functor[optype.CanFloordiv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1) // Functor(2)
        Functor(0)
        """
        return self.map2(optype.do_floordiv, x)

    def __mod__[X, Y](
        self: Functor[optype.CanMod[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(10) % Functor(7)
        Functor(3)
        """
        return self.map2(optype.do_mod, x)

    def __pow__[X, Y](
        self: Functor[optype.CanPow2[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(2) ** Functor(3)
        Functor(8)
        """
        return self.map2(optype.do_pow, x)

    def __lshift__[X, Y](
        self: Functor[optype.CanLshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1) << Functor(10)
        Functor(1024)
        """
        return self.map2(optype.do_lshift, x)

    def __rshift__[X, Y](
        self: Functor[optype.CanRshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(1024) >> Functor(4)
        Functor(64)
        """
        return self.map2(optype.do_rshift, x)

    def __and__[X, Y](
        self: Functor[optype.CanAnd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(True) & Functor(False)
        Functor(False)
        >>> Functor(3) & Functor(7)
        Functor(3)
        """
        return self.map2(optype.do_and, x)

    def __xor__[X, Y](
        self: Functor[optype.CanXor[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(True) ^ Functor(False)
        Functor(True)
        >>> Functor(3) ^ Functor(7)
        Functor(4)
        """
        return self.map2(optype.do_xor, x)

    def __or__[X, Y](
        self: Functor[optype.CanOr[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        """
        >>> Functor(True) | Functor(False)
        Functor(True)
        >>> Functor(3) | Functor(7)
        Functor(7)
        """
        return self.map2(optype.do_or, x)

    # binary reflected infix ops

    def __radd__[X, Y](
        self: Functor[optype.CanRAdd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_radd, x)

    def __rsub__[X, Y](
        self: Functor[optype.CanRSub[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rsub, x)

    def __rmul__[X, Y](
        self: Functor[optype.CanRMul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rmul, x)

    def __rmatmul__[X, Y](
        self: Functor[optype.CanRMatmul[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rmatmul, x)

    def __rtruediv__[X, Y](
        self: Functor[optype.CanRTruediv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rtruediv, x)

    def __rfloordiv__[X, Y](
        self: Functor[optype.CanRFloordiv[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rfloordiv, x)

    def __rmod__[X, Y](
        self: Functor[optype.CanRMod[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rmod, x)

    def __rpow__[X, Y](
        self: Functor[optype.CanRPow[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rpow, x)

    def __rlshift__[X, Y](
        self: Functor[optype.CanRLshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rlshift, x)

    def __rrshift__[X, Y](
        self: Functor[optype.CanRRshift[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rrshift, x)

    def __rand__[X, Y](
        self: Functor[optype.CanRAnd[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rand, x)

    def __rxor__[X, Y](
        self: Functor[optype.CanRXor[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_rxor, x)

    def __ror__[X, Y](
        self: Functor[optype.CanROr[X, Y]],
        x: Functor[X],
    ) -> Functor[Y]:
        return self.map2(optype.do_ror, x)
