from __future__ import annotations

import optype as op

from typing import final, overload, override

from collections.abc import Callable as CanCall
from types import NotImplementedType


@final
class Functor[T_co]:
    __match_args__ = __slots__ = ("value",)

    def __init__(self, value: T_co, /) -> None:
        self.value = value

    def map1[T, Y](self: Functor[T], f: CanCall[[T], Y], /) -> Functor[Y]:
        """
        Applies a unary operator `f` over the value of `self`,
        and return a new `Functor`.
        """
        return Functor(f(self.value))

    @overload
    def map2[T, X](
        self: Functor[T],
        f: CanCall[[T, X], NotImplementedType],
        x: Functor[X],
        /,
    ) -> NotImplementedType: ...
    @overload
    def map2[T, X, Y](
        self: Functor[T],
        f: CanCall[[T, X], Y],
        x: Functor[X],
        /,
    ) -> Functor[Y]: ...
    def map2[T, X, Y](
        self: Functor[T],
        f: CanCall[[T, X], Y],
        x: Functor[X],
        /,
    ) -> Functor[Y] | NotImplementedType:
        """
        Maps the binary operator `f: (T, X) -> Y` over `self: Functor[T]` and
        `other: Functor[X]`, and returns `Functor[Y]`. A `NotImplemented`
        is returned if `f` is not supported for the types, or if other is not
        a `Functor`.
        """
        y = f(self.value, x.value)
        return NotImplemented if y is NotImplemented else Functor(y)

    @override
    def __repr__(self, /) -> str:
        return f"{type(self).__name__}({self.value!r})"

    @override
    def __hash__(self: Functor[op.CanHash], /) -> int:
        return op.do_hash(self.value)

    # unary prefix ops

    def __neg__[Y](self: Functor[op.CanNeg[Y]], /) -> Functor[Y]:
        """
        >>> -Functor(3.14)
        Functor(-3.14)
        """
        return self.map1(op.do_neg)

    def __pos__[Y](self: Functor[op.CanPos[Y]], /) -> Functor[Y]:
        """
        >>> +Functor(True)
        Functor(1)
        """
        return self.map1(op.do_pos)

    def __invert__[Y](self: Functor[op.CanInvert[Y]], /) -> Functor[Y]:
        """
        >>> ~Functor(0)
        Functor(-1)
        """
        return self.map1(op.do_invert)

    # rich comparison ops

    def __lt__[X, Y](self: Functor[op.CanLt[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor({0}) < Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) < Functor((1, -1))
        Functor(True)
        """
        return self.map2(op.do_lt, x)

    def __le__[X, Y](self: Functor[op.CanLe[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor({0}) <= Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) <= Functor((1, -1))
        Functor(True)
        """
        return self.map2(op.do_le, x)

    def __gt__[T, Y](self: Functor[T], x: Functor[op.CanLt[T, Y]], /) -> Functor[Y]:
        """
        >>> Functor({0, 1}) > Functor({0})
        Functor(True)
        >>> Functor((0, 1)) > Functor((1, -1))
        Functor(False)
        """
        return self.map2(op.do_gt, x)

    def __ge__[T, Y](self: Functor[T], x: Functor[op.CanLe[T, Y]], /) -> Functor[Y]:
        """
        >>> Functor({0, 1}) >= Functor({0})
        Functor(True)
        >>> Functor((0, 1)) >= Functor((1, -1))
        Functor(False)
        """
        return self.map2(op.do_ge, x)

    # binary infix ops

    def __add__[X, Y](self: Functor[op.CanAdd[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(0) + Functor(1)
        Functor(1)
        >>> Functor(('spam',)) + Functor(('ham',)) + Functor(('eggs',))
        Functor(('spam', 'ham', 'eggs'))
        """
        return self.map2(op.do_add, x)

    def __sub__[X, Y](self: Functor[op.CanSub[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(0) - Functor(1)
        Functor(-1)
        """
        return self.map2(op.do_sub, x)

    def __mul__[X, Y](self: Functor[op.CanMul[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(('Developers!',)) * Functor(4)
        Functor(('Developers!', 'Developers!', 'Developers!', 'Developers!'))
        """
        return self.map2(op.do_mul, x)

    def __matmul__[X, Y](
        self: Functor[op.CanMatmul[X, Y]],
        x: Functor[X],
        /,
    ) -> Functor[Y]:
        return self.map2(op.do_matmul, x)

    def __truediv__[X, Y](
        self: Functor[op.CanTruediv[X, Y]],
        x: Functor[X],
        /,
    ) -> Functor[Y]:
        """
        >>> Functor(1) / Functor(2)
        Functor(0.5)
        """
        return self.map2(op.do_truediv, x)

    def __floordiv__[X, Y](
        self: Functor[op.CanFloordiv[X, Y]],
        x: Functor[X],
        /,
    ) -> Functor[Y]:
        """
        >>> Functor(1) // Functor(2)
        Functor(0)
        """
        return self.map2(op.do_floordiv, x)

    def __mod__[X, Y](self: Functor[op.CanMod[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(10) % Functor(7)
        Functor(3)
        """
        return self.map2(op.do_mod, x)

    def __pow__[X, Y](self: Functor[op.CanPow2[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(2) ** Functor(3)
        Functor(8)
        """
        return self.map2(op.do_pow, x)

    def __lshift__[X, Y](
        self: Functor[op.CanLshift[X, Y]],
        x: Functor[X],
        /,
    ) -> Functor[Y]:
        """
        >>> Functor(1) << Functor(10)
        Functor(1024)
        """
        return self.map2(op.do_lshift, x)

    def __rshift__[X, Y](
        self: Functor[op.CanRshift[X, Y]],
        x: Functor[X],
        /,
    ) -> Functor[Y]:
        """
        >>> Functor(1024) >> Functor(4)
        Functor(64)
        """
        return self.map2(op.do_rshift, x)

    def __and__[X, Y](self: Functor[op.CanAnd[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(True) & Functor(False)
        Functor(False)
        >>> Functor(3) & Functor(7)
        Functor(3)
        """
        return self.map2(op.do_and, x)

    def __xor__[X, Y](self: Functor[op.CanXor[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(True) ^ Functor(False)
        Functor(True)
        >>> Functor(3) ^ Functor(7)
        Functor(4)
        """
        return self.map2(op.do_xor, x)

    def __or__[X, Y](self: Functor[op.CanOr[X, Y]], x: Functor[X], /) -> Functor[Y]:
        """
        >>> Functor(True) | Functor(False)
        Functor(True)
        >>> Functor(3) | Functor(7)
        Functor(7)
        """
        return self.map2(op.do_or, x)
