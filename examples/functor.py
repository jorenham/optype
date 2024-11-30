from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Generic, TypeVar, final
import optype as op

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import NotImplementedType


_T_co = TypeVar("_T_co", covariant=True)
_X = TypeVar("_X")
_Y = TypeVar("_Y")


@final
class Functor(Generic[_T_co]):
    __match_args__ = __slots__ = ("value",)

    def __init__(self, value: _T_co, /) -> None:
        self.value = value

    def map1(self, f: Callable[[_T_co], _Y], /) -> Functor[_Y]:
        """
        Applies a unary operator `f` over the value of `self`,
        and return a new `Functor`.
        """
        return Functor(f(self.value))

    def map2(
        self,
        f: Callable[[_T_co, _X], _Y],
        other: Functor[_X],
        /,
    ) -> Functor[_Y] | NotImplementedType:
        """
        Maps the binary operator `f: (T, X) -> Y` over `self: Functor[T]` and
        `other: Functor[X]`, and returns `Functor[Y]`. A `NotImplemented`
        is returned if `f` is not supported for the types, or if other is not
        a `Functor`.
        """
        y = f(self.value, other.value)

        return NotImplemented if y is NotImplemented else Functor(y)

    @override
    def __repr__(self, /) -> str:
        return f"{type(self).__name__}({self.value!r})"

    @override
    def __hash__(self: Functor[op.CanHash], /) -> int:
        return op.do_hash(self.value)

    # unary prefix ops

    def __neg__(self: Functor[op.CanNeg[_Y]], /) -> Functor[_Y]:
        """
        >>> -Functor(3.14)
        Functor(-3.14)
        """
        return self.map1(op.do_neg)

    def __pos__(self: Functor[op.CanPos[_Y]], /) -> Functor[_Y]:
        """
        >>> +Functor(True)
        Functor(1)
        """
        return self.map1(op.do_pos)

    def __invert__(self: Functor[op.CanInvert[_Y]], /) -> Functor[_Y]:
        """
        >>> ~Functor(0)
        Functor(-1)
        """
        return self.map1(op.do_invert)

    # rich comparison ops

    def __lt__(
        self: Functor[op.CanLt[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0}) < Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) < Functor((1, -1))
        Functor(True)
        """
        return self.map2(op.do_lt, x)

    def __le__(
        self: Functor[op.CanLe[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0}) <= Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) <= Functor((1, -1))
        Functor(True)
        """
        return self.map2(op.do_le, x)

    def __gt__(self, x: Functor[op.CanLt[_T_co, _Y]], /) -> Functor[_Y]:
        """
        >>> Functor({0, 1}) > Functor({0})
        Functor(True)
        >>> Functor((0, 1)) > Functor((1, -1))
        Functor(False)
        """
        return self.map2(op.do_gt, x)

    def __ge__(self, x: Functor[op.CanLe[_T_co, _Y]], /) -> Functor[_Y]:
        """
        >>> Functor({0, 1}) >= Functor({0})
        Functor(True)
        >>> Functor((0, 1)) >= Functor((1, -1))
        Functor(False)
        """
        return self.map2(op.do_ge, x)

    # binary infix ops

    def __add__(
        self: Functor[op.CanAdd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(0) + Functor(1)
        Functor(1)
        >>> Functor(('spam',)) + Functor(('ham',)) + Functor(('eggs',))
        Functor(('spam', 'ham', 'eggs'))
        """
        return self.map2(op.do_add, x)

    def __sub__(
        self: Functor[op.CanSub[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(0) - Functor(1)
        Functor(-1)
        """
        return self.map2(op.do_sub, x)

    def __mul__(
        self: Functor[op.CanMul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(('Developers!',)) * Functor(4)
        Functor(('Developers!', 'Developers!', 'Developers!', 'Developers!'))
        """
        return self.map2(op.do_mul, x)

    def __matmul__(
        self: Functor[op.CanMatmul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(op.do_matmul, x)

    def __truediv__(
        self: Functor[op.CanTruediv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) / Functor(2)
        Functor(0.5)
        """
        return self.map2(op.do_truediv, x)

    def __floordiv__(
        self: Functor[op.CanFloordiv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) // Functor(2)
        Functor(0)
        """
        return self.map2(op.do_floordiv, x)

    def __mod__(
        self: Functor[op.CanMod[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(10) % Functor(7)
        Functor(3)
        """
        return self.map2(op.do_mod, x)

    def __pow__(
        self: Functor[op.CanPow2[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(2) ** Functor(3)
        Functor(8)
        """
        return self.map2(op.do_pow, x)

    def __lshift__(
        self: Functor[op.CanLshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) << Functor(10)
        Functor(1024)
        """
        return self.map2(op.do_lshift, x)

    def __rshift__(
        self: Functor[op.CanRshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1024) >> Functor(4)
        Functor(64)
        """
        return self.map2(op.do_rshift, x)

    def __and__(
        self: Functor[op.CanAnd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) & Functor(False)
        Functor(False)
        >>> Functor(3) & Functor(7)
        Functor(3)
        """
        return self.map2(op.do_and, x)

    def __xor__(
        self: Functor[op.CanXor[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) ^ Functor(False)
        Functor(True)
        >>> Functor(3) ^ Functor(7)
        Functor(4)
        """
        return self.map2(op.do_xor, x)

    def __or__(
        self: Functor[op.CanOr[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) | Functor(False)
        Functor(True)
        >>> Functor(3) | Functor(7)
        Functor(7)
        """
        return self.map2(op.do_or, x)
