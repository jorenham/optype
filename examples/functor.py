from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final
import optype as opt

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import NotImplementedType


_T_co = TypeVar('_T_co', covariant=True)
_X = TypeVar('_X')
_Y = TypeVar('_Y')


@final
class Functor(Generic[_T_co]):
    __match_args__ = __slots__ = ('value',)

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
        other: Functor[_X] | Any,
        /,
    ) -> Functor[_Y] | NotImplementedType:
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
    def __repr__(self, /) -> str:
        return f'{type(self).__name__}({self.value!r})'

    @override
    def __hash__(self: Functor[opt.CanHash], /) -> int:
        return opt.do_hash(self.value)

    # unary prefix ops

    def __neg__(self: Functor[opt.CanNeg[_Y]], /) -> Functor[_Y]:
        """
        >>> -Functor(3.14)
        Functor(-3.14)
        """
        return self.map1(opt.do_neg)

    def __pos__(self: Functor[opt.CanPos[_Y]], /) -> Functor[_Y]:
        """
        >>> +Functor(True)
        Functor(1)
        """
        return self.map1(opt.do_pos)

    def __invert__(self: Functor[opt.CanInvert[_Y]], /) -> Functor[_Y]:
        """
        >>> ~Functor(0)
        Functor(-1)
        """
        return self.map1(opt.do_invert)

    # rich comparison ops

    def __lt__(
        self: Functor[opt.CanLt[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0}) < Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) < Functor((1, -1))
        Functor(True)
        """
        return self.map2(opt.do_lt, x)

    def __le__(
        self: Functor[opt.CanLe[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0}) <= Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) <= Functor((1, -1))
        Functor(True)
        """
        return self.map2(opt.do_le, x)

    @override
    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Functor[opt.CanEq[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
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
    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Functor[opt.CanNe[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(object()) != Functor(object())
        Functor(True)
        >>> Functor(0) != Functor(0)
        Functor(False)
        >>> Functor(0) != 0
        True
        """
        return self.map2(opt.do_ne, x)

    def __gt__(
        self: Functor[opt.CanGt[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0, 1}) > Functor({0})
        Functor(True)
        >>> Functor((0, 1)) > Functor((1, -1))
        Functor(False)
        """
        return self.map2(opt.do_gt, x)

    def __ge__(
        self: Functor[opt.CanGe[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0, 1}) >= Functor({0})
        Functor(True)
        >>> Functor((0, 1)) >= Functor((1, -1))
        Functor(False)
        """
        return self.map2(opt.do_ge, x)

    # binary infix ops

    def __add__(
        self: Functor[opt.CanAdd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(0) + Functor(1)
        Functor(1)
        >>> Functor(('spam',)) + Functor(('ham',)) + Functor(('eggs',))
        Functor(('spam', 'ham', 'eggs'))
        """
        return self.map2(opt.do_add, x)

    def __sub__(
        self: Functor[opt.CanSub[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(0) - Functor(1)
        Functor(-1)
        """
        return self.map2(opt.do_sub, x)

    def __mul__(
        self: Functor[opt.CanMul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(('Developers!',)) * Functor(4)
        Functor(('Developers!', 'Developers!', 'Developers!', 'Developers!'))
        """
        return self.map2(opt.do_mul, x)

    def __matmul__(
        self: Functor[opt.CanMatmul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_matmul, x)

    def __truediv__(
        self: Functor[opt.CanTruediv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) / Functor(2)
        Functor(0.5)
        """
        return self.map2(opt.do_truediv, x)

    def __floordiv__(
        self: Functor[opt.CanFloordiv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) // Functor(2)
        Functor(0)
        """
        return self.map2(opt.do_floordiv, x)

    def __mod__(
        self: Functor[opt.CanMod[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(10) % Functor(7)
        Functor(3)
        """
        return self.map2(opt.do_mod, x)

    def __pow__(
        self: Functor[opt.CanPow2[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(2) ** Functor(3)
        Functor(8)
        """
        return self.map2(opt.do_pow, x)

    def __lshift__(
        self: Functor[opt.CanLshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) << Functor(10)
        Functor(1024)
        """
        return self.map2(opt.do_lshift, x)

    def __rshift__(
        self: Functor[opt.CanRshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1024) >> Functor(4)
        Functor(64)
        """
        return self.map2(opt.do_rshift, x)

    def __and__(
        self: Functor[opt.CanAnd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) & Functor(False)
        Functor(False)
        >>> Functor(3) & Functor(7)
        Functor(3)
        """
        return self.map2(opt.do_and, x)

    def __xor__(
        self: Functor[opt.CanXor[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) ^ Functor(False)
        Functor(True)
        >>> Functor(3) ^ Functor(7)
        Functor(4)
        """
        return self.map2(opt.do_xor, x)

    def __or__(
        self: Functor[opt.CanOr[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) | Functor(False)
        Functor(True)
        >>> Functor(3) | Functor(7)
        Functor(7)
        """
        return self.map2(opt.do_or, x)

    # binary reflected infix ops

    def __radd__(
        self: Functor[opt.CanRAdd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_radd, x)

    def __rsub__(
        self: Functor[opt.CanRSub[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rsub, x)

    def __rmul__(
        self: Functor[opt.CanRMul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rmul, x)

    def __rmatmul__(
        self: Functor[opt.CanRMatmul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rmatmul, x)

    def __rtruediv__(
        self: Functor[opt.CanRTruediv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rtruediv, x)

    def __rfloordiv__(
        self: Functor[opt.CanRFloordiv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rfloordiv, x)

    def __rmod__(
        self: Functor[opt.CanRMod[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rmod, x)

    def __rpow__(
        self: Functor[opt.CanRPow[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rpow, x)

    def __rlshift__(
        self: Functor[opt.CanRLshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rlshift, x)

    def __rrshift__(
        self: Functor[opt.CanRRshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rrshift, x)

    def __rand__(
        self: Functor[opt.CanRAnd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rand, x)

    def __rxor__(
        self: Functor[opt.CanRXor[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_rxor, x)

    def __ror__(
        self: Functor[opt.CanROr[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(opt.do_ror, x)
