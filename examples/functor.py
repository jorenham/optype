from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final
import optype as o

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

    def map2(  # type: ignore[no-any-explicit]
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
        if not isinstance(other, Functor):
            return NotImplemented

        other = cast(Functor[_X], other)
        y = f(self.value, other.value)

        return NotImplemented if y is NotImplemented else Functor(y)

    @override
    def __repr__(self, /) -> str:
        return f"{type(self).__name__}({self.value!r})"

    @override
    def __hash__(self: Functor[o.CanHash], /) -> int:
        return o.do_hash(self.value)

    # unary prefix ops

    def __neg__(self: Functor[o.CanNeg[_Y]], /) -> Functor[_Y]:
        """
        >>> -Functor(3.14)
        Functor(-3.14)
        """
        return self.map1(o.do_neg)

    def __pos__(self: Functor[o.CanPos[_Y]], /) -> Functor[_Y]:
        """
        >>> +Functor(True)
        Functor(1)
        """
        return self.map1(o.do_pos)

    def __invert__(self: Functor[o.CanInvert[_Y]], /) -> Functor[_Y]:
        """
        >>> ~Functor(0)
        Functor(-1)
        """
        return self.map1(o.do_invert)

    # rich comparison ops

    def __lt__(
        self: Functor[o.CanLt[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0}) < Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) < Functor((1, -1))
        Functor(True)
        """
        return self.map2(o.do_lt, x)

    def __le__(
        self: Functor[o.CanLe[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor({0}) <= Functor({0, 1})
        Functor(True)
        >>> Functor((0, 1)) <= Functor((1, -1))
        Functor(True)
        """
        return self.map2(o.do_le, x)

    def __gt__(self, x: Functor[o.CanLt[_T_co, _Y]], /) -> Functor[_Y]:
        """
        >>> Functor({0, 1}) > Functor({0})
        Functor(True)
        >>> Functor((0, 1)) > Functor((1, -1))
        Functor(False)
        """
        return self.map2(o.do_gt, x)

    def __ge__(self, x: Functor[o.CanLe[_T_co, _Y]], /) -> Functor[_Y]:
        """
        >>> Functor({0, 1}) >= Functor({0})
        Functor(True)
        >>> Functor((0, 1)) >= Functor((1, -1))
        Functor(False)
        """
        return self.map2(o.do_ge, x)

    # binary infix ops

    def __add__(
        self: Functor[o.CanAdd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(0) + Functor(1)
        Functor(1)
        >>> Functor(('spam',)) + Functor(('ham',)) + Functor(('eggs',))
        Functor(('spam', 'ham', 'eggs'))
        """
        return self.map2(o.do_add, x)

    def __sub__(
        self: Functor[o.CanSub[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(0) - Functor(1)
        Functor(-1)
        """
        return self.map2(o.do_sub, x)

    def __mul__(
        self: Functor[o.CanMul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(('Developers!',)) * Functor(4)
        Functor(('Developers!', 'Developers!', 'Developers!', 'Developers!'))
        """
        return self.map2(o.do_mul, x)

    def __matmul__(
        self: Functor[o.CanMatmul[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        return self.map2(o.do_matmul, x)

    def __truediv__(
        self: Functor[o.CanTruediv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) / Functor(2)
        Functor(0.5)
        """
        return self.map2(o.do_truediv, x)

    def __floordiv__(
        self: Functor[o.CanFloordiv[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) // Functor(2)
        Functor(0)
        """
        return self.map2(o.do_floordiv, x)

    def __mod__(
        self: Functor[o.CanMod[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(10) % Functor(7)
        Functor(3)
        """
        return self.map2(o.do_mod, x)

    def __pow__(
        self: Functor[o.CanPow2[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(2) ** Functor(3)
        Functor(8)
        """
        return self.map2(o.do_pow, x)

    def __lshift__(
        self: Functor[o.CanLshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1) << Functor(10)
        Functor(1024)
        """
        return self.map2(o.do_lshift, x)

    def __rshift__(
        self: Functor[o.CanRshift[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(1024) >> Functor(4)
        Functor(64)
        """
        return self.map2(o.do_rshift, x)

    def __and__(
        self: Functor[o.CanAnd[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) & Functor(False)
        Functor(False)
        >>> Functor(3) & Functor(7)
        Functor(3)
        """
        return self.map2(o.do_and, x)

    def __xor__(
        self: Functor[o.CanXor[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) ^ Functor(False)
        Functor(True)
        >>> Functor(3) ^ Functor(7)
        Functor(4)
        """
        return self.map2(o.do_xor, x)

    def __or__(
        self: Functor[o.CanOr[_X, _Y]],
        x: Functor[_X],
        /,
    ) -> Functor[_Y]:
        """
        >>> Functor(True) | Functor(False)
        Functor(True)
        >>> Functor(3) | Functor(7)
        Functor(7)
        """
        return self.map2(o.do_or, x)
