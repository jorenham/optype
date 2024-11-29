# mypy: disable-error-code="assignment"
# pyright: reportInvalidCast=false

from __future__ import annotations

import math
import operator as _o
import sys
from typing import TYPE_CHECKING, Final, Literal, ParamSpec, TypeVar, cast, overload

from ._utils import set_module


if TYPE_CHECKING:
    from collections.abc import Callable

    from . import _can as _c, _does as _d


# type conversion
do_bool: Final = cast("_d.DoesBool", bool)
do_int: Final = cast("_d.DoesInt", int)
do_float: Final = cast("_d.DoesFloat", float)
do_complex: Final = cast("_d.DoesComplex", complex)
do_bytes: Final = cast("_d.DoesBytes", bytes)
do_str: Final = cast("_d.DoesStr", str)

# formatting
do_repr: Final = cast("_d.DoesRepr", repr)
do_format: Final = cast("_d.DoesFormat", format)

# iteration
do_next: Final = cast("_d.DoesNext", next)
do_iter: Final = cast("_d.DoesIter", iter)

# async iteration
do_anext: Final = cast("_d.DoesANext", anext)
do_aiter: Final = cast("_d.DoesAIter", aiter)

# rich comparison
do_lt: Final = cast("_d.DoesLt", _o.lt)
do_le: Final = cast("_d.DoesLe", _o.le)
do_eq: Final = cast("_d.DoesEq", _o.eq)
do_ne: Final = cast("_d.DoesNe", _o.ne)
do_gt: Final = cast("_d.DoesGt", _o.gt)
do_ge: Final = cast("_d.DoesGe", _o.ge)

# attributes
do_getattr: Final = cast("_d.DoesGetattr", getattr)
do_setattr: Final = cast("_d.DoesSetattr", setattr)
do_delattr: Final = cast("_d.DoesDelattr", delattr)
do_dir: Final = cast("_d.DoesDir", dir)

# callables

if sys.version_info >= (3, 11):
    do_call: Final = cast("_d.DoesCall", _o.call)
else:
    _Pss = ParamSpec("_Pss")
    _R = TypeVar("_R")

    @set_module("optype")
    def do_call(f: Callable[_Pss, _R], /, *args: _Pss.args, **kw: _Pss.kwargs) -> _R:
        return f(*args, **kw)

# containers and sequences

do_len: Final = cast("_d.DoesLen", len)
do_length_hint: Final = cast("_d.DoesLengthHint", _o.length_hint)


# `operator.getitem` isn't used, because it has an (unreasonably loose, and
# redundant) overload for `(Sequence[T], slice) -> Sequence[T]`
# https://github.com/python/typeshed/blob/587ad6b/stdlib/_operator.pyi#L84-L86

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_DT = TypeVar("_DT")


@overload
@set_module("optype")
def do_getitem(obj: _c.CanGetMissing[_KT, _VT, _DT], key: _KT, /) -> _VT | _DT: ...
@overload
@set_module("optype")
def do_getitem(obj: _c.CanGetitem[_KT, _VT], key: _KT, /) -> _VT: ...
@set_module("optype")
def do_getitem(
    obj: _c.CanGetitem[_KT, _VT] | _c.CanGetMissing[_KT, _VT, _DT],
    key: _KT,
    /,
) -> _VT | _DT:
    """Same as `value = obj[key]`."""
    return obj[key]


@set_module("optype")
def do_setitem(obj: _c.CanSetitem[_KT, _VT], key: _KT, value: _VT, /) -> None:
    """Same as `obj[key] = value`."""
    obj[key] = value


@set_module("optype")
def do_delitem(obj: _c.CanDelitem[_KT], key: _KT, /) -> None:
    """Same as `del obj[key]`."""
    del obj[key]


@set_module("optype")
def do_missing(obj: _c.CanMissing[_KT, _DT], key: _KT, /) -> _DT:
    return obj.__missing__(key)


_BoolT = TypeVar("_BoolT", Literal[False], Literal[True], bool)


# `operator.contains` cannot be used, as it incorrectly requires `key`
# to be exactly of type `object`, so that it only accepts `object()`...
@set_module("optype")
def do_contains(obj: _c.CanContains[_KT, _BoolT], key: _KT, /) -> _BoolT:
    """Same as `key in obj`."""
    return cast("_BoolT", key in obj)  # type: ignore[redundant-cast]


# `builtins.reversed` is annotated incorrectly within typeshed:
# https://github.com/python/typeshed/issues/11645
do_reversed: Final = cast("_d.DoesReversed", reversed)


# infix ops
do_add: Final = cast("_d.DoesAdd", _o.add)
do_sub: Final = cast("_d.DoesSub", _o.sub)
do_mul: Final = cast("_d.DoesMul", _o.mul)
do_matmul: Final = cast("_d.DoesMatmul", _o.matmul)
do_truediv: Final = cast("_d.DoesTruediv", _o.truediv)
do_floordiv: Final = cast("_d.DoesFloordiv", _o.floordiv)
do_mod: Final = cast("_d.DoesMod", _o.mod)
do_divmod: Final = cast("_d.DoesDivmod", divmod)
do_pow: Final = cast("_d.DoesPow", pow)
do_lshift: Final = cast("_d.DoesLshift", _o.lshift)
do_rshift: Final = cast("_d.DoesRshift", _o.rshift)
do_and: Final = cast("_d.DoesAnd", _o.and_)
do_xor: Final = cast("_d.DoesXor", _o.xor)
do_or: Final = cast("_d.DoesOr", _o.or_)


# reflected ops
# (a DRY `do_r*` decorator won't work; the overloads get lost during casting to
# `CanCall` or `Callable`, within the decorator function signature).


_LeftT = TypeVar("_LeftT")
_OutT = TypeVar("_OutT")


@set_module("optype")
def do_radd(a: _c.CanRAdd[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b + a`."""
    return b + a


@set_module("optype")
def do_rsub(a: _c.CanRSub[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b - a`."""
    return b - a


@set_module("optype")
def do_rmul(a: _c.CanRMul[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b * a`."""
    return b * a


@set_module("optype")
def do_rmatmul(a: _c.CanRMatmul[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b @ a`."""
    return b @ a


@set_module("optype")
def do_rtruediv(a: _c.CanRTruediv[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b / a`."""
    return b / a


@set_module("optype")
def do_rfloordiv(a: _c.CanRFloordiv[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b // a`."""
    return b // a


@set_module("optype")
def do_rmod(a: _c.CanRMod[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b % a`."""
    return b % a


@set_module("optype")
def do_rdivmod(a: _c.CanRDivmod[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `divmod(b, a)`."""
    return divmod(b, a)


@set_module("optype")
def do_rpow(a: _c.CanRPow[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b ** a`."""
    return b**a


@set_module("optype")
def do_rlshift(a: _c.CanRLshift[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b << a`."""
    return b << a


@set_module("optype")
def do_rrshift(a: _c.CanRRshift[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b >> a`."""
    return b >> a


@set_module("optype")
def do_rand(a: _c.CanRAnd[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b & a`."""
    return b & a


@set_module("optype")
def do_rxor(a: _c.CanRXor[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b ^ a`."""
    return b ^ a


@set_module("optype")
def do_ror(a: _c.CanROr[_LeftT, _OutT], b: _LeftT, /) -> _OutT:
    """Same as `b | a`."""
    return b | a


# augmented ops
do_iadd: Final = cast("_d.DoesIAdd", _o.iadd)
do_isub: Final = cast("_d.DoesISub", _o.isub)
do_imul: Final = cast("_d.DoesIMul", _o.imul)
do_imatmul: Final = cast("_d.DoesIMatmul", _o.imatmul)
do_itruediv: Final = cast("_d.DoesITruediv", _o.itruediv)
do_ifloordiv: Final = cast("_d.DoesIFloordiv", _o.ifloordiv)
do_imod: Final = cast("_d.DoesIMod", _o.imod)
do_ipow: Final = cast("_d.DoesIPow", _o.ipow)
do_ilshift: Final = cast("_d.DoesILshift", _o.ilshift)
do_irshift: Final = cast("_d.DoesIRshift", _o.irshift)
do_iand: Final = cast("_d.DoesIAnd", _o.iand)
do_ixor: Final = cast("_d.DoesIXor", _o.ixor)
do_ior: Final = cast("_d.DoesIOr", _o.ior)

# unary ops
do_neg: Final = cast("_d.DoesNeg", _o.neg)
do_pos: Final = cast("_d.DoesPos", _o.pos)
do_abs: Final = cast("_d.DoesAbs", abs)
do_invert: Final = cast("_d.DoesInvert", _o.invert)

# fingerprinting
do_hash: Final = cast("_d.DoesHash", hash)
do_index: Final = cast("_d.DoesIndex", _o.index)

# rounding
# (the typeshed stubs for `round` are unnecessarily strict)
do_round: Final = cast("_d.DoesRound", round)
do_trunc: Final = cast("_d.DoesTrunc", math.trunc)
do_floor: Final = cast("_d.DoesFloor", math.floor)
do_ceil: Final = cast("_d.DoesCeil", math.ceil)


# type-check the custom ops
# TODO: move these to `tests/do.py`
if TYPE_CHECKING:
    _do_getitem: _d.DoesGetitem = do_getitem
    _do_setitem: _d.DoesSetitem = do_setitem
    _do_delitem: _d.DoesDelitem = do_delitem
    _do_missing: _d.DoesMissing = do_missing
    _do_contains: _d.DoesContains = do_contains

    _do_radd: _d.DoesRAdd = do_radd
    _do_rsub: _d.DoesRSub = do_rsub
    _do_rmul: _d.DoesRMul = do_rmul
    _do_rmatmul: _d.DoesRMatmul = do_rmatmul
    _do_rtruediv: _d.DoesRTruediv = do_rtruediv
    _do_rfloordiv: _d.DoesRFloordiv = do_rfloordiv
    _do_rmod: _d.DoesRMod = do_rmod
    _do_rdivmod: _d.DoesRDivmod = do_rdivmod
    _do_rpow: _d.DoesRPow = do_rpow
    _do_rlshift: _d.DoesRLshift = do_rlshift
    _do_rrshift: _d.DoesRRshift = do_rrshift
    _do_rand: _d.DoesRAnd = do_rand
    _do_rxor: _d.DoesRXor = do_rxor
    _do_ror: _d.DoesROr = do_ror
