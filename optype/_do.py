import math as _math
import operator as _o
from typing import TYPE_CHECKING

import optype._can as _c
import optype._does as _d


# type conversion
do_bool: _d.DoesBool = bool
do_int: _d.DoesInt = int
do_float: _d.DoesFloat = float
do_complex: _d.DoesComplex = complex
do_bytes: _d.DoesBytes = bytes
do_str: _d.DoesStr = str


# formatting
do_repr: _d.DoesRepr = repr
do_format: _d.DoesFormat = format


# iteration
do_next: _d.DoesNext = next
do_iter: _d.DoesIter = iter


# async iteration
# (the typeshed stubs for `round` are unnecessarily strict)
do_anext: _d.DoesANext = anext  # pyright: ignore[reportAssignmentType]
do_aiter: _d.DoesAIter = aiter


# rich comparison
do_lt: _d.DoesLt = _o.lt
do_le: _d.DoesLe = _o.le
do_eq: _d.DoesEq = _o.eq
do_ne: _d.DoesEq = _o.ne
do_gt: _d.DoesGt = _o.gt
do_ge: _d.DoesGe = _o.ge


# attributes
do_getattr: _d.DoesGetattr = getattr
do_setattr: _d.DoesSetattr = setattr
do_delattr: _d.DoesDelattr = delattr
do_dir: _d.DoesDir = dir


# callables

def do_call[**Xs, Y](
    func: _c.CanCall[Xs, Y],
    *args: Xs.args,
    **kwargs: Xs.kwargs,
) -> Y:
    return func(*args, **kwargs)


# containers

def do_getitem[K, V, M](
    obj:  _c.CanGetitem[K, V] | _c.CanGetMissing[K, V, M], key: K, /,
) -> V | M:
    """Same as `value = obj[key]`."""
    return obj[key]


def do_setitem[K, V](obj: _c.CanSetitem[K, V], key: K, value: V, /) -> None:
    """Same as `obj[key] = value`."""
    obj[key] = value


def do_delitem[K](obj: _c.CanDelitem[K], key: K, /) -> None:
    """Same as `del obj[key]`."""
    del obj[key]


def do_contains[K](obj: _c.CanContains[K], key: K, /) -> bool:
    """Same as `key in obj`."""
    return key in obj


# infix ops
do_add: _d.DoesAdd = _o.add
do_sub: _d.DoesSub = _o.sub
do_mul: _d.DoesMul = _o.mul
do_matmul: _d.DoesMatmul = _o.matmul
do_truediv: _d.DoesTruediv = _o.truediv
do_floordiv: _d.DoesFloordiv = _o.floordiv
do_mod: _d.DoesMod = _o.mod
do_divmod: _d.DoesDivmod = divmod
do_pow: _d.DoesPow = pow
do_lshift: _d.DoesLshift = _o.lshift
do_rshift: _d.DoesRshift = _o.rshift
do_and: _d.DoesAnd = _o.and_
do_xor: _d.DoesXor = _o.xor
do_or: _d.DoesOr = _o.or_


# reflected ops
# (a DRY `do_r*` decorator won't work; the overloads get lost during casting to
# `CanCall` or `Callable`, within the decorator function signature).

def do_radd[X, Y](a: _c.CanRAdd[X, Y], b: X) -> Y:
    """Same as `b + a`."""
    return b + a


def do_rsub[X, Y](a: _c.CanRSub[X, Y], b: X) -> Y:
    """Same as `b - a`."""
    return b - a


def do_rmul[X, Y](a: _c.CanRMul[X, Y], b: X) -> Y:
    """Same as `b * a`."""
    return b * a


def do_rmatmul[X, Y](a: _c.CanRMatmul[X, Y], b: X) -> Y:
    """Same as `b @ a`."""
    return b @ a


def do_rtruediv[X, Y](a: _c.CanRTruediv[X, Y], b: X) -> Y:
    """Same as `b / a`."""
    return b / a


def do_rfloordiv[X, Y](a: _c.CanRFloordiv[X, Y], b: X) -> Y:
    """Same as `b // a`."""
    return b // a


def do_rmod[X, Y](a: _c.CanRMod[X, Y], b: X) -> Y:
    """Same as `b % a`."""
    return b % a


def do_rdivmod[X, Y](a: _c.CanRDivmod[X, Y], b: X) -> Y:
    """Same as `divmod(b, a)`."""
    return divmod(b, a)


def do_rpow[X, Y](a: _c.CanRPow[X, Y], b: X) -> Y:
    """Same as `b ** a`."""
    return b ** a


def do_rlshift[X, Y](a: _c.CanRLshift[X, Y], b: X) -> Y:
    """Same as `b << a`."""
    return b << a


def do_rrshift[X, Y](a: _c.CanRRshift[X, Y], b: X) -> Y:
    """Same as `b >> a`."""
    return b >> a


def do_rand[X, Y](a: _c.CanRAnd[X, Y], b: X) -> Y:
    """Same as `b & a`."""
    return b & a


def do_rxor[X, Y](a: _c.CanRXor[X, Y], b: X) -> Y:
    """Same as `b ^ a`."""
    return b ^ a


def do_ror[X, Y](a: _c.CanROr[X, Y], b: X) -> Y:
    """Same as `b | a`."""
    return b | a


# augmented ops
do_iadd: _d.DoesIAdd = _o.iadd
do_isub: _d.DoesISub = _o.isub
do_imul: _d.DoesIMul = _o.imul
do_imatmul: _d.DoesIMatmul = _o.imatmul
do_itruediv: _d.DoesITruediv = _o.itruediv
do_ifloordiv: _d.DoesIFloordiv = _o.ifloordiv
do_imod: _d.DoesIMod = _o.imod
do_ipow: _d.DoesIPow = _o.ipow
do_ilshift: _d.DoesILshift = _o.ilshift
do_irshift: _d.DoesIRshift = _o.irshift
do_iand: _d.DoesIAnd = _o.iand
do_ixor: _d.DoesIXor = _o.ixor
do_ior: _d.DoesIOr = _o.ior


# unary ops
do_neg: _d.DoesNeg = _o.neg
do_pos: _d.DoesPos = _o.pos
do_abs: _d.DoesAbs = abs
do_invert: _d.DoesInvert = _o.invert


# fingerprinting
do_hash: _d.DoesHash = hash
do_index: _d.DoesIndex = _o.index


# rounding
# (the typeshed stubs for `round` are unnecessarily strict)
do_round: _d.DoesRound = round  # pyright: ignore[reportAssignmentType]
do_trunc: _d.DoesTrunc = _math.trunc
do_floor: _d.DoesFloor = _math.floor
do_ceil: _d.DoesCeil = _math.ceil


# type-check the custom ops
if TYPE_CHECKING:
    _do_call: _d.DoesCall = do_call

    _do_getitem: _d.DoesGetitem = do_getitem
    _do_setitem: _d.DoesSetitem = do_setitem
    _do_delitem: _d.DoesDelitem = do_delitem
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
