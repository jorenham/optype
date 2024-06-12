import math as _math
import operator as _o
import sys as _sys
from typing import TYPE_CHECKING, ParamSpec, TypeVar, overload

import optype._can as _c
import optype._does as _d


# type conversion
do_bool: _d.DoesBool = bool  # pyright: ignore[reportAssignmentType]
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
do_ne: _d.DoesNe = _o.ne
do_gt: _d.DoesGt = _o.gt
do_ge: _d.DoesGe = _o.ge


# attributes
do_getattr: _d.DoesGetattr = getattr
do_setattr: _d.DoesSetattr = setattr
do_delattr: _d.DoesDelattr = delattr
do_dir: _d.DoesDir = dir


# callables

if _sys.version_info < (3, 11):
    _Pss_call = ParamSpec('_Pss_call')
    _R_call = TypeVar('_R_call')

    def do_call(
        f: _c.CanCall[_Pss_call, _R_call],
        /,
        *args: _Pss_call.args,
        **kwargs: _Pss_call.kwargs,
    ) -> _R_call:
        return f(*args, **kwargs)
else:
    do_call: _d.DoesCall = _o.call


# containers and sequences

do_len: _d.DoesLen = len
do_length_hint: _d.DoesLengthHint = _o.length_hint


# `operator.getitem` isn't used, because it has an (unreasonably loose, and
# redundant) overload for `(Sequence[T], slice) -> Sequence[T]`
# https://github.com/python/typeshed/blob/587ad6b/stdlib/_operator.pyi#L84-L86

_K_getitem = TypeVar('_K_getitem')
_V_getitem = TypeVar('_V_getitem')
_D_getitem = TypeVar('_D_getitem')


@overload
def do_getitem(
    obj:  _c.CanGetMissing[_K_getitem, _V_getitem, _D_getitem],
    key: _K_getitem,
    /,
) -> _V_getitem | _D_getitem: ...
@overload
def do_getitem(
    obj:  _c.CanGetitem[_K_getitem, _V_getitem],
    key: _K_getitem,
    /,
) -> _V_getitem: ...
def do_getitem(
    obj:  (
        _c.CanGetitem[_K_getitem, _V_getitem]
        | _c.CanGetMissing[_K_getitem, _V_getitem, _D_getitem]
    ),
    key: _K_getitem,
    /,
) -> _V_getitem | _D_getitem:
    """Same as `value = obj[key]`."""
    return obj[key]


_K_setitem = TypeVar('_K_setitem')
_V_setitem = TypeVar('_V_setitem')


def do_setitem(
    obj: _c.CanSetitem[_K_setitem, _V_setitem],
    key: _K_setitem,
    value: _V_setitem,
    /,
) -> None:
    """Same as `obj[key] = value`."""
    obj[key] = value


_K_delitem = TypeVar('_K_delitem')


def do_delitem(obj: _c.CanDelitem[_K_delitem], key: _K_delitem, /) -> None:
    """Same as `del obj[key]`."""
    del obj[key]


_K_missing = TypeVar('_K_missing')
_D_missing = TypeVar('_D_missing')


def do_missing(
    obj: _c.CanMissing[_K_missing, _D_missing],
    key: _K_missing,
    /,
) -> _D_missing:
    return obj.__missing__(key)


# `operator.contains` cannot be used, as it incorrectly requires `key`
# to be an **invariant** `object` instance...
_K_contains = TypeVar('_K_contains', bound=object)


def do_contains(obj: _c.CanContains[_K_contains], key: _K_contains, /) -> bool:
    """Same as `key in obj`."""
    return key in obj


# `builtins.reversed` is annotated incorrectly within typeshed:
# https://github.com/python/typeshed/issues/11645
do_reversed: _d.DoesReversed = reversed  # pyright: ignore[reportAssignmentType]


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


_T_radd = TypeVar('_T_radd')
_R_radd = TypeVar('_R_radd')


def do_radd(a: _c.CanRAdd[_T_radd, _R_radd], b: _T_radd, /) -> _R_radd:
    """Same as `b + a`."""
    return b + a


_T_rsub = TypeVar('_T_rsub')
_R_rsub = TypeVar('_R_rsub')


def do_rsub(a: _c.CanRSub[_T_rsub, _R_rsub], b: _T_rsub, /) -> _R_rsub:
    """Same as `b - a`."""
    return b - a


_T_rmul = TypeVar('_T_rmul')
_R_rmul = TypeVar('_R_rmul')


def do_rmul(a: _c.CanRMul[_T_rmul, _R_rmul], b: _T_rmul, /) -> _R_rmul:
    """Same as `b * a`."""
    return b * a


_T_rmatmul = TypeVar('_T_rmatmul')
_R_rmatmul = TypeVar('_R_rmatmul')


def do_rmatmul(
    a: _c.CanRMatmul[_T_rmatmul, _R_rmatmul],
    b: _T_rmatmul,
    /,
) -> _R_rmatmul:
    """Same as `b @ a`."""
    return b @ a


_T_rtruediv = TypeVar('_T_rtruediv')
_R_rtruediv = TypeVar('_R_rtruediv')


def do_rtruediv(
    a: _c.CanRTruediv[_T_rtruediv, _R_rtruediv],
    b: _T_rtruediv,
    /,
) -> _R_rtruediv:
    """Same as `b / a`."""
    return b / a


_T_rfloordiv = TypeVar('_T_rfloordiv')
_R_rfloordiv = TypeVar('_R_rfloordiv')


def do_rfloordiv(
    a: _c.CanRFloordiv[_T_rfloordiv, _R_rfloordiv],
    b: _T_rfloordiv,
    /,
) -> _R_rfloordiv:
    """Same as `b // a`."""
    return b // a


_T_rmod = TypeVar('_T_rmod')
_R_rmod = TypeVar('_R_rmod')


def do_rmod(a: _c.CanRMod[_T_rmod, _R_rmod], b: _T_rmod, /) -> _R_rmod:
    """Same as `b % a`."""
    return b % a


_T_rdivmod = TypeVar('_T_rdivmod')
_R_rdivmod = TypeVar('_R_rdivmod')


def do_rdivmod(
    a: _c.CanRDivmod[_T_rdivmod, _R_rdivmod],
    b: _T_rdivmod,
    /,
) -> _R_rdivmod:
    """Same as `divmod(b, a)`."""
    return divmod(b, a)


_T_rpow = TypeVar('_T_rpow')
_R_rpow = TypeVar('_R_rpow')


def do_rpow(a: _c.CanRPow[_T_rpow, _R_rpow], b: _T_rpow, /) -> _R_rpow:
    """Same as `b ** a`."""
    return b**a


_T_rlshift = TypeVar('_T_rlshift')
_R_rlshift = TypeVar('_R_rlshift')


def do_rlshift(
    a: _c.CanRLshift[_T_rlshift, _R_rlshift],
    b: _T_rlshift,
    /,
) -> _R_rlshift:
    """Same as `b << a`."""
    return b << a


_T_rrshift = TypeVar('_T_rrshift')
_R_rrshift = TypeVar('_R_rrshift')


def do_rrshift(
    a: _c.CanRRshift[_T_rrshift, _R_rrshift],
    b: _T_rrshift,
    /,
) -> _R_rrshift:
    """Same as `b >> a`."""
    return b >> a


_T_rand = TypeVar('_T_rand')
_R_rand = TypeVar('_R_rand')


def do_rand(a: _c.CanRAnd[_T_rand, _R_rand], b: _T_rand, /) -> _R_rand:
    """Same as `b & a`."""
    return b & a


_T_rxor = TypeVar('_T_rxor')
_R_rxor = TypeVar('_R_rxor')


def do_rxor(a: _c.CanRXor[_T_rxor, _R_rxor], b: _T_rxor, /) -> _R_rxor:
    """Same as `b ^ a`."""
    return b ^ a


_T_ror = TypeVar('_T_ror')
_R_ror = TypeVar('_R_ror')


def do_ror(a: _c.CanROr[_T_ror, _R_ror], b: _T_ror, /) -> _R_ror:
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
