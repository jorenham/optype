from collections.abc import Callable
from typing import (
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

import optype._can as _c


# iteration


_R_next = TypeVar('_R_next')
_D_next = TypeVar('_D_next')


@final
class DoesNext(Protocol):
    @overload
    def __call__(
        self,
        iterator: _c.CanNext[_R_next],
        /,
    ) -> _R_next: ...
    @overload
    def __call__(
        self,
        iterator: _c.CanNext[_R_next],
        default: _D_next,
        /,
    ) -> _R_next | _D_next: ...


_R_anext = TypeVar('_R_anext')
_D_anext = TypeVar('_D_anext')


@final
class DoesANext(Protocol):
    @overload
    def __call__(
        self,
        aiterator: _c.CanANext[_R_anext],
        /,
    ) -> _R_anext: ...
    @overload
    async def __call__(
        self,
        aiterator: _c.CanANext[_c.CanAwait[_R_anext]],
        default: _D_anext,
        /,
    ) -> _R_anext | _D_anext: ...


_V_iter = TypeVar('_V_iter')
_Z_iter = TypeVar('_Z_iter', bound=object)
_R_iter = TypeVar('_R_iter', bound=_c.CanNext[Any])


@final
class DoesIter(Protocol):
    @overload
    def __call__(
        self,
        iterable: _c.CanIter[_R_iter],
        /,
    ) -> _R_iter: ...
    @overload
    def __call__(
        self,
        indexable: _c.CanGetitem[_c.CanIndex, _V_iter],
        /,
    ) -> _c.CanIterSelf[_V_iter]: ...
    @overload
    def __call__(
        self,
        callable_: _c.CanCall[[], _V_iter | None],
        sentinel: None,
        /,
    ) -> _c.CanIterSelf[_V_iter]: ...
    @overload
    def __call__(
        self,
        callable_: _c.CanCall[[], _V_iter | _Z_iter],
        sentinel: _Z_iter,
        /,
    ) -> _c.CanIterSelf[_V_iter]: ...


_R_aiter = TypeVar('_R_aiter', bound=_c.CanANext[Any])


@final
class DoesAIter(Protocol):
    def __call__(self, iterable: _c.CanAIter[_R_aiter], /) -> _R_aiter: ...


# type conversion

@final
class DoesComplex(Protocol):
    def __call__(self, obj: _c.CanComplex, /) -> complex: ...


@final
class DoesFloat(Protocol):
    def __call__(self, obj: _c.CanFloat, /) -> float: ...


_R_int = TypeVar('_R_int', bound=int)


@final
class DoesInt(Protocol):
    def __call__(self, obj: _c.CanInt[_R_int], /) -> _R_int: ...


# fmt: off
_IsTrue: TypeAlias = Literal[True]
_IsFalse: TypeAlias = Literal[False]
_PosInt: TypeAlias = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
]
# fmt: on
_R_bool = TypeVar('_R_bool', _IsTrue, _IsFalse, bool)


@final
class DoesBool(Protocol):
    @overload
    def __call__(self, obj: _c.CanBool[_R_bool], /) -> _R_bool: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __call__(self, obj: _c.CanLen[Literal[0]], /) -> _IsFalse: ...
    @overload
    def __call__(self, obj: _c.CanLen[_PosInt], /) -> _IsTrue: ...
    @overload
    def __call__(self, obj: _c.CanLen[int], /) -> bool: ...
    @overload
    def __call__(self, obj: object, /) -> _IsTrue: ...


_R_str = TypeVar('_R_str', bound=str)


@final
class DoesStr(Protocol):
    def __call__(self, obj: _c.CanStr[_R_str], /) -> _R_str: ...


_R_bytes = TypeVar('_R_bytes', bound=bytes)


@final
class DoesBytes(Protocol):
    def __call__(self, obj: _c.CanBytes[_R_bytes], /) -> _R_bytes: ...


# formatting


_R_repr = TypeVar('_R_repr', bound=str)


@final
class DoesRepr(Protocol):
    def __call__(self, obj: _c.CanRepr[_R_repr], /) -> _R_repr: ...


_T_format = TypeVar('_T_format', bound=str)
_R_format = TypeVar('_R_format', bound=str)


@final
class DoesFormat(Protocol):
    def __call__(
        self,
        obj: _c.CanFormat[_T_format, _R_format],
        fmt: _T_format = ...,
        /,
    ) -> _R_format: ...


# rich comparison


_T_lt_lhs = TypeVar('_T_lt_lhs')
_T_lt_rhs = TypeVar('_T_lt_rhs')
_R_lt = TypeVar('_R_lt')


@final
class DoesLt(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanLt[_T_lt_rhs, _R_lt],
        rhs: _T_lt_rhs,
        /,
    ) -> _R_lt: ...
    @overload
    def __call__(
        self,
        lhs: _T_lt_lhs,
        rhs: _c.CanGt[_T_lt_lhs, _R_lt],
        /,
    ) -> _R_lt: ...


_T_le_lhs = TypeVar('_T_le_lhs')
_T_le_rhs = TypeVar('_T_le_rhs')
_R_le = TypeVar('_R_le')


@final
class DoesLe(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanLe[_T_le_rhs, _R_le],
        rhs: _T_le_rhs,
        /,
    ) -> _R_le: ...
    @overload
    def __call__(
        self,
        lhs: _T_le_lhs,
        rhs: _c.CanGe[_T_le_lhs, _R_le],
        /,
    ) -> _R_le: ...


_T_eq_lhs = TypeVar('_T_eq_lhs')
_T_eq_rhs = TypeVar('_T_eq_rhs')
_R_eq = TypeVar('_R_eq')


@final
class DoesEq(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanEq[_T_eq_rhs, _R_eq],
        rhs: _T_eq_rhs,
        /,
    ) -> _R_eq: ...
    @overload
    def __call__(
        self,
        lhs: _T_eq_lhs,
        rhs: _c.CanEq[_T_eq_lhs, _R_eq],
        /,
    ) -> _R_eq: ...


_T_ne_lhs = TypeVar('_T_ne_lhs')
_T_ne_rhs = TypeVar('_T_ne_rhs')
_R_ne = TypeVar('_R_ne')


@final
class DoesNe(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanNe[_T_ne_rhs, _R_ne],
        rhs: _T_ne_rhs,
        /,
    ) -> _R_ne: ...
    @overload
    def __call__(
        self,
        lhs: _T_ne_lhs,
        rhs: _c.CanNe[_T_ne_lhs, _R_ne],
        /,
    ) -> _R_ne: ...


_T_gt_lhs = TypeVar('_T_gt_lhs')
_T_gt_rhs = TypeVar('_T_gt_rhs')
_R_gt = TypeVar('_R_gt')


@final
class DoesGt(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanGt[_T_gt_rhs, _R_gt],
        rhs: _T_gt_rhs,
        /,
    ) -> _R_gt: ...
    @overload
    def __call__(
        self,
        lhs: _T_gt_lhs,
        rhs: _c.CanLt[_T_gt_lhs, _R_gt],
        /,
    ) -> _R_gt: ...


_T_ge_lhs = TypeVar('_T_ge_lhs')
_T_ge_rhs = TypeVar('_T_ge_rhs')
_R_ge = TypeVar('_R_ge')


@final
class DoesGe(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanGe[_T_ge_rhs, _R_ge],
        rhs: _T_ge_rhs,
        /,
    ) -> _R_ge: ...
    @overload
    def __call__(
        self,
        lhs: _T_ge_lhs,
        rhs: _c.CanLe[_T_ge_lhs, _R_ge],
        /,
    ) -> _R_ge: ...


# dynamic attribute access


_N_getattr = TypeVar('_N_getattr', bound=str)
_V_getattr = TypeVar('_V_getattr')
_D_getattr = TypeVar('_D_getattr')


@final
class DoesGetattr(Protocol):
    @overload
    def __call__(
        self,
        obj: _c.CanGetattr[_N_getattr, _V_getattr],
        name: _N_getattr,
        /,
    ) -> _V_getattr: ...
    @overload
    def __call__(
        self,
        obj: _c.CanGetattribute[_N_getattr, _V_getattr],
        name: _N_getattr,
        /,
    ) -> _V_getattr: ...
    @overload
    def __call__(
        self,
        obj: _c.CanGetattr[_N_getattr, _V_getattr],
        name: _N_getattr,
        default: _D_getattr,
        /,
    ) -> _V_getattr | _D_getattr: ...
    @overload
    def __call__(
        self,
        obj: _c.CanGetattribute[_N_getattr, _V_getattr],
        name: _N_getattr,
        default: _D_getattr,
        /,
    ) -> _V_getattr | _D_getattr: ...


_N_setattr = TypeVar('_N_setattr', bound=str)
_V_setattr = TypeVar('_V_setattr')


@final
class DoesSetattr(Protocol):
    def __call__(
        self,
        obj: _c.CanSetattr[_N_setattr, _V_setattr],
        name: _N_setattr,
        value: _V_setattr,
        /,
    ) -> None: ...


_N_delattr = TypeVar('_N_delattr', bound=str)


@final
class DoesDelattr(Protocol):
    def __call__(
        self,
        obj: _c.CanDelattr[_N_delattr],
        name: _N_delattr,
        /,
    ) -> None: ...


_R_dir = TypeVar('_R_dir', bound=_c.CanIter[Any])


@final
class DoesDir(Protocol):
    @overload
    def __call__(self, /) -> list[str]: ...
    @overload
    def __call__(self, obj: _c.CanDir[_R_dir], /) -> _R_dir: ...


# callables

_Pss_call = ParamSpec('_Pss_call')
_R_call = TypeVar('_R_call')


@final
class DoesCall(Protocol):
    def __call__(
        self,
        # callable_: _c.CanCall[__Pss_call, _R_call],
        callable_: Callable[_Pss_call, _R_call],
        /,
        *args: _Pss_call.args,
        **kwargs: _Pss_call.kwargs,
    ) -> _R_call: ...


# containers and subscriptable types


_R_len = TypeVar('_R_len', bound=int)


@final
class DoesLen(Protocol):
    def __call__(self, obj: _c.CanLen[_R_len], /) -> _R_len: ...


_R_length_hint = TypeVar('_R_length_hint', bound=int)


@final
class DoesLengthHint(Protocol):
    def __call__(
        self,
        obj: _c.CanLengthHint[_R_length_hint],
        /,
    ) -> _R_length_hint: ...


_K_getitem = TypeVar('_K_getitem')
_V_getitem = TypeVar('_V_getitem')
_D_getitem = TypeVar('_D_getitem')


@final
class DoesGetitem(Protocol):
    def __call__(
        self,
        obj: (
            _c.CanGetitem[_K_getitem, _V_getitem]
            | _c.CanGetMissing[_K_getitem, _V_getitem, _D_getitem]
        ),
        key: _K_getitem,
        /,
    ) -> _V_getitem | _D_getitem: ...


_K_setitem = TypeVar('_K_setitem')
_V_setitem = TypeVar('_V_setitem')


@final
class DoesSetitem(Protocol):
    def __call__(
        self,
        obj: _c.CanSetitem[_K_setitem, _V_setitem],
        key: _K_setitem,
        value: _V_setitem,
        /,
    ) -> None: ...


_K_delitem = TypeVar('_K_delitem')


@final
class DoesDelitem(Protocol):
    def __call__(
        self,
        obj: _c.CanDelitem[_K_delitem],
        key: _K_delitem,
        /,
    ) -> None: ...


_K_missing = TypeVar('_K_missing')
_D_missing = TypeVar('_D_missing')


@final
class DoesMissing(Protocol):
    def __call__(
        self,
        obj: _c.CanMissing[_K_missing, _D_missing],
        key: _K_missing,
        /,
    ) -> _D_missing: ...


_K_contains = TypeVar('_K_contains', bound=object)
_R_contains = TypeVar('_R_contains', Literal[True], Literal[False], bool)


@final
class DoesContains(Protocol):
    def __call__(
        self,
        obj: _c.CanContains[_K_contains, _R_contains],
        key: _K_contains,
        /,
    ) -> _R_contains: ...


_V_reversed = TypeVar('_V_reversed')
_R_reversed = TypeVar('_R_reversed')


@final
class DoesReversed(Protocol):
    """
    This is correct type of `builtins.reversed`.

    Note that typeshed's annotations for `reversed` are completely wrong:
    https://github.com/python/typeshed/issues/11645
    """
    @overload
    def __call__(
        self,
        reversible: _c.CanReversed[_R_reversed],
        /,
    ) -> _R_reversed: ...
    @overload
    def __call__(
        self,
        sequence: _c.CanSequence[_c.CanIndex, _V_reversed],
        /,
    ) -> 'reversed[_V_reversed]': ...


# binary infix operators


_T_add_lhs = TypeVar('_T_add_lhs')
_T_add_rhs = TypeVar('_T_add_rhs')
_R_add = TypeVar('_R_add')


@final
class DoesAdd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanAdd[_T_add_rhs, _R_add],
        rhs: _T_add_rhs,
        /,
    ) -> _R_add: ...
    @overload
    def __call__(
        self,
        lhs: _T_add_lhs,
        rhs: _c.CanRAdd[_T_add_lhs, _R_add],
        /,
    ) -> _R_add: ...


_T_sub_lhs = TypeVar('_T_sub_lhs')
_T_sub_rhs = TypeVar('_T_sub_rhs')
_R_sub = TypeVar('_R_sub')


@final
class DoesSub(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanSub[_T_sub_rhs, _R_sub],
        rhs: _T_sub_rhs,
        /,
    ) -> _R_sub: ...
    @overload
    def __call__(
        self,
        lhs: _T_sub_lhs,
        rhs: _c.CanRSub[_T_sub_lhs, _R_sub],
        /,
    ) -> _R_sub: ...


_T_mul_lhs = TypeVar('_T_mul_lhs')
_T_mul_rhs = TypeVar('_T_mul_rhs')
_R_mul = TypeVar('_R_mul')


@final
class DoesMul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanMul[_T_mul_rhs, _R_mul],
        rhs: _T_mul_rhs,
        /,
    ) -> _R_mul: ...
    @overload
    def __call__(
        self,
        lhs: _T_mul_lhs,
        rhs: _c.CanRMul[_T_mul_lhs, _R_mul],
        /,
    ) -> _R_mul: ...


_T_matmul_lhs = TypeVar('_T_matmul_lhs')
_T_matmul_rhs = TypeVar('_T_matmul_rhs')
_R_matmul = TypeVar('_R_matmul')


@final
class DoesMatmul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanMatmul[_T_matmul_rhs, _R_matmul],
        rhs: _T_matmul_rhs,
        /,
    ) -> _R_matmul: ...
    @overload
    def __call__(
        self,
        lhs: _T_matmul_lhs,
        rhs: _c.CanRMatmul[_T_matmul_lhs, _R_matmul],
        /,
    ) -> _R_matmul: ...


_T_truediv_lhs = TypeVar('_T_truediv_lhs')
_T_truediv_rhs = TypeVar('_T_truediv_rhs')
_R_truediv = TypeVar('_R_truediv')


@final
class DoesTruediv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanTruediv[_T_truediv_rhs, _R_truediv],
        rhs: _T_truediv_rhs,
        /,
    ) -> _R_truediv: ...
    @overload
    def __call__(
        self,
        lhs: _T_truediv_lhs,
        rhs: _c.CanRTruediv[_T_truediv_lhs, _R_truediv],
        /,
    ) -> _R_truediv: ...


_T_floordiv_lhs = TypeVar('_T_floordiv_lhs')
_T_floordiv_rhs = TypeVar('_T_floordiv_rhs')
_R_floordiv = TypeVar('_R_floordiv')


@final
class DoesFloordiv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanFloordiv[_T_floordiv_rhs, _R_floordiv],
        rhs: _T_floordiv_rhs,
        /,
    ) -> _R_floordiv: ...
    @overload
    def __call__(
        self,
        lhs: _T_floordiv_lhs,
        rhs: _c.CanRFloordiv[_T_floordiv_lhs, _R_floordiv],
        /,
    ) -> _R_floordiv: ...


_T_mod_lhs = TypeVar('_T_mod_lhs')
_T_mod_rhs = TypeVar('_T_mod_rhs')
_R_mod = TypeVar('_R_mod')


@final
class DoesMod(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanMod[_T_mod_rhs, _R_mod],
        rhs: _T_mod_rhs,
        /,
    ) -> _R_mod: ...
    @overload
    def __call__(
        self,
        lhs: _T_mod_lhs,
        rhs: _c.CanRMod[_T_mod_lhs, _R_mod],
        /,
    ) -> _R_mod: ...


_T_divmod_lhs = TypeVar('_T_divmod_lhs')
_T_divmod_rhs = TypeVar('_T_divmod_rhs')
_R_divmod = TypeVar('_R_divmod')


@final
class DoesDivmod(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanDivmod[_T_divmod_rhs, _R_divmod],
        rhs: _T_divmod_rhs,
        /,
    ) -> _R_divmod: ...
    @overload
    def __call__(
        self,
        lhs: _T_divmod_lhs,
        rhs: _c.CanRDivmod[_T_divmod_lhs, _R_divmod],
        /,
    ) -> _R_divmod: ...


_T_pow_base = TypeVar('_T_pow_base')
_T_pow_exp = TypeVar('_T_pow_exp')
_T_pow_mod = TypeVar('_T_pow_mod')
_R_pow = TypeVar('_R_pow')


@final
class DoesPow(Protocol):
    @overload
    def __call__(
        self,
        base: _c.CanPow2[_T_pow_exp, _R_pow],
        exp: _T_pow_exp,
        /,
    ) -> _R_pow: ...
    @overload
    def __call__(
        self,
        base: _c.CanPow3[_T_pow_exp, _T_pow_mod, _R_pow],
        exp: _T_pow_exp,
        mod: _T_pow_mod,
        /,
    ) -> _R_pow: ...
    @overload
    def __call__(
        self,
        base: _T_pow_base,
        exp: _c.CanRPow[_T_pow_base, _R_pow],
        /,
    ) -> _R_pow: ...


_T_lshift_lhs = TypeVar('_T_lshift_lhs')
_T_lshift_rhs = TypeVar('_T_lshift_rhs')
_R_lshift = TypeVar('_R_lshift')


@final
class DoesLshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanLshift[_T_lshift_rhs, _R_lshift],
        rhs: _T_lshift_rhs,
        /,
    ) -> _R_lshift: ...
    @overload
    def __call__(
        self,
        lhs: _T_lshift_lhs,
        rhs: _c.CanRLshift[_T_lshift_lhs, _R_lshift],
        /,
    ) -> _R_lshift: ...


_T_rshift_lhs = TypeVar('_T_rshift_lhs')
_T_rshift_rhs = TypeVar('_T_rshift_rhs')
_R_rshift = TypeVar('_R_rshift')


@final
class DoesRshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanRshift[_T_rshift_rhs, _R_rshift],
        rhs: _T_rshift_rhs,
        /,
    ) -> _R_rshift: ...
    @overload
    def __call__(
        self,
        lhs: _T_rshift_lhs,
        rhs: _c.CanRRshift[_T_rshift_lhs, _R_rshift],
        /,
    ) -> _R_rshift: ...


_T_and_lhs = TypeVar('_T_and_lhs')
_T_and_rhs = TypeVar('_T_and_rhs')
_R_and = TypeVar('_R_and')


@final
class DoesAnd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanAnd[_T_and_rhs, _R_and],
        rhs: _T_and_rhs,
        /,
    ) -> _R_and: ...
    @overload
    def __call__(
        self,
        lhs: _T_and_lhs,
        rhs: _c.CanRAnd[_T_and_lhs, _R_and],
        /,
    ) -> _R_and: ...


_T_xor_lhs = TypeVar('_T_xor_lhs')
_T_xor_rhs = TypeVar('_T_xor_rhs')
_R_xor = TypeVar('_R_xor')


@final
class DoesXor(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanXor[_T_xor_rhs, _R_xor],
        rhs: _T_xor_rhs,
        /,
    ) -> _R_xor: ...
    @overload
    def __call__(
        self,
        lhs: _T_xor_lhs,
        rhs: _c.CanRXor[_T_xor_lhs, _R_xor],
        /,
    ) -> _R_xor: ...


_T_or_lhs = TypeVar('_T_or_lhs')
_T_or_rhs = TypeVar('_T_or_rhs')
_R_or = TypeVar('_R_or')


@final
class DoesOr(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanOr[_T_or_rhs, _R_or],
        rhs: _T_or_rhs,
        /,
    ) -> _R_or: ...
    @overload
    def __call__(
        self,
        lhs: _T_or_lhs,
        rhs: _c.CanROr[_T_or_lhs, _R_or],
        /,
    ) -> _R_or: ...


# binary reflected operators


_T_radd = TypeVar('_T_radd')
_R_radd = TypeVar('_R_radd')


@final
class DoesRAdd(Protocol):
    def __call__(
        self,
        lhs: _c.CanRAdd[_T_radd, _R_radd],
        rhs: _T_radd,
        /,
    ) -> _R_radd: ...


_T_rsub = TypeVar('_T_rsub')
_R_rsub = TypeVar('_R_rsub')


@final
class DoesRSub(Protocol):
    def __call__(
        self,
        lhs: _c.CanRSub[_T_rsub, _R_rsub],
        rhs: _T_rsub,
        /,
    ) -> _R_rsub: ...


_T_rmul = TypeVar('_T_rmul')
_R_rmul = TypeVar('_R_rmul')


@final
class DoesRMul(Protocol):
    def __call__(
        self,
        lhs: _c.CanRMul[_T_rmul, _R_rmul],
        rhs: _T_rmul,
        /,
    ) -> _R_rmul: ...


_T_rmatmul = TypeVar('_T_rmatmul')
_R_rmatmul = TypeVar('_R_rmatmul')


@final
class DoesRMatmul(Protocol):
    def __call__(
        self,
        lhs: _c.CanRMatmul[_T_rmatmul, _R_rmatmul],
        rhs: _T_rmatmul,
        /,
    ) -> _R_rmatmul: ...


_T_rtruediv = TypeVar('_T_rtruediv')
_R_rtruediv = TypeVar('_R_rtruediv')


@final
class DoesRTruediv(Protocol):
    def __call__(
        self,
        lhs: _c.CanRTruediv[_T_rtruediv, _R_rtruediv],
        rhs: _T_rtruediv,
        /,
    ) -> _R_rtruediv: ...


_T_rfloordiv = TypeVar('_T_rfloordiv')
_R_rfloordiv = TypeVar('_R_rfloordiv')


@final
class DoesRFloordiv(Protocol):
    def __call__(
        self,
        lhs: _c.CanRFloordiv[_T_rfloordiv, _R_rfloordiv],
        rhs: _T_rfloordiv,
        /,
    ) -> _R_rfloordiv: ...


_T_rmod = TypeVar('_T_rmod')
_R_rmod = TypeVar('_R_rmod')


@final
class DoesRMod(Protocol):
    def __call__(
        self,
        lhs: _c.CanRMod[_T_rmod, _R_rmod],
        rhs: _T_rmod,
        /,
    ) -> _R_rmod: ...


_T_rdivmod = TypeVar('_T_rdivmod')
_R_divmod = TypeVar('_R_divmod')


class DoesRDivmod(Protocol):
    def __call__(
        self,
        lhs: _c.CanRDivmod[_T_rdivmod, _R_divmod],
        rhs: _T_rdivmod,
        /,
    ) -> _R_divmod: ...


_T_rpow = TypeVar('_T_rpow')
_R_rpow = TypeVar('_R_rpow')


@final
class DoesRPow(Protocol):
    def __call__(
        self,
        lhs: _c.CanRPow[_T_rpow, _R_rpow],
        rhs: _T_rpow,
        /,
    ) -> _R_rpow: ...


_T_rlshift = TypeVar('_T_rlshift')
_R_rlshift = TypeVar('_R_rlshift')


@final
class DoesRLshift(Protocol):
    def __call__(
        self,
        lhs: _c.CanRLshift[_T_rlshift, _R_rlshift],
        rhs: _T_rlshift,
        /,
    ) -> _R_rlshift: ...


_T_rrshift = TypeVar('_T_rrshift')
_R_rrshift = TypeVar('_R_rrshift')


@final
class DoesRRshift(Protocol):
    def __call__(
        self,
        lhs: _c.CanRRshift[_T_rrshift, _R_rrshift],
        rhs: _T_rrshift,
        /,
    ) -> _R_rrshift: ...


_T_rand = TypeVar('_T_rand')
_R_rand = TypeVar('_R_rand')


@final
class DoesRAnd(Protocol):
    def __call__(
        self,
        lhs: _c.CanRAnd[_T_rand, _R_rand],
        rhs: _T_rand,
        /,
    ) -> _R_rand: ...


_T_rxor = TypeVar('_T_rxor')
_R_rxor = TypeVar('_R_rxor')


@final
class DoesRXor(Protocol):
    def __call__(
        self,
        lhs: _c.CanRXor[_T_rxor, _R_rxor],
        rhs: _T_rxor,
        /,
    ) -> _R_rxor: ...


_T_ror = TypeVar('_T_ror')
_R_ror = TypeVar('_R_ror')


@final
class DoesROr(Protocol):
    def __call__(
        self,
        lhs: _c.CanROr[_T_ror, _R_ror],
        rhs: _T_ror,
        /,
    ) -> _R_ror: ...


# augmented / in-place operators


_T_iadd = TypeVar('_T_iadd')
_R_iadd = TypeVar('_R_iadd')


@final
class DoesIAdd(Protocol):
    def __call__(
        self,
        lhs: _c.CanIAdd[_T_iadd, _R_iadd],
        rhs: _T_iadd,
        /,
    ) -> _R_iadd: ...


_T_isub = TypeVar('_T_isub')
_R_isub = TypeVar('_R_isub')


@final
class DoesISub(Protocol):
    def __call__(
        self,
        lhs: _c.CanISub[_T_isub, _R_isub],
        rhs: _T_isub,
        /,
    ) -> _R_isub: ...


_T_imul = TypeVar('_T_imul')
_R_imul = TypeVar('_R_imul')


@final
class DoesIMul(Protocol):
    def __call__(
        self,
        lhs: _c.CanIMul[_T_imul, _R_imul],
        rhs: _T_imul,
        /,
    ) -> _R_imul: ...


_T_imatmul = TypeVar('_T_imatmul')
_R_imatmul = TypeVar('_R_imatmul')


@final
class DoesIMatmul(Protocol):
    def __call__(
        self,
        lhs: _c.CanIMatmul[_T_imatmul, _R_imatmul],
        rhs: _T_imatmul,
        /,
    ) -> _R_imatmul: ...


_T_itruediv = TypeVar('_T_itruediv')
_R_itruediv = TypeVar('_R_itruediv')


@final
class DoesITruediv(Protocol):
    def __call__(
        self,
        lhs: _c.CanITruediv[_T_itruediv, _R_itruediv],
        rhs: _T_itruediv,
        /,
    ) -> _R_itruediv: ...


_T_ifloordiv = TypeVar('_T_ifloordiv')
_R_ifloordiv = TypeVar('_R_ifloordiv')


@final
class DoesIFloordiv(Protocol):
    def __call__(
        self,
        lhs: _c.CanIFloordiv[_T_ifloordiv, _R_ifloordiv],
        rhs: _T_ifloordiv,
        /,
    ) -> _R_ifloordiv: ...


_T_imod = TypeVar('_T_imod')
_R_imod = TypeVar('_R_imod')


@final
class DoesIMod(Protocol):
    def __call__(
        self,
        lhs: _c.CanIMod[_T_imod, _R_imod],
        rhs: _T_imod,
        /,
    ) -> _R_imod: ...


_T_ipow = TypeVar('_T_ipow')
_R_ipow = TypeVar('_R_ipow')


@final
class DoesIPow(Protocol):
    def __call__(self,
        lhs: _c.CanIPow[_T_ipow, _R_ipow],
        rhs: _T_ipow,
        /,
    ) -> _R_ipow: ...


_T_ilshift = TypeVar('_T_ilshift')
_R_ilshift = TypeVar('_R_ilshift')


@final
class DoesILshift(Protocol):
    def __call__(
        self,
        lhs: _c.CanILshift[_T_ilshift, _R_ilshift],
        rhs: _T_ilshift,
        /,
    ) -> _R_ilshift: ...


_T_irshift = TypeVar('_T_irshift')
_R_irshift = TypeVar('_R_irshift')


@final
class DoesIRshift(Protocol):
    def __call__(
        self,
        lhs: _c.CanIRshift[_T_irshift, _R_irshift],
        rhs: _T_irshift,
        /,
    ) -> _R_irshift: ...


_T_iand = TypeVar('_T_iand')
_R_iand = TypeVar('_R_iand')


@final
class DoesIAnd(Protocol):
    def __call__(
        self,
        lhs: _c.CanIAnd[_T_iand, _R_iand],
        rhs: _T_iand,
        /,
    ) -> _R_iand: ...


_T_ixor = TypeVar('_T_ixor')
_R_ixor = TypeVar('_R_ixor')


@final
class DoesIXor(Protocol):
    def __call__(
        self,
        lhs: _c.CanIXor[_T_ixor, _R_ixor],
        rhs: _T_ixor,
        /,
    ) -> _R_ixor: ...


_T_ior = TypeVar('_T_ior')
_R_ior = TypeVar('_R_ior')


@final
class DoesIOr(Protocol):
    def __call__(
        self,
        lhs: _c.CanIOr[_T_ior, _R_ior],
        rhs: _T_ior,
        /,
    ) -> _R_ior: ...


# unary arithmetic

_R_neg = TypeVar('_R_neg')


@final
class DoesNeg(Protocol):
    def __call__(self, val: _c.CanNeg[_R_neg], /) -> _R_neg: ...


_R_pos = TypeVar('_R_pos')


@final
class DoesPos(Protocol):
    def __call__(self, val: _c.CanPos[_R_pos], /) -> _R_pos: ...


_R_abs = TypeVar('_R_abs')


@final
class DoesAbs(Protocol):
    def __call__(self, val: _c.CanAbs[_R_abs], /) -> _R_abs: ...


_R_invert = TypeVar('_R_invert')


@final
class DoesInvert(Protocol):
    def __call__(self, val: _c.CanInvert[_R_invert], /) -> _R_invert: ...


# object identification


_R_index = TypeVar('_R_index', bound=int)


@final
class DoesIndex(Protocol):
    def __call__(self, obj: _c.CanIndex[_R_index], /) -> _R_index: ...


@final
class DoesHash(Protocol):
    def __call__(self, obj: _c.CanHash, /) -> int: ...


# rounding

_N_round = TypeVar('_N_round')
_R_round = TypeVar('_R_round')


@final
class DoesRound(Protocol):
    @overload
    def __call__(
        self,
        obj: _c.CanRound1[_R_round],
        /,
    ) -> _R_round: ...
    @overload
    def __call__(
        self,
        obj: _c.CanRound1[_R_round],
        /,
        ndigits: None = None,
    ) -> _R_round: ...
    @overload
    def __call__(
        self,
        obj: _c.CanRound2[_N_round, _R_round],
        /,
        ndigits: _N_round,
    ) -> _R_round: ...


_R_trunc = TypeVar('_R_trunc')


@final
class DoesTrunc(Protocol):
    def __call__(self, obj: _c.CanTrunc[_R_trunc], /) -> _R_trunc: ...


_R_floor = TypeVar('_R_floor')


@final
class DoesFloor(Protocol):
    def __call__(self, obj: _c.CanFloor[_R_floor], /) -> _R_floor: ...


_R_ceil = TypeVar('_R_ceil')


@final
class DoesCeil(Protocol):
    def __call__(self, obj: _c.CanCeil[_R_ceil], /) -> _R_ceil: ...
