from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias


if sys.version_info >= (3, 13):
    from typing import ParamSpec, Protocol, TypeVar, final, overload
else:
    from typing_extensions import ParamSpec, Protocol, TypeVar, final, overload

if TYPE_CHECKING:
    from collections.abc import Callable

    import optype._can as _c


from ._utils import set_module


# iteration


_R_next = TypeVar('_R_next')
_D_next = TypeVar('_D_next')


@set_module('optype')
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


@set_module('optype')
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
_R_iter = TypeVar('_R_iter', bound='_c.CanNext[Any]')


@set_module('optype')
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


_R_aiter = TypeVar('_R_aiter', bound='_c.CanANext[Any]')


@set_module('optype')
@final
class DoesAIter(Protocol):
    def __call__(self, iterable: _c.CanAIter[_R_aiter], /) -> _R_aiter: ...


# type conversion

@set_module('optype')
@final
class DoesComplex(Protocol):
    def __call__(self, obj: _c.CanComplex, /) -> complex: ...


@set_module('optype')
@final
class DoesFloat(Protocol):
    def __call__(self, obj: _c.CanFloat, /) -> float: ...


_R_int = TypeVar('_R_int', bound=int)


@set_module('optype')
@final
class DoesInt(Protocol):
    def __call__(self, obj: _c.CanInt[_R_int], /) -> _R_int: ...


_JustFalse: TypeAlias = Literal[False]
_JustTrue: TypeAlias = Literal[True]
_Just0: TypeAlias = Literal[0]
# cannot use `optype.typing.LiteralByte` here, as it starts at 0
_PosInt: TypeAlias = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
]  # fmt: skip
_R_bool = TypeVar('_R_bool', _JustTrue, _JustFalse, bool)


@set_module('optype')
@final
class DoesBool(Protocol):
    @overload
    def __call__(self, obj: _c.CanBool[_R_bool], /) -> _R_bool: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __call__(self, obj: _c.CanLen[_Just0], /) -> _JustFalse: ...
    @overload
    def __call__(self, obj: _c.CanLen[_PosInt], /) -> _JustTrue: ...
    @overload
    def __call__(self, obj: _c.CanLen[int], /) -> bool: ...
    @overload
    def __call__(self, obj: object, /) -> _JustTrue: ...


_R_str = TypeVar('_R_str', bound=str)


@set_module('optype')
@final
class DoesStr(Protocol):
    def __call__(self, obj: _c.CanStr[_R_str], /) -> _R_str: ...


_R_bytes = TypeVar('_R_bytes', bound=bytes)


@set_module('optype')
@final
class DoesBytes(Protocol):
    def __call__(self, obj: _c.CanBytes[_R_bytes], /) -> _R_bytes: ...


# formatting


_R_repr = TypeVar('_R_repr', bound=str)


@set_module('optype')
@final
class DoesRepr(Protocol):
    def __call__(self, obj: _c.CanRepr[_R_repr], /) -> _R_repr: ...


_T_format = TypeVar('_T_format', bound=str)
_R_format = TypeVar('_R_format', bound=str)


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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
    def __call__(  # pyright: ignore[reportOverlappingOverload]
        self,
        lhs: _T_eq_lhs,
        rhs: _c.CanEq[_T_eq_lhs, _R_eq],
        /,
    ) -> _R_eq: ...


_T_ne_lhs = TypeVar('_T_ne_lhs')
_T_ne_rhs = TypeVar('_T_ne_rhs')
_R_ne = TypeVar('_R_ne')


@set_module('optype')
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
    def __call__(  # pyright: ignore[reportOverlappingOverload]
        self,
        lhs: _T_ne_lhs,
        rhs: _c.CanNe[_T_ne_lhs, _R_ne],
        /,
    ) -> _R_ne: ...


_T_gt_lhs = TypeVar('_T_gt_lhs')
_T_gt_rhs = TypeVar('_T_gt_rhs')
_R_gt = TypeVar('_R_gt')


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
@final
class DoesDelattr(Protocol):
    def __call__(
        self,
        obj: _c.CanDelattr[_N_delattr],
        name: _N_delattr,
        /,
    ) -> None: ...


_R_dir = TypeVar('_R_dir', bound='_c.CanIter[Any]')


@set_module('optype')
@final
class DoesDir(Protocol):
    @overload
    def __call__(self, /) -> list[str]: ...
    @overload
    def __call__(self, obj: _c.CanDir[_R_dir], /) -> _R_dir: ...


# callables

_Pss_call = ParamSpec('_Pss_call')
_R_call = TypeVar('_R_call')


@set_module('optype')
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


@set_module('optype')
@final
class DoesLen(Protocol):
    def __call__(self, obj: _c.CanLen[_R_len], /) -> _R_len: ...


_R_length_hint = TypeVar('_R_length_hint', bound=int)


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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
    ) -> reversed[_V_reversed]: ...


# binary infix operators


_T_add_lhs = TypeVar('_T_add_lhs')
_T_add_rhs = TypeVar('_T_add_rhs')
_R_add = TypeVar('_R_add')


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
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


@set_module('optype')
@final
class DoesROr(Protocol):
    def __call__(
        self,
        lhs: _c.CanROr[_T_ror, _R_ror],
        rhs: _T_ror,
        /,
    ) -> _R_ror: ...


# augmented / in-place operators


_T_iadd_rhs = TypeVar('_T_iadd_rhs')
_T_iadd_lhs = TypeVar('_T_iadd_lhs')
_R_iadd = TypeVar('_R_iadd')


@set_module('optype')
@final
class DoesIAdd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIAdd[_T_iadd_rhs, _R_iadd],
        rhs: _T_iadd_rhs,
        /,
    ) -> _R_iadd: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanAdd[_T_iadd_rhs, _R_iadd],
        rhs: _T_iadd_rhs,
        /,
    ) -> _R_iadd: ...
    @overload
    def __call__(
        self,
        lhs: _T_iadd_lhs,
        rhs: _c.CanRAdd[_T_iadd_lhs, _R_iadd],
        /,
    ) -> _R_iadd: ...


_T_isub_lhs = TypeVar('_T_isub_lhs')
_T_isub_rhs = TypeVar('_T_isub_rhs')
_R_isub = TypeVar('_R_isub')


@set_module('optype')
@final
class DoesISub(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanISub[_T_isub_rhs, _R_isub],
        rhs: _T_isub_rhs,
        /,
    ) -> _R_isub: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanSub[_T_isub_rhs, _R_isub],
        rhs: _T_isub_rhs,
        /,
    ) -> _R_isub: ...
    @overload
    def __call__(
        self,
        lhs: _T_isub_lhs,
        rhs: _c.CanRSub[_T_isub_lhs, _R_isub],
        /,
    ) -> _R_isub: ...


_T_imul_lhs = TypeVar('_T_imul_lhs')
_T_imul_rhs = TypeVar('_T_imul_rhs')
_R_imul = TypeVar('_R_imul')


@set_module('optype')
@final
class DoesIMul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIMul[_T_imul_rhs, _R_imul],
        rhs: _T_imul_rhs,
        /,
    ) -> _R_imul: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanMul[_T_imul_rhs, _R_imul],
        rhs: _T_imul_rhs,
        /,
    ) -> _R_imul: ...
    @overload
    def __call__(
        self,
        lhs: _T_imul_lhs,
        rhs: _c.CanRMul[_T_imul_lhs, _R_imul],
        /,
    ) -> _R_imul: ...


_T_imatmul_lhs = TypeVar('_T_imatmul_lhs')
_T_imatmul_rhs = TypeVar('_T_imatmul_rhs')
_R_imatmul = TypeVar('_R_imatmul')


@set_module('optype')
@final
class DoesIMatmul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIMatmul[_T_imatmul_rhs, _R_imatmul],
        rhs: _T_imatmul_rhs,
        /,
    ) -> _R_imatmul: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanMatmul[_T_imatmul_rhs, _R_imatmul],
        rhs: _T_imatmul_rhs,
        /,
    ) -> _R_imatmul: ...
    @overload
    def __call__(
        self,
        lhs: _T_imatmul_lhs,
        rhs: _c.CanRMatmul[_T_imatmul_lhs, _R_imatmul],
        /,
    ) -> _R_imatmul: ...


_T_itruediv_lhs = TypeVar('_T_itruediv_lhs')
_T_itruediv_rhs = TypeVar('_T_itruediv_rhs')
_R_itruediv = TypeVar('_R_itruediv')


@set_module('optype')
@final
class DoesITruediv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanITruediv[_T_itruediv_rhs, _R_itruediv],
        rhs: _T_itruediv_rhs,
        /,
    ) -> _R_itruediv: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanTruediv[_T_itruediv_rhs, _R_itruediv],
        rhs: _T_itruediv_rhs,
        /,
    ) -> _R_itruediv: ...
    @overload
    def __call__(
        self,
        lhs: _T_itruediv_lhs,
        rhs: _c.CanRTruediv[_T_itruediv_lhs, _R_itruediv],
        /,
    ) -> _R_itruediv: ...


_T_ifloordiv_lhs = TypeVar('_T_ifloordiv_lhs')
_T_ifloordiv_rhs = TypeVar('_T_ifloordiv_rhs')
_R_ifloordiv = TypeVar('_R_ifloordiv')


@set_module('optype')
@final
class DoesIFloordiv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIFloordiv[_T_ifloordiv_rhs, _R_ifloordiv],
        rhs: _T_ifloordiv_rhs,
        /,
    ) -> _R_ifloordiv: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanFloordiv[_T_ifloordiv_rhs, _R_ifloordiv],
        rhs: _T_ifloordiv_rhs,
        /,
    ) -> _R_ifloordiv: ...
    @overload
    def __call__(
        self,
        lhs: _T_ifloordiv_lhs,
        rhs: _c.CanRFloordiv[_T_ifloordiv_lhs, _R_ifloordiv],
        /,
    ) -> _R_ifloordiv: ...


_T_imod_lhs = TypeVar('_T_imod_lhs')
_T_imod_rhs = TypeVar('_T_imod_rhs')
_R_imod = TypeVar('_R_imod')


@set_module('optype')
@final
class DoesIMod(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIMod[_T_imod_rhs, _R_imod],
        rhs: _T_imod_rhs,
        /,
    ) -> _R_imod: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanMod[_T_imod_rhs, _R_imod],
        rhs: _T_imod_rhs,
        /,
    ) -> _R_imod: ...
    @overload
    def __call__(
        self,
        lhs: _T_imod_lhs,
        rhs: _c.CanRMod[_T_imod_lhs, _R_imod],
        /,
    ) -> _R_imod: ...


_T_ipow_lhs = TypeVar('_T_ipow_lhs')
_T_ipow_rhs = TypeVar('_T_ipow_rhs')
_R_ipow = TypeVar('_R_ipow')


@set_module('optype')
@final
class DoesIPow(Protocol):
    @overload
    def __call__(self,
        lhs: _c.CanIPow[_T_ipow_rhs, _R_ipow],
        rhs: _T_ipow_rhs,
        /,
    ) -> _R_ipow: ...
    @overload
    def __call__(self,
        lhs: _c.CanPow2[_T_ipow_rhs, _R_ipow],
        rhs: _T_ipow_rhs,
        /,
    ) -> _R_ipow: ...
    @overload
    def __call__(self,
        lhs: _T_ipow_lhs,
        rhs: _c.CanRPow[_T_ipow_lhs, _R_ipow],
        /,
    ) -> _R_ipow: ...


_T_ilshift_lhs = TypeVar('_T_ilshift_lhs')
_T_ilshift_rhs = TypeVar('_T_ilshift_rhs')
_R_ilshift = TypeVar('_R_ilshift')


@set_module('optype')
@final
class DoesILshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanILshift[_T_ilshift_rhs, _R_ilshift],
        rhs: _T_ilshift_rhs,
        /,
    ) -> _R_ilshift: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanLshift[_T_ilshift_rhs, _R_ilshift],
        rhs: _T_ilshift_rhs,
        /,
    ) -> _R_ilshift: ...
    @overload
    def __call__(
        self,
        lhs: _T_ilshift_lhs,
        rhs: _c.CanRLshift[_T_ilshift_lhs, _R_ilshift],
        /,
    ) -> _R_ilshift: ...


_T_irshift_lhs = TypeVar('_T_irshift_lhs')
_T_irshift_rhs = TypeVar('_T_irshift_rhs')
_R_irshift = TypeVar('_R_irshift')


@set_module('optype')
@final
class DoesIRshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIRshift[_T_irshift_rhs, _R_irshift],
        rhs: _T_irshift_rhs,
        /,
    ) -> _R_irshift: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanRshift[_T_irshift_rhs, _R_irshift],
        rhs: _T_irshift_rhs,
        /,
    ) -> _R_irshift: ...
    @overload
    def __call__(
        self,
        lhs: _T_irshift_lhs,
        rhs: _c.CanRRshift[_T_irshift_lhs, _R_irshift],
        /,
    ) -> _R_irshift: ...


_T_iand_lhs = TypeVar('_T_iand_lhs')
_T_iand_rhs = TypeVar('_T_iand_rhs')
_R_iand = TypeVar('_R_iand')


@set_module('optype')
@final
class DoesIAnd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIAnd[_T_iand_rhs, _R_iand],
        rhs: _T_iand_rhs,
        /,
    ) -> _R_iand: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanAnd[_T_iand_rhs, _R_iand],
        rhs: _T_iand_rhs,
        /,
    ) -> _R_iand: ...
    @overload
    def __call__(
        self,
        lhs: _T_iand_lhs,
        rhs: _c.CanRAnd[_T_iand_lhs, _R_iand],
        /,
    ) -> _R_iand: ...


_T_ixor_lhs = TypeVar('_T_ixor_lhs')
_T_ixor_rhs = TypeVar('_T_ixor_rhs')
_R_ixor = TypeVar('_R_ixor')


@set_module('optype')
@final
class DoesIXor(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIXor[_T_ixor_rhs, _R_ixor],
        rhs: _T_ixor_rhs,
        /,
    ) -> _R_ixor: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanXor[_T_ixor_rhs, _R_ixor],
        rhs: _T_ixor_rhs,
        /,
    ) -> _R_ixor: ...
    @overload
    def __call__(
        self,
        lhs: _T_ixor_lhs,
        rhs: _c.CanRXor[_T_ixor_lhs, _R_ixor],
        /,
    ) -> _R_ixor: ...


_T_ior_lhs = TypeVar('_T_ior_lhs')
_T_ior_rhs = TypeVar('_T_ior_rhs')
_R_ior = TypeVar('_R_ior')


@set_module('optype')
@final
class DoesIOr(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIOr[_T_ior_rhs, _R_ior],
        rhs: _T_ior_rhs,
        /,
    ) -> _R_ior: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanOr[_T_ior_rhs, _R_ior],
        rhs: _T_ior_rhs,
        /,
    ) -> _R_ior: ...
    @overload
    def __call__(
        self,
        lhs: _T_ior_lhs,
        rhs: _c.CanROr[_T_ior_lhs, _R_ior],
        /,
    ) -> _R_ior: ...


# unary arithmetic

_R_neg = TypeVar('_R_neg')


@set_module('optype')
@final
class DoesNeg(Protocol):
    def __call__(self, val: _c.CanNeg[_R_neg], /) -> _R_neg: ...


_R_pos = TypeVar('_R_pos')


@set_module('optype')
@final
class DoesPos(Protocol):
    def __call__(self, val: _c.CanPos[_R_pos], /) -> _R_pos: ...


_R_abs = TypeVar('_R_abs')


@set_module('optype')
@final
class DoesAbs(Protocol):
    def __call__(self, val: _c.CanAbs[_R_abs], /) -> _R_abs: ...


_R_invert = TypeVar('_R_invert')


@set_module('optype')
@final
class DoesInvert(Protocol):
    def __call__(self, val: _c.CanInvert[_R_invert], /) -> _R_invert: ...


# object identification


_R_index = TypeVar('_R_index', bound=int)


@set_module('optype')
@final
class DoesIndex(Protocol):
    def __call__(self, obj: _c.CanIndex[_R_index], /) -> _R_index: ...


@set_module('optype')
@final
class DoesHash(Protocol):
    def __call__(self, obj: _c.CanHash, /) -> int: ...


# rounding

_N_round = TypeVar('_N_round')
_R_round = TypeVar('_R_round')


@set_module('optype')
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


@set_module('optype')
@final
class DoesTrunc(Protocol):
    def __call__(self, obj: _c.CanTrunc[_R_trunc], /) -> _R_trunc: ...


_R_floor = TypeVar('_R_floor')


@set_module('optype')
@final
class DoesFloor(Protocol):
    def __call__(self, obj: _c.CanFloor[_R_floor], /) -> _R_floor: ...


_R_ceil = TypeVar('_R_ceil')


@set_module('optype')
@final
class DoesCeil(Protocol):
    def __call__(self, obj: _c.CanCeil[_R_ceil], /) -> _R_ceil: ...
