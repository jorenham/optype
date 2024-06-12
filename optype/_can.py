from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)


if sys.version_info >= (3, 13):
    from typing import ParamSpec, Self, TypeVar, TypeVarTuple, Unpack, override
else:
    from typing_extensions import (
        ParamSpec,
        Self,  # noqa: TCH002
        TypeVar,
        TypeVarTuple,
        Unpack,
        override,
    )


if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType


_Ignored: TypeAlias = Any


#
# Type conversion
#

_T_bool_co = TypeVar(
    '_T_bool_co',
    Literal[True],
    Literal[False],
    bool,
    covariant=True,
    default=bool,
)


@runtime_checkable
class CanBool(Protocol[_T_bool_co]):
    def __bool__(self, /) -> _T_bool_co: ...


_T_int_co = TypeVar('_T_int_co', covariant=True, bound=int, default=int)


@runtime_checkable
class CanInt(Protocol[_T_int_co]):
    def __int__(self, /) -> _T_int_co: ...


_T_next_co = TypeVar('_T_next_co', covariant=True)


@runtime_checkable
class CanFloat(Protocol):
    def __float__(self, /) -> float: ...


@runtime_checkable
class CanComplex(Protocol):
    def __complex__(self, /) -> complex: ...


_T_bytes_co = TypeVar(
    '_T_bytes_co',
    covariant=True,
    bound=bytes,
    default=bytes,
)


@runtime_checkable
class CanBytes(Protocol[_T_bytes_co]):
    """
    The `__bytes__: (CanBytes[Y]) -> Y` method is *co*variant on `+Y`.
    So if `__bytes__` returns an instance of a custom `bytes` subtype
    `Y <: bytes`, then `bytes()` will also return `Y` (i.e. no upcasting).
    """
    def __bytes__(self, /) -> _T_bytes_co: ...


_T_str_co = TypeVar('_T_str_co', covariant=True, bound=str, default=str)


@runtime_checkable
class CanStr(Protocol[_T_str_co]):
    """
    Each `object` has a *co*variant `__str__: (CanStr[Y=str]) -> Y` method on
    `+Y`. That means that if `__str__()` returns an instance of a custom `str`
    subtype `Y <: str`, then `str()` will also return `Y` (i.e. no upcasting).
    """
    @override
    def __str__(self, /) -> _T_str_co: ...


#
# Representation
#

@runtime_checkable
class CanHash(Protocol):
    @override
    def __hash__(self, /) -> int: ...


_T_index_co = TypeVar('_T_index_co', covariant=True, bound=int, default=int)


@runtime_checkable
class CanIndex(Protocol[_T_index_co]):
    def __index__(self, /) -> _T_index_co: ...


_T_repr_co = TypeVar('_T_repr_co', covariant=True, bound=str, default=str)


@runtime_checkable
class CanRepr(Protocol[_T_repr_co]):
    """
    Each `object` has a *co*variant `__repr__: (CanRepr[Y=str]) -> Y` method.
    That means that if `__repr__` returns an instance of a custom `str`
    subtype `Y <: str`, then `repr()` will also return `Y` (i.e. no upcasting).
    """
    @override
    def __repr__(self, /) -> _T_repr_co: ...


_T_format_contra = TypeVar(
    '_T_format_contra',
    contravariant=True,
    bound=str,
    default=str,
)
_T_format_co = TypeVar(
    '_T_format_co',
    covariant=True,
    bound=str,
    default=str,
)


@runtime_checkable
class CanFormat(Protocol[_T_format_contra, _T_format_co]):
    """
    Each `object` has a `__format__: (CanFormat[X, Y], X) -> Y` method, with
    `-X` *contra*variant, and `+Y` *co*variant. Both `X` and `Y` can be `str`
    or `str` subtypes. Note that `format()` *does not* upcast `Y` to `str`.
    """
    @override
    def __format__(  # pyright:ignore[reportIncompatibleMethodOverride]
        self,
        format_spec: _T_format_contra,
        /,
    ) -> _T_format_co: ...


#
# Iteration
#

@runtime_checkable
class CanNext(Protocol[_T_next_co]):
    """
    Similar to `collections.abc.Iterator`, but without the (often redundant)
    requirement to also have a `__iter__` method.
    """
    def __next__(self, /) -> _T_next_co: ...


_T_iter_co = TypeVar('_T_iter_co', covariant=True, bound=CanNext[Any])


@runtime_checkable
class CanIter(Protocol[_T_iter_co]):
    """Similar to `collections.abc.Iterable`, but more flexible."""
    def __iter__(self, /) -> _T_iter_co: ...


_T_iter_self_co = TypeVar('_T_iter_self_co', covariant=True)


@runtime_checkable
class CanIterSelf(
    CanNext[_T_iter_self_co],
    CanIter[CanNext[_T_iter_self_co]],
    Protocol[_T_iter_self_co],
):
    """
    Equivalent to `collections.abc.Iterator[T]`, minus the `abc` nonsense.
    """
    @override
    def __iter__(self, /) -> Self: ...


#
# Async iteration
#

_T_anext_co = TypeVar('_T_anext_co', covariant=True)


@runtime_checkable
class CanANext(Protocol[_T_anext_co]):
    def __anext__(self, /) -> _T_anext_co: ...


_T_aiter_co = TypeVar('_T_aiter_co', covariant=True, bound=CanANext[Any])


@runtime_checkable
class CanAIter(Protocol[_T_aiter_co]):
    def __aiter__(self, /) -> _T_aiter_co: ...


@runtime_checkable
class CanAIterSelf(
    CanANext[_T_anext_co],
    CanAIter[CanANext[_T_anext_co]],
    Protocol[_T_anext_co],
):
    """A less inflexible variant of `collections.abc.AsyncIterator[T]`."""
    @override
    def __aiter__(self, /) -> Self: ...


#
# "Rich" comparison ops
#

_T_eq_contra = TypeVar('_T_eq_contra', contravariant=True, default=object)
_T_eq_co = TypeVar('_T_eq_co', covariant=True, default=bool)


@runtime_checkable
class CanEq(Protocol[_T_eq_contra, _T_eq_co]):  # noqa: PLW1641
    """
    Unfortunately, `typeshed` (incorrectly) annotates `object.__eq__` as
    `(Self, object) -> bool`.
    As a counter-example, consider `numpy.ndarray`. It's `__eq__` method
    returns a boolean (mask) array of the same shape as the input array.
    Moreover, `numpy.ndarray` doesn't even implement `CanBool` (`bool()`
    raises a `TypeError` for shapes of size > 1).
    There is nothing wrong with this implementation, even though `typeshed`
    (incorrectly) won't allow it (because `numpy.ndarray <: object`).

    So in reality, it should be `__eq__: (Self, X) -> Y`, with `-X` unbounded
    and *contra*variant, and `+Y` unbounded and *co*variant.
    """
    @override
    def __eq__(self, rhs: _T_eq_contra, /) -> _T_eq_co: ...  # pyright:ignore[reportIncompatibleMethodOverride]


_T_ne_contra = TypeVar('_T_ne_contra', contravariant=True, default=object)
_T_ne_co = TypeVar('_T_ne_co', covariant=True, default=bool)


@runtime_checkable
class CanNe(Protocol[_T_ne_contra, _T_ne_co]):
    """
    Just like `__eq__`, the `__ne__` method is incorrectly annotated in
    `typeshed`. See `CanEq` for why this is, and how `optype` fixes this.
    """
    @override
    def __ne__(self, rhs: _T_ne_contra, /) -> _T_ne_co: ...  # pyright:ignore[reportIncompatibleMethodOverride]


_T_lt_contra = TypeVar('_T_lt_contra', contravariant=True)
_T_lt_co = TypeVar('_T_lt_co', covariant=True, default=bool)


@runtime_checkable
class CanLt(Protocol[_T_lt_contra, _T_lt_co]):
    def __lt__(self, rhs: _T_lt_contra, /) -> _T_lt_co: ...


_T_le_contra = TypeVar('_T_le_contra', contravariant=True)
_T_le_co = TypeVar('_T_le_co', covariant=True, default=bool)


@runtime_checkable
class CanLe(Protocol[_T_le_contra, _T_le_co]):
    def __le__(self, rhs: _T_le_contra, /) -> _T_le_co: ...


_T_gt_contra = TypeVar('_T_gt_contra', contravariant=True)
_T_gt_co = TypeVar('_T_gt_co', covariant=True, default=bool)


@runtime_checkable
class CanGt(Protocol[_T_gt_contra, _T_gt_co]):
    def __gt__(self, rhs: _T_gt_contra, /) -> _T_gt_co: ...


_T_ge_contra = TypeVar('_T_ge_contra', contravariant=True)
_T_ge_co = TypeVar('_T_ge_co', covariant=True, default=bool)


@runtime_checkable
class CanGe(Protocol[_T_ge_contra, _T_ge_co]):
    def __ge__(self, rhs: _T_ge_contra, /) -> _T_ge_co: ...


#
# Callables
#

# This should (obviously) be contravariant; but that's not possible, because
# (apparently) the PEP authors didn't seem to have figured out what
# co/contravariance means, or the fact that paramspec's are quite literally a
# fundamentally broken feature without it (according to type theory, that is).
_Tss_call = ParamSpec('_Tss_call')
_V_call_co = TypeVar('_V_call_co', covariant=True)


@runtime_checkable
class CanCall(Protocol[_Tss_call, _V_call_co]):
    def __call__(
        self,
        /,
        *args: _Tss_call.args,
        **kwargs: _Tss_call.kwargs,
    ) -> _V_call_co: ...


#
# Dynamic attribute access
#

_N_getattr_contra = TypeVar(
    '_N_getattr_contra',
    contravariant=True,
    bound=str,
    default=str,
)
_V_getattr_co = TypeVar('_V_getattr_co', covariant=True, default=Any)


@runtime_checkable
class CanGetattr(Protocol[_N_getattr_contra, _V_getattr_co]):
    def __getattr__(self, name: _N_getattr_contra, /) -> _V_getattr_co: ...


_N_getattribute_contra = TypeVar(
    '_N_getattribute_contra',
    contravariant=True,
    bound=str,
    default=str,
)
_V_getattribute_co = TypeVar('_V_getattribute_co', covariant=True, default=Any)


@runtime_checkable
class CanGetattribute(Protocol[_N_getattribute_contra, _V_getattribute_co]):
    """Note that `isinstance(x, CanGetattribute)` is always true."""
    @override
    def __getattribute__(  # pyright:ignore[reportIncompatibleMethodOverride]
        self,
        name: _N_getattribute_contra,
        /,
    ) -> _V_getattribute_co: ...


_N_setattr_contra = TypeVar(
    '_N_setattr_contra',
    contravariant=True,
    bound=str,
    default=str,
)
_V_setattr_contra = TypeVar(
    '_V_setattr_contra',
    contravariant=True,
    default=Any,
)


@runtime_checkable
class CanSetattr(Protocol[_N_setattr_contra, _V_setattr_contra]):
    """Note that `isinstance(x, CanSetattr)` is always true."""
    @override
    def __setattr__(  # pyright:ignore[reportIncompatibleMethodOverride]
        self,
        name: _N_setattr_contra,
        value: _V_setattr_contra,
        /,
    ) -> _Ignored: ...


_N_delattr_contra = TypeVar(
    '_N_delattr_contra',
    contravariant=True,
    bound=str,
    default=str,
)


@runtime_checkable
class CanDelattr(Protocol[_N_delattr_contra]):
    @override
    def __delattr__(self, name: _N_delattr_contra, /) -> Any: ...  # pyright:ignore[reportIncompatibleMethodOverride]


_T_dir_co = TypeVar(
    '_T_dir_co',
    covariant=True,
    bound=CanIter[Any],
    default=CanIter[CanIterSelf[str]],
)


@runtime_checkable
class CanDir(Protocol[_T_dir_co]):
    @override
    def __dir__(self, /) -> _T_dir_co: ...


#
# Descriptors
#

_T_get_contra = TypeVar('_T_get_contra', contravariant=True, bound=object)
_V_get_co = TypeVar('_V_get_co', covariant=True)
_VT_get_co = TypeVar('_VT_get_co', covariant=True, default=_V_get_co)


@runtime_checkable
class CanGet(Protocol[_T_get_contra, _V_get_co, _VT_get_co]):
    @overload
    def __get__(
        self,
        owner: _T_get_contra,
        owner_type: type[_T_get_contra] | None = ...,
        /,
    ) -> _V_get_co: ...
    @overload
    def __get__(
        self,
        owner: None,
        owner_type: type[_T_get_contra],
        /,
    ) -> _VT_get_co: ...


_T_set_contra = TypeVar('_T_set_contra', contravariant=True, bound=object)
_V_set_contra = TypeVar('_V_set_contra', contravariant=True)


@runtime_checkable
class CanSet(Protocol[_T_set_contra, _V_set_contra]):
    def __set__(
        self,
        owner: _T_set_contra,
        value: _V_set_contra,
        /,
    ) -> _Ignored: ...


_T_delete_contra = TypeVar(
    '_T_delete_contra',
    contravariant=True,
    bound=object,
)


@runtime_checkable
class CanDelete(Protocol[_T_delete_contra]):
    def __delete__(self, owner: _T_delete_contra, /) -> _Ignored: ...


_T_set_name_contra = TypeVar(
    '_T_set_name_contra',
    contravariant=True,
    bound=object,
)
_N_set_name_contra = TypeVar(
    '_N_set_name_contra',
    contravariant=True,
    bound=str,
    default=str,
)


@runtime_checkable
class CanSetName(Protocol[_T_set_name_contra, _N_set_name_contra]):
    def __set_name__(
        self,
        owner_type: type[_T_set_name_contra],
        name: _N_set_name_contra,
        /,
    ) -> _Ignored: ...


#
# Containers
#

_V_len_co = TypeVar('_V_len_co', covariant=True, bound=int, default=int)


@runtime_checkable
class CanLen(Protocol[_V_len_co]):
    def __len__(self, /) -> _V_len_co: ...


_V_length_hint_co = TypeVar(
    '_V_length_hint_co',
    covariant=True,
    bound=int,
    default=int,
)


@runtime_checkable
class CanLengthHint(Protocol[_V_length_hint_co]):
    def __length_hint__(self, /) -> _V_length_hint_co: ...


_K_getitem_contra = TypeVar('_K_getitem_contra', contravariant=True)
_V_getitem_co = TypeVar('_V_getitem_co', covariant=True)


@runtime_checkable
class CanGetitem(Protocol[_K_getitem_contra, _V_getitem_co]):
    def __getitem__(self, key: _K_getitem_contra, /) -> _V_getitem_co: ...


_K_setitem_contra = TypeVar('_K_setitem_contra', contravariant=True)
_V_setitem_contra = TypeVar('_V_setitem_contra', contravariant=True)


@runtime_checkable
class CanSetitem(Protocol[_K_setitem_contra, _V_setitem_contra]):
    def __setitem__(
        self,
        key: _K_setitem_contra,
        value: _V_setitem_contra,
        /,
    ) -> _Ignored: ...


_K_delitem_contra = TypeVar('_K_delitem_contra', contravariant=True)


@runtime_checkable
class CanDelitem(Protocol[_K_delitem_contra]):
    def __delitem__(self, key: _K_delitem_contra, /) -> None: ...


# theoretically not required to be iterable; but it probably should be
_V_reversed_co = TypeVar('_V_reversed_co', covariant=True)


@runtime_checkable
class CanReversed(Protocol[_V_reversed_co]):
    def __reversed__(self, /) -> _V_reversed_co: ...


# theoretically not required to be hashable; but it probably should be
_K_contains_contra = TypeVar(
    '_K_contains_contra',
    contravariant=True,
    default=object,
)
# could be set to e.g. Literal[False] empty (user-defined) container types
_V_contains_co = TypeVar(
    '_V_contains_co',
    Literal[True],
    Literal[False],
    bool,
    covariant=True,
    default=bool,
)


@runtime_checkable
class CanContains(Protocol[_K_contains_contra, _V_contains_co]):
    def __contains__(self, key: _K_contains_contra, /) -> _V_contains_co: ...


_K_missing_contra = TypeVar('_K_missing_contra', contravariant=True)
_V_missing_co = TypeVar('_V_missing_co', covariant=True)


@runtime_checkable
class CanMissing(Protocol[_K_missing_contra, _V_missing_co]):
    def __missing__(self, key: _K_missing_contra, /) -> _V_missing_co: ...


_K_get_missing_contra = TypeVar('_K_get_missing_contra', contravariant=True)
_V_get_missing_co = TypeVar('_V_get_missing_co', covariant=True)
_D_get_missing_co = TypeVar(
    '_D_get_missing_co',
    covariant=True,
    default=_V_get_missing_co,
)


@runtime_checkable
class CanGetMissing(
    CanGetitem[_K_get_missing_contra, _V_get_missing_co],
    CanMissing[_K_get_missing_contra, _D_get_missing_co],
    Protocol[_K_get_missing_contra, _V_get_missing_co, _D_get_missing_co],
): ...


_K_sequence_contra = TypeVar(
    '_K_sequence_contra',
    contravariant=True,
    bound=CanIndex | slice,
)
_V_sequence_co = TypeVar('_V_sequence_co', covariant=True)


@runtime_checkable
class CanSequence(
    CanLen,
    CanGetitem[_K_sequence_contra, _V_sequence_co],
    Protocol[_K_sequence_contra, _V_sequence_co],
):
    """
    A sequence is an object with a __len__ method and a
    __getitem__ method that takes int(-like) argument as key (the index).
    Additionally, it is expected to be 0-indexed (the first element is at
    index 0) and "dense" (i.e. the indices are consecutive integers, and are
    obtainable with e.g. `range(len(_))`).
    """


#
# Numeric ops
#

_T_add_contra = TypeVar('_T_add_contra', contravariant=True)
_T_add_co = TypeVar('_T_add_co', covariant=True)


@runtime_checkable
class CanAdd(Protocol[_T_add_contra, _T_add_co]):
    def __add__(self, rhs: _T_add_contra, /) -> _T_add_co: ...


_T_sub_contra = TypeVar('_T_sub_contra', contravariant=True)
_T_sub_co = TypeVar('_T_sub_co', covariant=True)


@runtime_checkable
class CanSub(Protocol[_T_sub_contra, _T_sub_co]):
    def __sub__(self, rhs: _T_sub_contra, /) -> _T_sub_co: ...


_T_mul_contra = TypeVar('_T_mul_contra', contravariant=True)
_T_mul_co = TypeVar('_T_mul_co', covariant=True)


@runtime_checkable
class CanMul(Protocol[_T_mul_contra, _T_mul_co]):
    def __mul__(self, rhs: _T_mul_contra, /) -> _T_mul_co: ...


_T_matmul_contra = TypeVar('_T_matmul_contra', contravariant=True)
_T_matmul_co = TypeVar('_T_matmul_co', covariant=True)


@runtime_checkable
class CanMatmul(Protocol[_T_matmul_contra, _T_matmul_co]):
    def __matmul__(self, rhs: _T_matmul_contra, /) -> _T_matmul_co: ...


_T_truediv_contra = TypeVar('_T_truediv_contra', contravariant=True)
_T_truediv_co = TypeVar('_T_truediv_co', covariant=True)


@runtime_checkable
class CanTruediv(Protocol[_T_truediv_contra, _T_truediv_co]):
    def __truediv__(self, rhs: _T_truediv_contra, /) -> _T_truediv_co: ...


_T_floordiv_contra = TypeVar('_T_floordiv_contra', contravariant=True)
_T_floordiv_co = TypeVar('_T_floordiv_co', covariant=True)


@runtime_checkable
class CanFloordiv(Protocol[_T_floordiv_contra, _T_floordiv_co]):
    def __floordiv__(self, rhs: _T_floordiv_contra, /) -> _T_floordiv_co: ...


_T_mod_contra = TypeVar('_T_mod_contra', contravariant=True)
_T_mod_co = TypeVar('_T_mod_co', covariant=True)


@runtime_checkable
class CanMod(Protocol[_T_mod_contra, _T_mod_co]):
    def __mod__(self, rhs: _T_mod_contra, /) -> _T_mod_co: ...


_T_divmod_contra = TypeVar('_T_divmod_contra', contravariant=True)
_T_divmod_co = TypeVar('_T_divmod_co', covariant=True)


@runtime_checkable
class CanDivmod(Protocol[_T_divmod_contra, _T_divmod_co]):
    def __divmod__(self, rhs: _T_divmod_contra, /) -> _T_divmod_co: ...


_T_pow2_contra = TypeVar('_T_pow2_contra', contravariant=True)
_T_pow2_co = TypeVar('_T_pow2_co', covariant=True)


@runtime_checkable
class CanPow2(Protocol[_T_pow2_contra, _T_pow2_co]):
    @overload
    def __pow__(self, exp: _T_pow2_contra, /) -> _T_pow2_co: ...
    @overload
    def __pow__(
        self,
        exp: _T_pow2_contra,
        mod: None = ...,
        /,
    ) -> _T_pow2_co: ...


_T_pow3_contra = TypeVar('_T_pow3_contra', contravariant=True)
_M_pow3_contra = TypeVar('_M_pow3_contra', contravariant=True)
_T_pow3_co = TypeVar('_T_pow3_co', covariant=True)


@runtime_checkable
class CanPow3(Protocol[_T_pow3_contra, _M_pow3_contra, _T_pow3_co]):
    def __pow__(
        self,
        exp: _T_pow3_contra,
        mod: _M_pow3_contra,
        /,
    ) -> _T_pow3_co: ...


_T_pow_contra = TypeVar('_T_pow_contra', contravariant=True)
_M_pow_contra = TypeVar('_M_pow_contra', contravariant=True)
_T_pow_co = TypeVar('_T_pow_co', covariant=True)
_TM_pow_co = TypeVar('_TM_pow_co', covariant=True, default=_T_pow_co)


@runtime_checkable
class CanPow(
    CanPow2[_T_pow_contra, _T_pow_co],
    CanPow3[_T_pow_contra, _M_pow_contra, _TM_pow_co],
    Protocol[_T_pow_contra, _M_pow_contra, _T_pow_co, _TM_pow_co],
):
    @overload
    def __pow__(self, exp: _T_pow_contra, /) -> _T_pow_co: ...
    @overload
    def __pow__(self, exp: _T_pow_contra, mod: None = ..., /) -> _T_pow_co: ...
    @overload
    def __pow__(
        self,
        exp: _T_pow_contra,
        mod: _M_pow_contra,
        /,
    ) -> _TM_pow_co: ...


_T_lshift_contra = TypeVar('_T_lshift_contra', contravariant=True)
_T_lshift_co = TypeVar('_T_lshift_co', covariant=True)


@runtime_checkable
class CanLshift(Protocol[_T_lshift_contra, _T_lshift_co]):
    def __lshift__(self, rhs: _T_lshift_contra, /) -> _T_lshift_co: ...


_T_rshift_contra = TypeVar('_T_rshift_contra', contravariant=True)
_T_rshift_co = TypeVar('_T_rshift_co', covariant=True)


@runtime_checkable
class CanRshift(Protocol[_T_rshift_contra, _T_rshift_co]):
    def __rshift__(self, rhs: _T_rshift_contra, /) -> _T_rshift_co: ...


_T_and_contra = TypeVar('_T_and_contra', contravariant=True)
_T_and_co = TypeVar('_T_and_co', covariant=True)


@runtime_checkable
class CanAnd(Protocol[_T_and_contra, _T_and_co]):
    def __and__(self, rhs: _T_and_contra, /) -> _T_and_co: ...


_T_xor_contra = TypeVar('_T_xor_contra', contravariant=True)
_T_xor_co = TypeVar('_T_xor_co', covariant=True)


@runtime_checkable
class CanXor(Protocol[_T_xor_contra, _T_xor_co]):
    def __xor__(self, rhs: _T_xor_contra, /) -> _T_xor_co: ...


_T_or_contra = TypeVar('_T_or_contra', contravariant=True)
_T_or_co = TypeVar('_T_or_co', covariant=True)


@runtime_checkable
class CanOr(Protocol[_T_or_contra, _T_or_co]):
    def __or__(self, rhs: _T_or_contra, /) -> _T_or_co: ...


#
# Reflected numeric ops
#

_T_radd_contra = TypeVar('_T_radd_contra', contravariant=True)
_T_radd_co = TypeVar('_T_radd_co', covariant=True)


@runtime_checkable
class CanRAdd(Protocol[_T_radd_contra, _T_radd_co]):
    def __radd__(self, rhs: _T_radd_contra, /) -> _T_radd_co: ...


_T_rsub_contra = TypeVar('_T_rsub_contra', contravariant=True)
_T_rsub_co = TypeVar('_T_rsub_co', covariant=True)


@runtime_checkable
class CanRSub(Protocol[_T_rsub_contra, _T_rsub_co]):
    def __rsub__(self, rhs: _T_rsub_contra, /) -> _T_rsub_co: ...


_T_rmul_contra = TypeVar('_T_rmul_contra', contravariant=True)
_T_rmul_co = TypeVar('_T_rmul_co', covariant=True)


@runtime_checkable
class CanRMul(Protocol[_T_rmul_contra, _T_rmul_co]):
    def __rmul__(self, rhs: _T_rmul_contra, /) -> _T_rmul_co: ...


_T_rmatmul_contra = TypeVar('_T_rmatmul_contra', contravariant=True)
_T_rmatmul_co = TypeVar('_T_rmatmul_co', covariant=True)


@runtime_checkable
class CanRMatmul(Protocol[_T_rmatmul_contra, _T_rmatmul_co]):
    def __rmatmul__(self, rhs: _T_rmatmul_contra, /) -> _T_rmatmul_co: ...


_T_rtruediv_contra = TypeVar('_T_rtruediv_contra', contravariant=True)
_T_rtruediv_co = TypeVar('_T_rtruediv_co', covariant=True)


@runtime_checkable
class CanRTruediv(Protocol[_T_rtruediv_contra, _T_rtruediv_co]):
    def __rtruediv__(self, rhs: _T_rtruediv_contra, /) -> _T_rtruediv_co: ...


_T_rfloordiv_contra = TypeVar('_T_rfloordiv_contra', contravariant=True)
_T_rfloordiv_co = TypeVar('_T_rfloordiv_co', covariant=True)


@runtime_checkable
class CanRFloordiv(Protocol[_T_rfloordiv_contra, _T_rfloordiv_co]):
    def __rfloordiv__(
        self,
        rhs: _T_rfloordiv_contra,
        /,
    ) -> _T_rfloordiv_co: ...


_T_rmod_contra = TypeVar('_T_rmod_contra', contravariant=True)
_T_rmod_co = TypeVar('_T_rmod_co', covariant=True)


@runtime_checkable
class CanRMod(Protocol[_T_rmod_contra, _T_rmod_co]):
    def __rmod__(self, rhs: _T_rmod_contra, /) -> _T_rmod_co: ...


_T_rdivmod_contra = TypeVar('_T_rdivmod_contra', contravariant=True)
_T_rdivmod_co = TypeVar('_T_rdivmod_co', covariant=True)


@runtime_checkable
class CanRDivmod(Protocol[_T_rdivmod_contra, _T_rdivmod_co]):
    def __rdivmod__(self, rhs: _T_rdivmod_contra, /) -> _T_rdivmod_co: ...


_T_rpow_contra = TypeVar('_T_rpow_contra', contravariant=True)
_T_rpow_co = TypeVar('_T_rpow_co', covariant=True)


@runtime_checkable
class CanRPow(Protocol[_T_rpow_contra, _T_rpow_co]):
    def __rpow__(self, x: _T_rpow_contra) -> _T_rpow_co: ...


_T_rlshift_contra = TypeVar('_T_rlshift_contra', contravariant=True)
_T_rlshift_co = TypeVar('_T_rlshift_co', covariant=True)


@runtime_checkable
class CanRLshift(Protocol[_T_rlshift_contra, _T_rlshift_co]):
    def __rlshift__(self, rhs: _T_rlshift_contra, /) -> _T_rlshift_co: ...


_T_rrshift_contra = TypeVar('_T_rrshift_contra', contravariant=True)
_T_rrshift_co = TypeVar('_T_rrshift_co', covariant=True)


@runtime_checkable
class CanRRshift(Protocol[_T_rrshift_contra, _T_rrshift_co]):
    def __rrshift__(self, rhs: _T_rrshift_contra, /) -> _T_rrshift_co: ...


_T_rand_contra = TypeVar('_T_rand_contra', contravariant=True)
_T_rand_co = TypeVar('_T_rand_co', covariant=True)


@runtime_checkable
class CanRAnd(Protocol[_T_rand_contra, _T_rand_co]):
    def __rand__(self, rhs: _T_rand_contra, /) -> _T_rand_co: ...


_T_rxor_contra = TypeVar('_T_rxor_contra', contravariant=True)
_T_rxor_co = TypeVar('_T_rxor_co', covariant=True)


@runtime_checkable
class CanRXor(Protocol[_T_rxor_contra, _T_rxor_co]):
    def __rxor__(self, rhs: _T_rxor_contra, /) -> _T_rxor_co: ...


_T_ror_contra = TypeVar('_T_ror_contra', contravariant=True)
_T_ror_co = TypeVar('_T_ror_co', covariant=True)


@runtime_checkable
class CanROr(Protocol[_T_ror_contra, _T_ror_co]):
    def __ror__(self, rhs: _T_ror_contra, /) -> _T_ror_co: ...


#
# Augmented numeric ops
#

_T_iadd_contra = TypeVar('_T_iadd_contra', contravariant=True)
_T_iadd_co = TypeVar('_T_iadd_co', covariant=True)


@runtime_checkable
class CanIAdd(Protocol[_T_iadd_contra, _T_iadd_co]):
    def __iadd__(self, rhs: _T_iadd_contra, /) -> _T_iadd_co: ...


_T_isub_contra = TypeVar('_T_isub_contra', contravariant=True)
_T_isub_co = TypeVar('_T_isub_co', covariant=True)


@runtime_checkable
class CanISub(Protocol[_T_isub_contra, _T_isub_co]):
    def __isub__(self, rhs: _T_isub_contra, /) -> _T_isub_co: ...


_T_imul_contra = TypeVar('_T_imul_contra', contravariant=True)
_T_imul_co = TypeVar('_T_imul_co', covariant=True)


@runtime_checkable
class CanIMul(Protocol[_T_imul_contra, _T_imul_co]):
    def __imul__(self, rhs: _T_imul_contra, /) -> _T_imul_co: ...


_T_imatmul_contra = TypeVar('_T_imatmul_contra', contravariant=True)
_T_imatmul_co = TypeVar('_T_imatmul_co', covariant=True)


@runtime_checkable
class CanIMatmul(Protocol[_T_imatmul_contra, _T_imatmul_co]):
    def __imatmul__(self, rhs: _T_imatmul_contra, /) -> _T_imatmul_co: ...


_T_itruediv_contra = TypeVar('_T_itruediv_contra', contravariant=True)
_T_itruediv_co = TypeVar('_T_itruediv_co', covariant=True)


@runtime_checkable
class CanITruediv(Protocol[_T_itruediv_contra, _T_itruediv_co]):
    def __itruediv__(self, rhs: _T_itruediv_contra, /) -> _T_itruediv_co: ...


_T_ifloordiv_contra = TypeVar('_T_ifloordiv_contra', contravariant=True)
_T_ifloordiv_co = TypeVar('_T_ifloordiv_co', covariant=True)


@runtime_checkable
class CanIFloordiv(Protocol[_T_ifloordiv_contra, _T_ifloordiv_co]):
    def __ifloordiv__(
        self,
        rhs: _T_ifloordiv_contra,
        /,
    ) -> _T_ifloordiv_co: ...


_T_imod_contra = TypeVar('_T_imod_contra', contravariant=True)
_T_imod_co = TypeVar('_T_imod_co', covariant=True)


@runtime_checkable
class CanIMod(Protocol[_T_imod_contra, _T_imod_co]):
    def __imod__(self, rhs: _T_imod_contra, /) -> _T_imod_co: ...


_T_ipow_contra = TypeVar('_T_ipow_contra', contravariant=True)
_T_ipow_co = TypeVar('_T_ipow_co', covariant=True)


@runtime_checkable
class CanIPow(Protocol[_T_ipow_contra, _T_ipow_co]):
    # no augmented pow/3 exists
    def __ipow__(self, rhs: _T_ipow_contra, /) -> _T_ipow_co: ...


_T_ilshift_contra = TypeVar('_T_ilshift_contra', contravariant=True)
_T_ilshift_co = TypeVar('_T_ilshift_co', covariant=True)


@runtime_checkable
class CanILshift(Protocol[_T_ilshift_contra, _T_ilshift_co]):
    def __ilshift__(self, rhs: _T_ilshift_contra, /) -> _T_ilshift_co: ...


_T_irshift_contra = TypeVar('_T_irshift_contra', contravariant=True)
_T_irshift_co = TypeVar('_T_irshift_co', covariant=True)


@runtime_checkable
class CanIRshift(Protocol[_T_irshift_contra, _T_irshift_co]):
    def __irshift__(self, rhs: _T_irshift_contra, /) -> _T_irshift_co: ...


_T_iand_contra = TypeVar('_T_iand_contra', contravariant=True)
_T_iand_co = TypeVar('_T_iand_co', covariant=True)


@runtime_checkable
class CanIAnd(Protocol[_T_iand_contra, _T_iand_co]):
    def __iand__(self, rhs: _T_iand_contra, /) -> _T_iand_co: ...


_T_ixor_contra = TypeVar('_T_ixor_contra', contravariant=True)
_T_ixor_co = TypeVar('_T_ixor_co', covariant=True)


@runtime_checkable
class CanIXor(Protocol[_T_ixor_contra, _T_ixor_co]):
    def __ixor__(self, rhs: _T_ixor_contra, /) -> _T_ixor_co: ...


_T_ior_contra = TypeVar('_T_ior_contra', contravariant=True)
_T_ior_co = TypeVar('_T_ior_co', covariant=True)


@runtime_checkable
class CanIOr(Protocol[_T_ior_contra, _T_ior_co]):
    def __ior__(self, rhs: _T_ior_contra, /) -> _T_ior_co: ...


#
# Unary arithmetic ops
#

_T_neg_co = TypeVar('_T_neg_co', covariant=True)


@runtime_checkable
class CanNeg(Protocol[_T_neg_co]):
    def __neg__(self, /) -> _T_neg_co: ...


_T_pos_co = TypeVar('_T_pos_co', covariant=True)


@runtime_checkable
class CanPos(Protocol[_T_pos_co]):
    def __pos__(self, /) -> _T_pos_co: ...


_T_abs_co = TypeVar('_T_abs_co', covariant=True)


@runtime_checkable
class CanAbs(Protocol[_T_abs_co]):
    def __abs__(self, /) -> _T_abs_co: ...


_T_invert_co = TypeVar('_T_invert_co', covariant=True)


@runtime_checkable
class CanInvert(Protocol[_T_invert_co]):
    def __invert__(self, /) -> _T_invert_co: ...


#
# Rounding
#

_T_round1_co = TypeVar('_T_round1_co', covariant=True, default=int)


@runtime_checkable
class CanRound1(Protocol[_T_round1_co]):
    @overload
    def __round__(self, /) -> _T_round1_co: ...
    @overload
    def __round__(self, ndigits: None = ...) -> _T_round1_co: ...


_T_round2_contra = TypeVar('_T_round2_contra', contravariant=True, default=int)
_T_round2_co = TypeVar('_T_round2_co', covariant=True, default=float)


@runtime_checkable
class CanRound2(Protocol[_T_round2_contra, _T_round2_co]):
    def __round__(self, ndigits: _T_round2_contra) -> _T_round2_co: ...


_T_round_contra = TypeVar('_T_round_contra', contravariant=True, default=int)
_T1_round_co = TypeVar('_T1_round_co', covariant=True, default=int)
_T2_round_co = TypeVar('_T2_round_co', covariant=True, default=float)


@runtime_checkable
class CanRound(
    CanRound1[_T1_round_co],
    CanRound2[_T_round_contra, _T2_round_co],
    Protocol[_T_round_contra, _T1_round_co, _T2_round_co],
):
    @overload
    def __round__(self, /) -> _T1_round_co: ...
    @overload
    def __round__(self, ndigits: None = ...) -> _T1_round_co: ...
    @overload
    def __round__(self, ndigits: _T_round_contra) -> _T2_round_co: ...


_T_trunc_co = TypeVar('_T_trunc_co', covariant=True, default=int)


@runtime_checkable
class CanTrunc(Protocol[_T_trunc_co]):
    def __trunc__(self, /) -> _T_trunc_co: ...


_T_floor_co = TypeVar('_T_floor_co', covariant=True, default=int)


@runtime_checkable
class CanFloor(Protocol[_T_floor_co]):
    def __floor__(self, /) -> _T_floor_co: ...


_T_ceil_co = TypeVar('_T_ceil_co', covariant=True, default=int)


@runtime_checkable
class CanCeil(Protocol[_T_ceil_co]):
    def __ceil__(self, /) -> _T_ceil_co: ...


#
# Context managers
#

_T_enter_co = TypeVar('_T_enter_co', covariant=True)


@runtime_checkable
class CanEnter(Protocol[_T_enter_co]):
    def __enter__(self, /) -> _T_enter_co: ...


_E_exit = TypeVar('_E_exit', bound=BaseException)
_T_exit_co = TypeVar('_T_exit_co', covariant=True, default=None)


@runtime_checkable
class CanExit(Protocol[_T_exit_co]):
    @overload
    def __exit__(
        self,
        exc_type: None,
        exc_instance: None,
        exc_traceback: None,
        /,
    ) -> None: ...
    @overload
    def __exit__(
        self,
        exc_type: type[_E_exit],
        exc_instance: _E_exit,
        exc_traceback: TracebackType,
        /,
    ) -> _T_exit_co: ...


@runtime_checkable
class CanWith(
    CanEnter[_T_enter_co],
    CanExit[_T_exit_co],
    Protocol[_T_enter_co, _T_exit_co],
): ...


#
# Async context managers
#


_T_aenter_co = TypeVar('_T_aenter_co', covariant=True)


@runtime_checkable
class CanAEnter(Protocol[_T_aenter_co]):
    def __aenter__(self, /) -> CanAwait[_T_aenter_co]: ...


_E_aexit = TypeVar('_E_aexit', bound=BaseException)
_T_aexit_co = TypeVar('_T_aexit_co', covariant=True, default=None)


@runtime_checkable
class CanAExit(Protocol[_T_aexit_co]):
    @overload
    def __aexit__(
        self,
        exc_type: None,
        exc_instance: None,
        exc_traceback: None,
        /,
    ) -> CanAwait[None]: ...
    @overload
    def __aexit__(
        self,
        exc_type: type[_E_aexit],
        exc_instance: _E_aexit,
        exc_traceback: TracebackType,
        /,
    ) -> CanAwait[_T_aexit_co]: ...


@runtime_checkable
class CanAsyncWith(
    CanAEnter[_T_aenter_co],
    CanAExit[_T_aexit_co],
    Protocol[_T_aenter_co, _T_aexit_co],
): ...


#
# Buffer protocol
#

_T_buffer_contra = TypeVar(
    '_T_buffer_contra',
    contravariant=True,
    bound=int,
    default=int,
)


@runtime_checkable
class CanBuffer(Protocol[_T_buffer_contra]):
    def __buffer__(self, buffer: _T_buffer_contra, /) -> memoryview: ...


@runtime_checkable
class CanReleaseBuffer(Protocol):
    def __release_buffer__(self, buffer: memoryview, /) -> None: ...


#
# Awaitables
#

_T_await_co = TypeVar('_T_await_co', covariant=True)

# This should be `None | asyncio.Future[Any]`. But that would make this
# incompatible with `collections.abc.Awaitable`, because it (annoyingly)
# uses `Any`...
_MaybeFuture: TypeAlias = Any


@runtime_checkable
class CanAwait(Protocol[_T_await_co]):
    # Technically speaking, this can return any
    # `CanNext[None | asyncio.Future[Any]]`. But in theory, the return value
    # of generators are currently impossible to type, because the return value
    # of a `yield from _` is # piggybacked using a `raise StopIteration(value)`
    # from `__next__`. So that also makes `__await__` theoretically
    # impossible to type. In practice, typecheckers work around that, by
    # accepting the lie called `collections.abc.Generator`...
    @overload
    def __await__(self: CanAwait[None]) -> CanNext[_MaybeFuture]: ...
    @overload
    def __await__(
        self: CanAwait[_T_await_co],
    ) -> Generator[_MaybeFuture, None, _T_await_co]: ...


#
# Standard library `copy`
#


_T_copy_co = TypeVar('_T_copy_co', covariant=True, bound=object)


@runtime_checkable
class CanCopy(Protocol[_T_copy_co]):
    """Support for creating shallow copies through `copy.copy`."""
    def __copy__(self, /) -> _T_copy_co: ...


_T_deepcopy_co = TypeVar('_T_deepcopy_co', covariant=True, bound=object)


@runtime_checkable
class CanDeepcopy(Protocol[_T_deepcopy_co]):
    """Support for creating deep copies through `copy.deepcopy`."""
    def __deepcopy__(self, memo: dict[int, Any], /) -> _T_deepcopy_co: ...


_T_replace_contra = TypeVar('_T_replace_contra', contravariant=True)
_T_replace_co = TypeVar('_T_replace_co', covariant=True)


@runtime_checkable
class CanReplace(Protocol[_T_replace_contra, _T_replace_co]):
    """Support for `copy.replace` in Python 3.13+."""
    def __replace__(
        self,
        /,
        **changes: _T_replace_contra,
    ) -> _T_replace_co: ...


@runtime_checkable
class CanCopySelf(CanCopy['CanCopySelf'], Protocol):
    """Variant of `CanCopy` that returns `Self` (as it should)."""
    @override
    def __copy__(self, /) -> Self: ...


class CanDeepcopySelf(CanDeepcopy['CanDeepcopySelf'], Protocol):
    """Variant of `CanDeepcopy` that returns `Self` (as it should)."""
    @override
    def __deepcopy__(self, memo: dict[int, Any], /) -> Self: ...


@runtime_checkable
class CanReplaceSelf(
    CanReplace[_T_replace_contra, 'CanReplaceSelf[Any]'],
    Protocol[_T_replace_contra],
):
    """Variant of `CanReplace[T, Self]`."""
    @override
    def __replace__(self, /, **changes: _T_replace_contra) -> Self: ...


#
# Standard library `pickle`
#

_T_reduce_co = TypeVar(
    '_T_reduce_co',
    covariant=True,
    bound=str | tuple[Any, ...],
    default=str | tuple[Any, ...],
)


@runtime_checkable
class CanReduce(Protocol[_T_reduce_co]):
    @override
    def __reduce__(self, /) -> _T_reduce_co: ...


_T_reduce_ex_co = TypeVar(
    '_T_reduce_ex_co',
    covariant=True,
    bound=str | tuple[Any, ...],
    default=str | tuple[Any, ...],
)


@runtime_checkable
class CanReduceEx(Protocol[_T_reduce_ex_co]):
    @override
    def __reduce_ex__(self, protocol: CanIndex, /) -> _T_reduce_ex_co: ...


_T_getstate_co = TypeVar('_T_getstate_co', covariant=True)


@runtime_checkable
class CanGetstate(Protocol[_T_getstate_co]):
    def __getstate__(self, /) -> _T_getstate_co: ...


_T_setstate_contra = TypeVar('_T_setstate_contra', contravariant=True)


@runtime_checkable
class CanSetstate(Protocol[_T_setstate_contra]):
    def __setstate__(self, state: _T_setstate_contra, /) -> None: ...


_Ts_getnewargs = TypeVarTuple('_Ts_getnewargs')


@runtime_checkable
class CanGetnewargs(Protocol[Unpack[_Ts_getnewargs]]):
    def __new__(cls, /, *args: Unpack[_Ts_getnewargs]) -> Self: ...
    def __getnewargs__(self, /) -> tuple[Unpack[_Ts_getnewargs]]: ...


_Ts_getnewargs_ex = TypeVarTuple('_Ts_getnewargs_ex')
_T_getnewargs_ex = TypeVar('_T_getnewargs_ex')


@runtime_checkable
class CanGetnewargsEx(Protocol[Unpack[_Ts_getnewargs_ex], _T_getnewargs_ex]):
    def __new__(
        cls,
        /,
        *args: Unpack[_Ts_getnewargs_ex],
        **kwargs: _T_getnewargs_ex,
    ) -> Self: ...
    def __getnewargs_ex__(self, /) -> tuple[
        tuple[Unpack[_Ts_getnewargs_ex]],
        dict[str, _T_getnewargs_ex],
    ]: ...
