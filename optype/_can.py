from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias


if sys.version_info >= (3, 13):
    from typing import (
        ParamSpec,
        Protocol,
        Self,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        ParamSpec,
        Protocol,
        Self,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType

from optype._utils import set_module


_Ignored: TypeAlias = Any
_JustFalse: TypeAlias = Literal[False]
_JustTrue: TypeAlias = Literal[True]


#
# Type conversion
#

_R_bool = TypeVar(
    '_R_bool',
    _JustFalse,
    _JustTrue,
    bool,
    infer_variance=True,
    default=bool,
)


@set_module('optype')
@runtime_checkable
class CanBool(Protocol[_R_bool]):
    def __bool__(self, /) -> _R_bool: ...


_R_int = TypeVar('_R_int', infer_variance=True, bound=int, default=int)


@set_module('optype')
@runtime_checkable
class CanInt(Protocol[_R_int]):
    def __int__(self, /) -> _R_int: ...


@set_module('optype')
@runtime_checkable
class CanFloat(Protocol):
    def __float__(self, /) -> float: ...


@set_module('optype')
@runtime_checkable
class CanComplex(Protocol):
    def __complex__(self, /) -> complex: ...


_R_bytes = TypeVar('_R_bytes', infer_variance=True, bound=bytes, default=bytes)


@set_module('optype')
@runtime_checkable
class CanBytes(Protocol[_R_bytes]):
    """
    The `__bytes__: (CanBytes[Y]) -> Y` method is *co*variant on `+Y`.
    So if `__bytes__` returns an instance of a custom `bytes` subtype
    `Y <: bytes`, then `bytes()` will also return `Y` (i.e. no upcasting).
    """
    def __bytes__(self, /) -> _R_bytes: ...


_R_str = TypeVar('_R_str', infer_variance=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class CanStr(Protocol[_R_str]):
    """
    Each `object` has a *co*variant `__str__: (CanStr[Y=str]) -> Y` method on
    `+Y`. That means that if `__str__()` returns an instance of a custom `str`
    subtype `Y <: str`, then `str()` will also return `Y` (i.e. no upcasting).
    """
    @override
    def __str__(self, /) -> _R_str: ...


#
# Representation
#

@set_module('optype')
@runtime_checkable
class CanHash(Protocol):
    @override
    def __hash__(self, /) -> int: ...


_R_index = TypeVar('_R_index', infer_variance=True, bound=int, default=int)


@set_module('optype')
@runtime_checkable
class CanIndex(Protocol[_R_index]):
    def __index__(self, /) -> _R_index: ...


_R_repr = TypeVar('_R_repr', infer_variance=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class CanRepr(Protocol[_R_repr]):
    """
    Each `object` has a *co*variant `__repr__: (CanRepr[Y=str]) -> Y` method.
    That means that if `__repr__` returns an instance of a custom `str`
    subtype `Y <: str`, then `repr()` will also return `Y` (i.e. no upcasting).
    """
    @override
    def __repr__(self, /) -> _R_repr: ...


_T_format = TypeVar('_T_format', infer_variance=True, bound=str, default=str)
_R_format = TypeVar('_R_format', infer_variance=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class CanFormat(Protocol[_T_format, _R_format]):
    """
    Each `object` has a `__format__: (CanFormat[X, Y], X) -> Y` method, with
    `-X` *contra*variant, and `+Y` *co*variant. Both `X` and `Y` can be `str`
    or `str` subtypes. Note that `format()` *does not* upcast `Y` to `str`.
    """
    @override
    def __format__(self, fmt: _T_format, /) -> _R_format: ...


#
# Iteration
#


_V_next = TypeVar('_V_next', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanNext(Protocol[_V_next]):
    """
    Similar to `collections.abc.Iterator[V]`, but without the (often redundant)
    requirement to also have a `__iter__` method.
    """
    def __next__(self, /) -> _V_next: ...


_R_iter = TypeVar('_R_iter', infer_variance=True, bound=CanNext[Any])


@set_module('optype')
@runtime_checkable
class CanIter(Protocol[_R_iter]):
    """
    Similar to `collections.abc.Iterable[V]`, but more flexible in its
    return type.
    """
    def __iter__(self, /) -> _R_iter: ...


_V_iter_self = TypeVar('_V_iter_self', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIterSelf(
    CanNext[_V_iter_self],
    CanIter[CanNext[_V_iter_self]],
    Protocol[_V_iter_self],
):
    """
    Equivalent to `collections.abc.Iterator[V]`, minus the `abc` nonsense.
    """
    @override
    def __iter__(self, /) -> Self: ...


#
# Async iteration
#

_V_anext = TypeVar('_V_anext', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanANext(Protocol[_V_anext]):
    def __anext__(self, /) -> _V_anext: ...


_R_aiter = TypeVar('_R_aiter', infer_variance=True, bound=CanANext[Any])


@set_module('optype')
@runtime_checkable
class CanAIter(Protocol[_R_aiter]):
    def __aiter__(self, /) -> _R_aiter: ...


_V_aiter_self = TypeVar('_V_aiter_self', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanAIterSelf(
    CanANext[_V_aiter_self],
    CanAIter[CanANext[_V_aiter_self]],
    Protocol[_V_aiter_self],
):
    """A less inflexible variant of `collections.abc.AsyncIterator[T]`."""
    @override
    def __aiter__(self, /) -> Self: ...


#
# "Rich" comparison ops
#

_T_eq = TypeVar('_T_eq', infer_variance=True, default=object)
_R_eq = TypeVar('_R_eq', infer_variance=True, default=bool)


@set_module('optype')
@runtime_checkable
class CanEq(Protocol[_T_eq, _R_eq]):  # noqa: PLW1641
    """
    Unfortunately, `typeshed` (incorrectly) annotates `object.__eq__` as
    `(Self, object) -> bool`.
    As a counter-example, consider `numpy.ndarray`. It's `__eq__` method
    returns a boolean (mask) array of the same shape as the input array.
    Moreover, `numpy.ndarray` doesn't even implement `CanBool` (`bool()`
    raises a `TypeError` for shapes of size > 1).
    There is nothing wrong with this implementation, even though `typeshed`
    (incorrectly) won't allow it (because `numpy.ndarray <: object`).

    So in reality, it should be `__eq__: (Self, X, /) -> Y`, with `X` unbounded
    and *contra*variant, and `+Y` unbounded and *co*variant.
    """
    @override
    def __eq__(self, rhs: _T_eq, /) -> _R_eq: ...  # pyright:ignore[reportIncompatibleMethodOverride]


_T_ne = TypeVar('_T_ne', infer_variance=True, default=object)
_R_ne = TypeVar('_R_ne', infer_variance=True, default=bool)


@set_module('optype')
@runtime_checkable
class CanNe(Protocol[_T_ne, _R_ne]):
    """
    Just like `__eq__`, the `__ne__` method is incorrectly annotated in
    typeshed. Refer to `CanEq` for why this is.
    """
    @override
    def __ne__(self, rhs: _T_ne, /) -> _R_ne: ...  # pyright:ignore[reportIncompatibleMethodOverride]


_T_lt = TypeVar('_T_lt', infer_variance=True)
_R_lt = TypeVar('_R_lt', infer_variance=True, default=bool)


@set_module('optype')
@runtime_checkable
class CanLt(Protocol[_T_lt, _R_lt]):
    def __lt__(self, rhs: _T_lt, /) -> _R_lt: ...


_T_le = TypeVar('_T_le', infer_variance=True)
_R_le = TypeVar('_R_le', infer_variance=True, default=bool)


@set_module('optype')
@runtime_checkable
class CanLe(Protocol[_T_le, _R_le]):
    def __le__(self, rhs: _T_le, /) -> _R_le: ...


_T_gt = TypeVar('_T_gt', infer_variance=True)
_R_gt = TypeVar('_R_gt', infer_variance=True, default=bool)


@set_module('optype')
@runtime_checkable
class CanGt(Protocol[_T_gt, _R_gt]):
    def __gt__(self, rhs: _T_gt, /) -> _R_gt: ...


_T_ge = TypeVar('_T_ge', infer_variance=True)
_R_ge = TypeVar('_R_ge', infer_variance=True, default=bool)


@set_module('optype')
@runtime_checkable
class CanGe(Protocol[_T_ge, _R_ge]):
    def __ge__(self, rhs: _T_ge, /) -> _R_ge: ...


#
# Callables
#

_Pss_call = ParamSpec('_Pss_call')
_R_call = TypeVar('_R_call', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanCall(Protocol[_Pss_call, _R_call]):
    def __call__(
        self,
        /,
        *args: _Pss_call.args,
        **kwargs: _Pss_call.kwargs,
    ) -> _R_call: ...


#
# Dynamic attribute access
#

_N_getattr = TypeVar('_N_getattr', infer_variance=True, bound=str, default=str)
_V_getattr = TypeVar('_V_getattr', infer_variance=True, default=Any)


@set_module('optype')
@runtime_checkable
class CanGetattr(Protocol[_N_getattr, _V_getattr]):
    def __getattr__(self, name: _N_getattr, /) -> _V_getattr: ...


_N_getattribute = TypeVar(
    '_N_getattribute',
    infer_variance=True,
    bound=str,
    default=str,
)
_V_getattribute = TypeVar('_V_getattribute', infer_variance=True, default=Any)


@set_module('optype')
@runtime_checkable
class CanGetattribute(Protocol[_N_getattribute, _V_getattribute]):
    """Note that `isinstance(x, CanGetattribute)` is always true."""
    @override
    def __getattribute__(
        self,
        name: _N_getattribute,
        /,
    ) -> _V_getattribute: ...


_N_setattr = TypeVar('_N_setattr', infer_variance=True, bound=str, default=str)
_V_setattr = TypeVar('_V_setattr', infer_variance=True, default=Any)


@set_module('optype')
@runtime_checkable
class CanSetattr(Protocol[_N_setattr, _V_setattr]):
    """Note that `isinstance(x, CanSetattr)` is always true."""
    @override
    def __setattr__(
        self,
        name: _N_setattr,
        value: _V_setattr,
        /,
    ) -> _Ignored: ...


_N_delattr = TypeVar('_N_delattr', infer_variance=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class CanDelattr(Protocol[_N_delattr]):
    @override
    def __delattr__(self, name: _N_delattr, /) -> Any: ...


_R_dir = TypeVar(
    '_R_dir',
    infer_variance=True,
    bound=CanIter[Any],
    default=CanIter[CanIterSelf[str]],
)


@set_module('optype')
@runtime_checkable
class CanDir(Protocol[_R_dir]):
    @override
    def __dir__(self, /) -> _R_dir: ...


#
# Descriptors
#

_T_get = TypeVar('_T_get', infer_variance=True, bound=object)
_V_get = TypeVar('_V_get', infer_variance=True)
_VT_get = TypeVar('_VT_get', infer_variance=True, default=_V_get)


@set_module('optype')
@runtime_checkable
class CanGet(Protocol[_T_get, _V_get, _VT_get]):
    @overload
    def __get__(
        self,
        owner: _T_get,
        owner_type: type[_T_get] | None = ...,
        /,
    ) -> _V_get: ...
    @overload
    def __get__(self, owner: None, owner_type: type[_T_get], /) -> _VT_get: ...


_T_set = TypeVar('_T_set', infer_variance=True, bound=object)
_V_set = TypeVar('_V_set', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanSet(Protocol[_T_set, _V_set]):
    def __set__(self, owner: _T_set, value: _V_set, /) -> _Ignored: ...


_T_delete = TypeVar('_T_delete', infer_variance=True, bound=object)


@set_module('optype')
@runtime_checkable
class CanDelete(Protocol[_T_delete]):
    def __delete__(self, owner: _T_delete, /) -> _Ignored: ...


_T_set_name = TypeVar('_T_set_name', infer_variance=True, bound=object)
_N_set_name = TypeVar(
    '_N_set_name',
    infer_variance=True,
    bound=str,
    default=str,
)


@set_module('optype')
@runtime_checkable
class CanSetName(Protocol[_T_set_name, _N_set_name]):
    def __set_name__(
        self,
        owner_type: type[_T_set_name],
        name: _N_set_name,
        /,
    ) -> _Ignored: ...


#
# Containers
#

_R_len = TypeVar('_R_len', infer_variance=True, bound=int, default=int)


@set_module('optype')
@runtime_checkable
class CanLen(Protocol[_R_len]):
    def __len__(self, /) -> _R_len: ...


_R_length_hint = TypeVar(
    '_R_length_hint',
    infer_variance=True,
    bound=int,
    default=int,
)


@set_module('optype')
@runtime_checkable
class CanLengthHint(Protocol[_R_length_hint]):
    def __length_hint__(self, /) -> _R_length_hint: ...


_K_getitem = TypeVar('_K_getitem', infer_variance=True)
_V_getitem = TypeVar('_V_getitem', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanGetitem(Protocol[_K_getitem, _V_getitem]):
    def __getitem__(self, key: _K_getitem, /) -> _V_getitem: ...


_K_setitem = TypeVar('_K_setitem', infer_variance=True)
_V_setitem = TypeVar('_V_setitem', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanSetitem(Protocol[_K_setitem, _V_setitem]):
    def __setitem__(
        self,
        key: _K_setitem,
        value: _V_setitem,
        /,
    ) -> _Ignored: ...


_K_delitem = TypeVar('_K_delitem', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanDelitem(Protocol[_K_delitem]):
    def __delitem__(self, key: _K_delitem, /) -> None: ...


# theoretically not required to be iterable; but it probably should be
_R_reversed = TypeVar('_R_reversed', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanReversed(Protocol[_R_reversed]):
    def __reversed__(self, /) -> _R_reversed: ...


# theoretically not required to be hashable; but it probably should be
_K_contains = TypeVar(
    '_K_contains',
    infer_variance=True,
    default=object,
)
# could be set to e.g. _IsFalse empty (user-defined) container types
_R_contains = TypeVar(
    '_R_contains',
    _JustFalse,
    _JustTrue,
    bool,
    infer_variance=True,
    default=bool,
)


@set_module('optype')
@runtime_checkable
class CanContains(Protocol[_K_contains, _R_contains]):
    def __contains__(self, key: _K_contains, /) -> _R_contains: ...


_K_missing = TypeVar('_K_missing', infer_variance=True)
_D_missing = TypeVar('_D_missing', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanMissing(Protocol[_K_missing, _D_missing]):
    def __missing__(self, key: _K_missing, /) -> _D_missing: ...


_K_get_missing = TypeVar('_K_get_missing', infer_variance=True)
_V_get_missing = TypeVar('_V_get_missing', infer_variance=True)
_D_get_missing = TypeVar(
    '_D_get_missing',
    infer_variance=True,
    default=_V_get_missing,
)


@set_module('optype')
@runtime_checkable
class CanGetMissing(
    CanGetitem[_K_get_missing, _V_get_missing],
    CanMissing[_K_get_missing, _D_get_missing],
    Protocol[_K_get_missing, _V_get_missing, _D_get_missing],
): ...


_K_sequence = TypeVar(
    '_K_sequence',
    infer_variance=True,
    bound=CanIndex | slice,
)
_V_sequence = TypeVar('_V_sequence', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanSequence(
    CanLen,
    CanGetitem[_K_sequence, _V_sequence],
    Protocol[_K_sequence, _V_sequence],
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

_T_add = TypeVar('_T_add', infer_variance=True)
_R_add = TypeVar('_R_add', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanAdd(Protocol[_T_add, _R_add]):
    def __add__(self, rhs: _T_add, /) -> _R_add: ...


_T_sub = TypeVar('_T_sub', infer_variance=True)
_R_sub = TypeVar('_R_sub', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanSub(Protocol[_T_sub, _R_sub]):
    def __sub__(self, rhs: _T_sub, /) -> _R_sub: ...


_T_mul = TypeVar('_T_mul', infer_variance=True)
_R_mul = TypeVar('_R_mul', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanMul(Protocol[_T_mul, _R_mul]):
    def __mul__(self, rhs: _T_mul, /) -> _R_mul: ...


_T_matmul = TypeVar('_T_matmul', infer_variance=True)
_R_matmul = TypeVar('_R_matmul', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanMatmul(Protocol[_T_matmul, _R_matmul]):
    def __matmul__(self, rhs: _T_matmul, /) -> _R_matmul: ...


_T_truediv = TypeVar('_T_truediv', infer_variance=True)
_R_truediv = TypeVar('_R_truediv', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanTruediv(Protocol[_T_truediv, _R_truediv]):
    def __truediv__(self, rhs: _T_truediv, /) -> _R_truediv: ...


_T_floordiv = TypeVar('_T_floordiv', infer_variance=True)
_R_floordiv = TypeVar('_R_floordiv', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanFloordiv(Protocol[_T_floordiv, _R_floordiv]):
    def __floordiv__(self, rhs: _T_floordiv, /) -> _R_floordiv: ...


_T_mod = TypeVar('_T_mod', infer_variance=True)
_R_mod = TypeVar('_R_mod', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanMod(Protocol[_T_mod, _R_mod]):
    def __mod__(self, rhs: _T_mod, /) -> _R_mod: ...


_T_divmod = TypeVar('_T_divmod', infer_variance=True)
_R_divmod = TypeVar('_R_divmod', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanDivmod(Protocol[_T_divmod, _R_divmod]):
    def __divmod__(self, rhs: _T_divmod, /) -> _R_divmod: ...


_T_pow2 = TypeVar('_T_pow2', infer_variance=True)
_R_pow2 = TypeVar('_R_pow2', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanPow2(Protocol[_T_pow2, _R_pow2]):
    @overload
    def __pow__(self, exp: _T_pow2, /) -> _R_pow2: ...
    @overload
    def __pow__(self, exp: _T_pow2, mod: None = ..., /) -> _R_pow2: ...


_T_pow3_exp = TypeVar('_T_pow3_exp', infer_variance=True)
_T_pow3_mod = TypeVar('_T_pow3_mod', infer_variance=True)
_R_pow3 = TypeVar('_R_pow3', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanPow3(Protocol[_T_pow3_exp, _T_pow3_mod, _R_pow3]):
    def __pow__(self, exp: _T_pow3_exp, mod: _T_pow3_mod, /) -> _R_pow3: ...


_T_pow_exp = TypeVar('_T_pow_exp', infer_variance=True)
_T_pow_mod = TypeVar('_T_pow_mod', infer_variance=True)
_R_pow = TypeVar('_R_pow', infer_variance=True)
_R_pow_mod = TypeVar('_R_pow_mod', infer_variance=True, default=_R_pow)


@set_module('optype')
@runtime_checkable
class CanPow(
    CanPow2[_T_pow_exp, _R_pow],
    CanPow3[_T_pow_exp, _T_pow_mod, _R_pow_mod],
    Protocol[_T_pow_exp, _T_pow_mod, _R_pow, _R_pow_mod],
):
    @overload
    def __pow__(self, exp: _T_pow_exp, /) -> _R_pow: ...
    @overload
    def __pow__(self, exp: _T_pow_exp, mod: None = ..., /) -> _R_pow: ...
    @overload
    def __pow__(self, exp: _T_pow_exp, mod: _T_pow_mod, /) -> _R_pow_mod: ...


_T_lshift = TypeVar('_T_lshift', infer_variance=True)
_R_lshift = TypeVar('_R_lshift', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanLshift(Protocol[_T_lshift, _R_lshift]):
    def __lshift__(self, rhs: _T_lshift, /) -> _R_lshift: ...


_T_rshift = TypeVar('_T_rshift', infer_variance=True)
_R_rshift = TypeVar('_R_rshift', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRshift(Protocol[_T_rshift, _R_rshift]):
    def __rshift__(self, rhs: _T_rshift, /) -> _R_rshift: ...


_T_and = TypeVar('_T_and', infer_variance=True)
_R_and = TypeVar('_R_and', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanAnd(Protocol[_T_and, _R_and]):
    def __and__(self, rhs: _T_and, /) -> _R_and: ...


_T_xor = TypeVar('_T_xor', infer_variance=True)
_R_xor = TypeVar('_R_xor', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanXor(Protocol[_T_xor, _R_xor]):
    def __xor__(self, rhs: _T_xor, /) -> _R_xor: ...


_T_or = TypeVar('_T_or', infer_variance=True)
_R_or = TypeVar('_R_or', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanOr(Protocol[_T_or, _R_or]):
    def __or__(self, rhs: _T_or, /) -> _R_or: ...


#
# Reflected numeric ops
#

_T_radd = TypeVar('_T_radd', infer_variance=True)
_R_radd = TypeVar('_R_radd', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRAdd(Protocol[_T_radd, _R_radd]):
    def __radd__(self, rhs: _T_radd, /) -> _R_radd: ...


_T_rsub = TypeVar('_T_rsub', infer_variance=True)
_R_rsub = TypeVar('_R_rsub', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRSub(Protocol[_T_rsub, _R_rsub]):
    def __rsub__(self, rhs: _T_rsub, /) -> _R_rsub: ...


_T_rmul = TypeVar('_T_rmul', infer_variance=True)
_R_rmul = TypeVar('_R_rmul', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRMul(Protocol[_T_rmul, _R_rmul]):
    def __rmul__(self, rhs: _T_rmul, /) -> _R_rmul: ...


_T_rmatmul = TypeVar('_T_rmatmul', infer_variance=True)
_R_rmatmul = TypeVar('_R_rmatmul', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRMatmul(Protocol[_T_rmatmul, _R_rmatmul]):
    def __rmatmul__(self, rhs: _T_rmatmul, /) -> _R_rmatmul: ...


_T_rtruediv = TypeVar('_T_rtruediv', infer_variance=True)
_R_rtruediv = TypeVar('_R_rtruediv', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRTruediv(Protocol[_T_rtruediv, _R_rtruediv]):
    def __rtruediv__(self, rhs: _T_rtruediv, /) -> _R_rtruediv: ...


_T_rfloordiv = TypeVar('_T_rfloordiv', infer_variance=True)
_R_rfloordiv = TypeVar('_R_rfloordiv', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRFloordiv(Protocol[_T_rfloordiv, _R_rfloordiv]):
    def __rfloordiv__(self, rhs: _T_rfloordiv, /) -> _R_rfloordiv: ...


_T_rmod = TypeVar('_T_rmod', infer_variance=True)
_R_rmod = TypeVar('_R_rmod', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRMod(Protocol[_T_rmod, _R_rmod]):
    def __rmod__(self, rhs: _T_rmod, /) -> _R_rmod: ...


_T_rdivmod = TypeVar('_T_rdivmod', infer_variance=True)
_R_rdivmod = TypeVar('_R_rdivmod', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRDivmod(Protocol[_T_rdivmod, _R_rdivmod]):
    def __rdivmod__(self, rhs: _T_rdivmod, /) -> _R_rdivmod: ...


_T_rpow = TypeVar('_T_rpow', infer_variance=True)
_R_rpow = TypeVar('_R_rpow', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRPow(Protocol[_T_rpow, _R_rpow]):
    def __rpow__(self, x: _T_rpow, /) -> _R_rpow: ...


_T_rlshift = TypeVar('_T_rlshift', infer_variance=True)
_R_rlshift = TypeVar('_R_rlshift', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRLshift(Protocol[_T_rlshift, _R_rlshift]):
    def __rlshift__(self, rhs: _T_rlshift, /) -> _R_rlshift: ...


_T_rrshift = TypeVar('_T_rrshift', infer_variance=True)
_R_rrshift = TypeVar('_R_rrshift', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRRshift(Protocol[_T_rrshift, _R_rrshift]):
    def __rrshift__(self, rhs: _T_rrshift, /) -> _R_rrshift: ...


_T_rand = TypeVar('_T_rand', infer_variance=True)
_R_rand = TypeVar('_R_rand', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRAnd(Protocol[_T_rand, _R_rand]):
    def __rand__(self, rhs: _T_rand, /) -> _R_rand: ...


_T_rxor = TypeVar('_T_rxor', infer_variance=True)
_R_rxor = TypeVar('_R_rxor', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanRXor(Protocol[_T_rxor, _R_rxor]):
    def __rxor__(self, rhs: _T_rxor, /) -> _R_rxor: ...


_T_ror = TypeVar('_T_ror', infer_variance=True)
_R_ror = TypeVar('_R_ror', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanROr(Protocol[_T_ror, _R_ror]):
    def __ror__(self, rhs: _T_ror, /) -> _R_ror: ...


#
# Augmented numeric ops
#

_T_iadd = TypeVar('_T_iadd', infer_variance=True)
_R_iadd = TypeVar('_R_iadd', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIAdd(Protocol[_T_iadd, _R_iadd]):
    def __iadd__(self, rhs: _T_iadd, /) -> _R_iadd: ...


@set_module('optype')
@runtime_checkable
class CanIAddSelf(CanIAdd[_T_iadd, 'CanIAddSelf[Any]'], Protocol[_T_iadd]):
    @override
    def __iadd__(self, rhs: _T_iadd, /) -> Self: ...


_T_isub = TypeVar('_T_isub', infer_variance=True)
_R_isub = TypeVar('_R_isub', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanISub(Protocol[_T_isub, _R_isub]):
    def __isub__(self, rhs: _T_isub, /) -> _R_isub: ...


@set_module('optype')
@runtime_checkable
class CanISubSelf(CanISub[_T_isub, 'CanISubSelf[Any]'], Protocol[_T_isub]):
    @override
    def __isub__(self, rhs: _T_isub, /) -> Self: ...


_T_imul = TypeVar('_T_imul', infer_variance=True)
_R_imul = TypeVar('_R_imul', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIMul(Protocol[_T_imul, _R_imul]):
    def __imul__(self, rhs: _T_imul, /) -> _R_imul: ...


@set_module('optype')
@runtime_checkable
class CanIMulSelf(CanIMul[_T_imul, 'CanIMulSelf[Any]'], Protocol[_T_imul]):
    @override
    def __imul__(self, rhs: _T_imul, /) -> Self: ...


_T_imatmul = TypeVar('_T_imatmul', infer_variance=True)
_R_imatmul = TypeVar('_R_imatmul', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIMatmul(Protocol[_T_imatmul, _R_imatmul]):
    def __imatmul__(self, rhs: _T_imatmul, /) -> _R_imatmul: ...


@set_module('optype')
@runtime_checkable
class CanIMatmulSelf(
    CanIMatmul[_T_imatmul, 'CanIMatmulSelf[Any]'],
    Protocol[_T_imatmul],
):
    @override
    def __imatmul__(self, rhs: _T_imatmul, /) -> Self: ...


_T_itruediv = TypeVar('_T_itruediv', infer_variance=True)
_R_itruediv = TypeVar('_R_itruediv', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanITruediv(Protocol[_T_itruediv, _R_itruediv]):
    def __itruediv__(self, rhs: _T_itruediv, /) -> _R_itruediv: ...


@set_module('optype')
@runtime_checkable
class CanITruedivSelf(
    CanITruediv[_T_itruediv, 'CanITruedivSelf[Any]'],
    Protocol[_T_itruediv],
):
    @override
    def __itruediv__(self, rhs: _T_itruediv, /) -> Self: ...


_T_ifloordiv = TypeVar('_T_ifloordiv', infer_variance=True)
_R_ifloordiv = TypeVar('_R_ifloordiv', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIFloordiv(Protocol[_T_ifloordiv, _R_ifloordiv]):
    def __ifloordiv__(self, rhs: _T_ifloordiv, /) -> _R_ifloordiv: ...


@set_module('optype')
@runtime_checkable
class CanIFloordivSelf(
    CanIFloordiv[_T_ifloordiv, 'CanIFloordivSelf[Any]'],
    Protocol[_T_ifloordiv],
):
    @override
    def __ifloordiv__(self, rhs: _T_ifloordiv, /) -> Self: ...


_T_imod = TypeVar('_T_imod', infer_variance=True)
_R_imod = TypeVar('_R_imod', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIMod(Protocol[_T_imod, _R_imod]):
    def __imod__(self, rhs: _T_imod, /) -> _R_imod: ...


@set_module('optype')
@runtime_checkable
class CanIModSelf(CanIMod[_T_imod, 'CanIModSelf[Any]'], Protocol[_T_imod]):
    @override
    def __imod__(self, rhs: _T_imod, /) -> Self: ...


_T_ipow = TypeVar('_T_ipow', infer_variance=True)
_R_ipow = TypeVar('_R_ipow', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIPow(Protocol[_T_ipow, _R_ipow]):
    # no augmented pow/3 exists
    def __ipow__(self, rhs: _T_ipow, /) -> _R_ipow: ...


@set_module('optype')
@runtime_checkable
class CanIPowSelf(CanIPow[_T_ipow, 'CanIPowSelf[Any]'], Protocol[_T_ipow]):
    @override
    def __ipow__(self, rhs: _T_ipow, /) -> Self: ...


_T_ilshift = TypeVar('_T_ilshift', infer_variance=True)
_R_ilshift = TypeVar('_R_ilshift', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanILshift(Protocol[_T_ilshift, _R_ilshift]):
    def __ilshift__(self, rhs: _T_ilshift, /) -> _R_ilshift: ...


@set_module('optype')
@runtime_checkable
class CanILshiftSelf(
    CanILshift[_T_ilshift, 'CanILshiftSelf[Any]'],
    Protocol[_T_ilshift],
):
    @override
    def __ilshift__(self, rhs: _T_ilshift, /) -> Self: ...


_T_irshift = TypeVar('_T_irshift', infer_variance=True)
_R_irshift = TypeVar('_R_irshift', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIRshift(Protocol[_T_irshift, _R_irshift]):
    def __irshift__(self, rhs: _T_irshift, /) -> _R_irshift: ...


@set_module('optype')
@runtime_checkable
class CanIRshiftSelf(
    CanIRshift[_T_irshift, 'CanIRshiftSelf[Any]'],
    Protocol[_T_irshift],
):
    @override
    def __irshift__(self, rhs: _T_irshift, /) -> Self: ...


_T_iand = TypeVar('_T_iand', infer_variance=True)
_R_iand = TypeVar('_R_iand', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIAnd(Protocol[_T_iand, _R_iand]):
    def __iand__(self, rhs: _T_iand, /) -> _R_iand: ...


@set_module('optype')
@runtime_checkable
class CanIAndSelf(CanIAnd[_T_iand, 'CanIAndSelf[Any]'], Protocol[_T_iand]):
    @override
    def __iand__(self, rhs: _T_iand, /) -> Self: ...


_T_ixor = TypeVar('_T_ixor', infer_variance=True)
_R_ixor = TypeVar('_R_ixor', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIXor(Protocol[_T_ixor, _R_ixor]):
    def __ixor__(self, rhs: _T_ixor, /) -> _R_ixor: ...


@set_module('optype')
@runtime_checkable
class CanIXorSelf(CanIXor[_T_ixor, 'CanIXorSelf[Any]'], Protocol[_T_ixor]):
    @override
    def __ixor__(self, rhs: _T_ixor, /) -> Self: ...


_T_ior = TypeVar('_T_ior', infer_variance=True)
_R_ior = TypeVar('_R_ior', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanIOr(Protocol[_T_ior, _R_ior]):
    def __ior__(self, rhs: _T_ior, /) -> _R_ior: ...


@set_module('optype')
@runtime_checkable
class CanIOrSelf(CanIOr[_T_ior, 'CanIOrSelf[Any]'], Protocol[_T_ior]):
    @override
    def __ior__(self, rhs: _T_ior, /) -> Self: ...


#
# Unary arithmetic ops
#

_R_neg = TypeVar('_R_neg', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanNeg(Protocol[_R_neg]):
    def __neg__(self, /) -> _R_neg: ...


@set_module('optype')
@runtime_checkable
class CanNegSelf(CanNeg['CanNegSelf'], Protocol):
    @override
    def __neg__(self, /) -> Self: ...


_R_pos = TypeVar('_R_pos', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanPos(Protocol[_R_pos]):
    def __pos__(self, /) -> _R_pos: ...


@set_module('optype')
@runtime_checkable
class CanPosSelf(CanPos['CanPosSelf'], Protocol):
    @override
    def __pos__(self, /) -> Self: ...


_R_abs = TypeVar('_R_abs', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanAbs(Protocol[_R_abs]):
    def __abs__(self, /) -> _R_abs: ...


@set_module('optype')
@runtime_checkable
class CanAbsSelf(CanAbs['CanAbsSelf'], Protocol):
    @override
    def __abs__(self, /) -> Self: ...


_R_invert = TypeVar('_R_invert', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanInvert(Protocol[_R_invert]):
    def __invert__(self, /) -> _R_invert: ...


@set_module('optype')
@runtime_checkable
class CanInvertSelf(CanInvert['CanInvertSelf'], Protocol):
    @override
    def __invert__(self, /) -> Self: ...


#
# Rounding
#

_R_round1 = TypeVar('_R_round1', infer_variance=True, default=int)


@set_module('optype')
@runtime_checkable
class CanRound1(Protocol[_R_round1]):
    @overload
    def __round__(self, /) -> _R_round1: ...
    @overload
    def __round__(self, /, ndigits: None = None) -> _R_round1: ...


_T_round2 = TypeVar('_T_round2', infer_variance=True, default=int)
_RT_round2 = TypeVar('_RT_round2', infer_variance=True, default=float)


@set_module('optype')
@runtime_checkable
class CanRound2(Protocol[_T_round2, _RT_round2]):
    def __round__(self, /, ndigits: _T_round2) -> _RT_round2: ...


_T_round = TypeVar('_T_round', infer_variance=True, default=int)
_R_round = TypeVar('_R_round', infer_variance=True, default=int)
_RT_round = TypeVar('_RT_round', infer_variance=True, default=float)


@set_module('optype')
@runtime_checkable
class CanRound(
    CanRound1[_R_round],
    CanRound2[_T_round, _RT_round],
    Protocol[_T_round, _R_round, _RT_round],
):
    @overload
    def __round__(self, /) -> _R_round: ...
    @overload
    def __round__(self, /, ndigits: None = None) -> _R_round: ...
    @overload
    def __round__(self, /, ndigits: _T_round) -> _RT_round: ...


_R_trunc = TypeVar('_R_trunc', infer_variance=True, default=int)


@set_module('optype')
@runtime_checkable
class CanTrunc(Protocol[_R_trunc]):
    def __trunc__(self, /) -> _R_trunc: ...


_R_floor = TypeVar('_R_floor', infer_variance=True, default=int)


@set_module('optype')
@runtime_checkable
class CanFloor(Protocol[_R_floor]):
    def __floor__(self, /) -> _R_floor: ...


_R_ceil = TypeVar('_R_ceil', infer_variance=True, default=int)


@set_module('optype')
@runtime_checkable
class CanCeil(Protocol[_R_ceil]):
    def __ceil__(self, /) -> _R_ceil: ...


#
# Context managers
#

_C_enter = TypeVar('_C_enter', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanEnter(Protocol[_C_enter]):
    def __enter__(self, /) -> _C_enter: ...


@set_module('optype')
@runtime_checkable
class CanEnterSelf(CanEnter['CanEnterSelf'], Protocol):
    @override
    def __enter__(self, /) -> Self: ...  # pyright: ignore[reportMissingSuperCall]


_E_exit = TypeVar('_E_exit', bound=BaseException)
_R_exit = TypeVar('_R_exit', infer_variance=True, default=None)


@set_module('optype')
@runtime_checkable
class CanExit(Protocol[_R_exit]):
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
    ) -> _R_exit: ...


_C_with = TypeVar('_C_with', infer_variance=True)
_R_with = TypeVar('_R_with', infer_variance=True, default=None)


@set_module('optype')
@runtime_checkable
class CanWith(CanEnter[_C_with], CanExit[_R_with], Protocol[_C_with, _R_with]):
    """
    The intersection type of `CanEnter` and `CanExit`, i.e.
    `CanWith[C, R=None] = CanEnter[C] & CanExit[R]`.
    """


_R_with_self = TypeVar('_R_with_self', infer_variance=True, default=None)


@set_module('optype')
@runtime_checkable
class CanWithSelf(CanEnterSelf, CanExit[_R_with_self], Protocol[_R_with_self]):
    """
    The intersection type of `CanEnterSelf` and `CanExit`, i.e.
    `CanWithSelf[R=None] = CanEnterSelf & CanExit[R]`.
    """


#
# Async context managers
#

_C_aenter = TypeVar('_C_aenter', infer_variance=True)


@set_module('optype')
@runtime_checkable
class CanAEnter(Protocol[_C_aenter]):
    def __aenter__(self, /) -> CanAwait[_C_aenter]: ...


@set_module('optype')
@runtime_checkable
class CanAEnterSelf(CanAEnter['CanAEnterSelf'], Protocol):
    @override
    def __aenter__(self, /) -> CanAwait[Self]: ...


_E_aexit = TypeVar('_E_aexit', bound=BaseException)
_R_aexit = TypeVar('_R_aexit', infer_variance=True, default=None)


@set_module('optype')
@runtime_checkable
class CanAExit(Protocol[_R_aexit]):
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
    ) -> CanAwait[_R_aexit]: ...


_C_async_with = TypeVar('_C_async_with', infer_variance=True)
_R_async_with = TypeVar('_R_async_with', infer_variance=True, default=None)


@set_module('optype')
@runtime_checkable
class CanAsyncWith(
    CanAEnter[_C_async_with],
    CanAExit[_R_async_with],
    Protocol[_C_async_with, _R_async_with],
):
    """
    The intersection type of `CanAEnter` and `CanAExit`, i.e.
    `CanAsyncWith[C, R=None] = CanAEnter[C] & CanAExit[R]`.
    """


@set_module('optype')
@runtime_checkable
class CanAsyncWithSelf(
    CanAEnterSelf,
    CanAExit[_R_async_with],
    Protocol[_R_async_with],
):
    """
    The intersection type of `CanAEnterSelf` and `CanAExit`, i.e.
    `CanAsyncWithSelf[R=None] = CanAEnterSelf & CanAExit[R]`.
    """


#
# Buffer protocol
#

_T_buffer = TypeVar('_T_buffer', infer_variance=True, bound=int, default=int)


@set_module('optype')
@runtime_checkable
class CanBuffer(Protocol[_T_buffer]):
    def __buffer__(self, buffer: _T_buffer, /) -> memoryview: ...


@set_module('optype')
@runtime_checkable
class CanReleaseBuffer(Protocol):
    def __release_buffer__(self, buffer: memoryview, /) -> None: ...


#
# Awaitables
#

_R_await = TypeVar('_R_await', infer_variance=True)

# This should be `asyncio.Future[typing.Any] | None`. But that would make this
# incompatible with `collections.abc.Awaitable` -- it (annoyingly) uses `Any`:
# https://github.com/python/typeshed/blob/587ad6/stdlib/asyncio/futures.pyi#L51
_FutureOrNone: TypeAlias = Any
_AsyncGen: TypeAlias = 'Generator[_FutureOrNone, None, _R_await]'


@set_module('optype')
@runtime_checkable
class CanAwait(Protocol[_R_await]):
    # Technically speaking, this can return any
    # `CanNext[None | asyncio.Future[Any]]`. But in theory, the return value
    # of generators are currently impossible to type, because the return value
    # of a `yield from _` is # piggybacked using a `raise StopIteration(value)`
    # from `__next__`. So that also makes `__await__` theoretically
    # impossible to type. In practice, typecheckers work around that, by
    # accepting the lie called `collections.abc.Generator`...
    @overload
    def __await__(self: CanAwait[None], /) -> CanNext[_FutureOrNone]: ...
    @overload
    def __await__(self: CanAwait[_R_await], /) -> _AsyncGen[_R_await]: ...
