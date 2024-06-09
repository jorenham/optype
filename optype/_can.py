# ruff: noqa: PYI034
import sys
from collections.abc import Generator
from types import TracebackType
from typing import (
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
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        override,
    )


_Ignored: TypeAlias = Any

_T_co = TypeVar('_T_co', bound=object, covariant=True)
_T_contra = TypeVar('_T_contra', bound=object, contravariant=True)

_V = TypeVar('_V')
_V_co = TypeVar('_V_co', covariant=True)
_V_contra = TypeVar('_V_contra', contravariant=True)
_V_exc = TypeVar('_V_exc', bound=BaseException)
_V_int_contra = TypeVar('_V_int_contra', bound=int, contravariant=True)
_V_anext_co = TypeVar('_V_anext_co', bound='CanANext[Any]', covariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_X_co = TypeVar('_X_co', covariant=True)
_X_contra = TypeVar('_X_contra', contravariant=True)
_Xs = TypeVarTuple('_Xs')


# Iterator types
# https://docs.python.org/3/library/stdtypes.html#iterator-types

_T_next_co = TypeVar('_T_next_co', covariant=True)


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


# 3.3.1. Basic customization
# https://docs.python.org/3/reference/datamodel.html#basic-customization


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
    default=_T_format_contra,
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


@runtime_checkable
class CanHash(Protocol):
    @override
    def __hash__(self, /) -> int: ...


# 3.3.1. Basic customization - Rich comparison method
# https://docs.python.org/3/reference/datamodel.html#object.__lt__


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


# 3.3.2. Customizing attribute access
# https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access


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


# 3.3.2.2. Implementing Descriptors
# https://docs.python.org/3/reference/datamodel.html#implementing-descriptors


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


# 3.3.3. Customizing class creation
# https://docs.python.org/3/reference/datamodel.html#customizing-class-creation


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


# 3.3.6. Emulating callable objects
# https://docs.python.org/3/reference/datamodel.html#emulating-callable-objects


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


# 3.3.7. Emulating container types
# https://docs.python.org/3/reference/datamodel.html#emulating-container-types


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


# 3.3.8. Emulating numeric types
# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

@runtime_checkable
class CanAdd(Protocol[_X_contra, _Y_co]):
    def __add__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanSub(Protocol[_X_contra, _Y_co]):
    def __sub__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanMul(Protocol[_X_contra, _Y_co]):
    def __mul__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanMatmul(Protocol[_X_contra, _Y_co]):
    def __matmul__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanTruediv(Protocol[_X_contra, _Y_co]):
    def __truediv__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanFloordiv(Protocol[_X_contra, _Y_co]):
    def __floordiv__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanMod(Protocol[_X_contra, _Y_co]):
    def __mod__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanDivmod(Protocol[_X_contra, _Y_co]):
    def __divmod__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanPow2(Protocol[_X_contra, _Y_co]):
    def __pow__(self, rhs: _X_contra, /) -> _Y_co: ...


_VY_co = TypeVar('_VY_co', covariant=True)


@runtime_checkable
class CanPow3(Protocol[_X_contra, _V_contra, _VY_co]):
    def __pow__(self, x: _X_contra, mod: _V_contra, /) -> _VY_co: ...


@runtime_checkable
class CanPow(
    CanPow2[_X_contra, _Y_co],
    CanPow3[_X_contra, _V_contra, _VY_co],
    Protocol[_X_contra, _V_contra, _Y_co, _VY_co],
):
    @overload
    def __pow__(self, rhs: _X_contra, /) -> _Y_co: ...
    @overload
    def __pow__(self, x: _X_contra, mod: _V_contra, /) -> _VY_co: ...


@runtime_checkable
class CanLshift(Protocol[_X_contra, _Y_co]):
    def __lshift__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRshift(Protocol[_X_contra, _Y_co]):
    def __rshift__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanAnd(Protocol[_X_contra, _Y_co]):
    def __and__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanXor(Protocol[_X_contra, _Y_co]):
    def __xor__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanOr(Protocol[_X_contra, _Y_co]):
    def __or__(self, rhs: _X_contra, /) -> _Y_co: ...


# reflected

@runtime_checkable
class CanRAdd(Protocol[_X_contra, _Y_co]):
    def __radd__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRSub(Protocol[_X_contra, _Y_co]):
    def __rsub__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRMul(Protocol[_X_contra, _Y_co]):
    def __rmul__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRMatmul(Protocol[_X_contra, _Y_co]):
    def __rmatmul__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRTruediv(Protocol[_X_contra, _Y_co]):
    def __rtruediv__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRFloordiv(Protocol[_X_contra, _Y_co]):
    def __rfloordiv__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRMod(Protocol[_X_contra, _Y_co]):
    def __rmod__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRDivmod(Protocol[_X_contra, _Y_co]):
    def __rdivmod__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRPow(Protocol[_X_contra, _Y_co]):
    def __rpow__(self, x: _X_contra) -> _Y_co: ...


@runtime_checkable
class CanRLshift(Protocol[_X_contra, _Y_co]):
    def __rlshift__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRRshift(Protocol[_X_contra, _Y_co]):
    def __rrshift__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRAnd(Protocol[_X_contra, _Y_co]):
    def __rand__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanRXor(Protocol[_X_contra, _Y_co]):
    def __rxor__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanROr(Protocol[_X_contra, _Y_co]):
    def __ror__(self, rhs: _X_contra, /) -> _Y_co: ...


# augmented / in-place

@runtime_checkable
class CanIAdd(Protocol[_X_contra, _Y_co]):
    def __iadd__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanISub(Protocol[_X_contra, _Y_co]):
    def __isub__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIMul(Protocol[_X_contra, _Y_co]):
    def __imul__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIMatmul(Protocol[_X_contra, _Y_co]):
    def __imatmul__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanITruediv(Protocol[_X_contra, _Y_co]):
    def __itruediv__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIFloordiv(Protocol[_X_contra, _Y_co]):
    def __ifloordiv__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIMod(Protocol[_X_contra, _Y_co]):
    def __imod__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIPow(Protocol[_X_contra, _Y_co]):
    # no augmented pow/3 exists
    def __ipow__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanILshift(Protocol[_X_contra, _Y_co]):
    def __ilshift__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIRshift(Protocol[_X_contra, _Y_co]):
    def __irshift__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIAnd(Protocol[_X_contra, _Y_co]):
    def __iand__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIXor(Protocol[_X_contra, _Y_co]):
    def __ixor__(self, rhs: _X_contra, /) -> _Y_co: ...


@runtime_checkable
class CanIOr(Protocol[_X_contra, _Y_co]):
    def __ior__(self, rhs: _X_contra, /) -> _Y_co: ...


# unary arithmetic

@runtime_checkable
class CanNeg(Protocol[_Y_co]):
    def __neg__(self, /) -> _Y_co: ...


@runtime_checkable
class CanPos(Protocol[_Y_co]):
    def __pos__(self, /) -> _Y_co: ...


@runtime_checkable
class CanAbs(Protocol[_Y_co]):
    def __abs__(self, /) -> _Y_co: ...


@runtime_checkable
class CanInvert(Protocol[_Y_co]):
    def __invert__(self, /) -> _Y_co: ...


# numeric conversion

@runtime_checkable
class CanComplex(Protocol):
    def __complex__(self, /) -> complex: ...


@runtime_checkable
class CanFloat(Protocol):
    def __float__(self, /) -> float: ...


@runtime_checkable
class CanInt(Protocol):
    def __int__(self, /) -> int: ...


@runtime_checkable
class CanIndex(Protocol):
    def __index__(self, /) -> int: ...

# rounding


@runtime_checkable
class CanRound1(Protocol[_Y_co]):
    @overload
    def __round__(self, /) -> _Y_co: ...
    @overload
    def __round__(self, ndigits: None = ...) -> _Y_co: ...


@runtime_checkable
class CanRound2(Protocol[_V_contra, _VY_co]):
    def __round__(self, ndigits: _V_contra) -> _VY_co: ...


@runtime_checkable
class CanRound(
    CanRound1[_Y_co],
    CanRound2[_V_contra, _VY_co],
    Protocol[_V_contra, _Y_co, _VY_co],
):
    @overload
    def __round__(self, /) -> _Y_co: ...
    @overload
    def __round__(self, ndigits: None = ...) -> _Y_co: ...
    @overload
    def __round__(self, ndigits: _V_contra) -> _VY_co: ...


@runtime_checkable
class CanTrunc(Protocol[_Y_co]):
    def __trunc__(self, /) -> _Y_co: ...


@runtime_checkable
class CanFloor(Protocol[_Y_co]):
    def __floor__(self, /) -> _Y_co: ...


@runtime_checkable
class CanCeil(Protocol[_Y_co]):
    def __ceil__(self, /) -> _Y_co: ...


# 3.3.9. With Statement Context Managers
# https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers


@runtime_checkable
class CanEnter(Protocol[_X_co]):
    def __enter__(self, /) -> _X_co: ...


@runtime_checkable
class CanExit(Protocol[_Y_co]):
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
        exc_type: type[_V_exc],
        exc_instance: _V_exc,
        exc_traceback: TracebackType,
        /,
    ) -> _Y_co: ...


@runtime_checkable
class CanWith(CanEnter[_X_co], CanExit[_Y_co], Protocol[_X_co, _Y_co]): ...


# 3.3.11. Emulating buffer types
# https://docs.python.org/3/reference/datamodel.html#emulating-buffer-types


@runtime_checkable
class CanBuffer(Protocol[_V_int_contra]):
    def __buffer__(self, buffer: _V_int_contra, /) -> memoryview: ...


@runtime_checkable
class CanReleaseBuffer(Protocol):
    def __release_buffer__(self, buffer: memoryview, /) -> None: ...


# 3.4.1. Awaitable Objects
# https://docs.python.org/3/reference/datamodel.html#awaitable-objects


# This should be `None | asyncio.Future[Any]`. But that would make this
# incompatible with `collections.abc.Awaitable`, because it (annoyingly)
# uses `Any`...
_MaybeFuture: TypeAlias = Any


@runtime_checkable
class CanAwait(Protocol[_V_co]):
    # Technically speaking, this can return any
    # `CanNext[None | asyncio.Future[Any]]`. But in theory, the return value
    # of generators are currently impossible to type, because the return value
    # of a `yield from _` is # piggybacked using a `raise StopIteration(value)`
    # from `__next__`. So that also makes `__await__` theoretically
    # impossible to type. In practice, typecheckers work around that, by
    # accepting the lie called `collections.abc.Generator`...
    @overload
    def __await__(self: 'CanAwait[None]') -> CanNext[_MaybeFuture]: ...
    @overload
    def __await__(
        self: 'CanAwait[_V_co]',
    ) -> Generator[_MaybeFuture, None, _V_co]: ...


# 3.4.3. Asynchronous Iterators
# https://docs.python.org/3/reference/datamodel.html#asynchronous-iterators

@runtime_checkable
class CanANext(Protocol[_V_co]):
    def __anext__(self, /) -> _V_co: ...


@runtime_checkable
class CanAIter(Protocol[_V_anext_co]):
    def __aiter__(self, /) -> _V_anext_co: ...


@runtime_checkable
class CanAIterSelf(
    CanANext[_V_co],
    CanAIter[CanANext[_V_co]],
    Protocol[_V_co],
):
    """A less inflexible variant of `collections.abc.AsyncIterator[T]`."""
    @override
    def __aiter__(self, /) -> Self: ...


# 3.4.4. Asynchronous Context Managers
# https://docs.python.org/3/reference/datamodel.html#asynchronous-context-managers

@runtime_checkable
class CanAEnter(Protocol[_X_co]):
    def __aenter__(self, /) -> CanAwait[_X_co]: ...


@runtime_checkable
class CanAExit(Protocol[_Y_co]):
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
        exc_type: type[_V_exc],
        exc_instance: _V_exc,
        exc_traceback: TracebackType,
        /,
    ) -> CanAwait[_Y_co]: ...


@runtime_checkable
class CanAsyncWith(CanAEnter[_X_co], CanAExit[_Y_co], Protocol[_X_co, _Y_co]):
    ...


# `copy` stdlib
# https://docs.python.org/3.13/library/copy.html


@runtime_checkable
class CanCopy(Protocol[_T_co]):
    """Support for creating shallow copies through `copy.copy`."""
    def __copy__(self, /) -> _T_co: ...


@runtime_checkable
class CanDeepcopy(Protocol[_T_co]):
    """Support for creating deep copies through `copy.deepcopy`."""
    def __deepcopy__(self, memo: dict[int, Any], /) -> _T_co: ...


@runtime_checkable
class CanReplace(Protocol[_V_contra, _T_co]):
    """Support for `copy.replace` in Python 3.13+."""
    def __replace__(self, /, **changes: _V_contra) -> _T_co: ...


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
class CanReplaceSelf(CanReplace[_V_contra, 'CanReplaceSelf[Any]'], Protocol):
    """Variant of `CanReplace` that returns `Self`."""
    @override
    def __replace__(self, /, **changes: _V_contra) -> Self: ...


# `pickle` stdlib
# https://docs.python.org/3.13/library/pickle.html


_Y_str_tuple_co = TypeVar(
    '_Y_str_tuple_co',
    covariant=True,
    bound=str | tuple[Any, ...],
)


@runtime_checkable
class CanReduce(Protocol[_Y_str_tuple_co]):
    @override
    def __reduce__(self, /) -> _Y_str_tuple_co: ...


@runtime_checkable
class CanReduceEx(Protocol[_Y_str_tuple_co]):
    @override
    def __reduce_ex__(self, protocol: CanIndex, /) -> _Y_str_tuple_co: ...


@runtime_checkable
class CanGetstate(Protocol[_T_co]):
    def __getstate__(self, /) -> _T_co: ...


@runtime_checkable
class CanSetstate(Protocol[_T_contra]):
    def __setstate__(self, state: _T_contra, /) -> None: ...


@runtime_checkable
class CanGetnewargs(Protocol[Unpack[_Xs]]):
    def __new__(cls, *args: Unpack[_Xs]) -> Self: ...
    def __getnewargs__(self, /) -> tuple[Unpack[_Xs]]: ...


@runtime_checkable
class CanGetnewargsEx(Protocol[Unpack[_Xs], _V]):
    def __new__(cls, *args: Unpack[_Xs], **kwargs: _V) -> Self: ...
    def __getnewargs_ex__(self, /) -> tuple[
        tuple[Unpack[_Xs]],
        dict[str, _V],
    ]: ...


# ruff: noqa: TD002, TD003, FIX002
_K_sequence_contra = TypeVar(
    '_K_sequence_contra',
    contravariant=True,
    # TODO: use a protocol to narrow the slice values to ints
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
