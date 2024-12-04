# mypy: disable-error-code="override"
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol, TypeAlias


if sys.version_info >= (3, 13):
    from typing import (
        ParamSpec,
        Self,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        ParamSpec,
        Self,
        TypeVar,
        overload,
        override,
        runtime_checkable,
    )

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType

from ._utils import set_module


# return type that is usually `None`, but can be anything, as it is ignored at runtime
_Ignored: TypeAlias = object | None


_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)

_K_contra = TypeVar("_K_contra", contravariant=True)

_V_contra = TypeVar("_V_contra", contravariant=True)
_V_co = TypeVar("_V_co", covariant=True)
_VV_co = TypeVar("_VV_co", covariant=True, default=_V_co)

_BoolT_co = TypeVar("_BoolT_co", bound=bool, default=bool, covariant=True)
_IntT_contra = TypeVar("_IntT_contra", bound=int, contravariant=True, default=int)
_IntT_co = TypeVar("_IntT_co", bound=int, covariant=True, default=int)
_BytesT_co = TypeVar("_BytesT_co", bound=bytes, covariant=True, default=bytes)
_StrT_contra = TypeVar("_StrT_contra", bound=str, contravariant=True, default=str)
_StrT_co = TypeVar("_StrT_co", bound=str, covariant=True, default=str)

_AnyT_contra = TypeVar("_AnyT_contra", contravariant=True, default=object)
_AnyT_co = TypeVar("_AnyT_co", covariant=True, default=object)
_ObjectT_contra = TypeVar("_ObjectT_contra", contravariant=True, default=object)
# can be anything, but defaults to `bool`
_AnyBoolT_co = TypeVar("_AnyBoolT_co", covariant=True, default=bool)
_AnyIntT_contra = TypeVar("_AnyIntT_contra", contravariant=True, default=int)
_AnyIntT_co = TypeVar("_AnyIntT_co", covariant=True, default=int)
_AnyFloatT_co = TypeVar("_AnyFloatT_co", covariant=True, default=float)
_AnyNoneT_co = TypeVar("_AnyNoneT_co", covariant=True, default=None)

# Type conversion


@set_module("optype")
@runtime_checkable
class CanBool(Protocol[_BoolT_co]):
    def __bool__(self, /) -> _BoolT_co: ...


@set_module("optype")
@runtime_checkable
class CanInt(Protocol[_IntT_co]):
    def __int__(self, /) -> _IntT_co: ...


@set_module("optype")
@runtime_checkable
class CanFloat(Protocol):
    def __float__(self, /) -> float: ...


@set_module("optype")
@runtime_checkable
class CanComplex(Protocol):
    def __complex__(self, /) -> complex: ...


@set_module("optype")
@runtime_checkable
class CanBytes(Protocol[_BytesT_co]):
    """
    The `__bytes__: (CanBytes[Y]) -> Y` method is *co*variant on `+Y`.
    So if `__bytes__` returns an instance of a custom `bytes` subtype
    `Y <: bytes`, then `bytes()` will also return `Y` (i.e. no upcasting).
    """

    def __bytes__(self, /) -> _BytesT_co: ...


@set_module("optype")
@runtime_checkable
class CanStr(Protocol[_StrT_co]):
    """
    Each `object` has a *co*variant `__str__: (CanStr[Y=str]) -> Y` method on
    `+Y`. That means that if `__str__()` returns an instance of a custom `str`
    subtype `Y <: str`, then `str()` will also return `Y` (i.e. no upcasting).
    """

    @override
    def __str__(self, /) -> _StrT_co: ...


# Object representation


@set_module("optype")
@runtime_checkable
class CanHash(Protocol):
    @override
    def __hash__(self, /) -> int: ...


@set_module("optype")
@runtime_checkable
class CanIndex(Protocol[_IntT_co]):
    def __index__(self, /) -> _IntT_co: ...


@set_module("optype")
@runtime_checkable
class CanRepr(Protocol[_StrT_co]):
    """
    Each `object` has a *co*variant `__repr__: (CanRepr[Y=str]) -> Y` method.
    That means that if `__repr__` returns an instance of a custom `str`
    subtype `Y <: str`, then `repr()` will also return `Y` (i.e. no upcasting).
    """

    @override
    def __repr__(self, /) -> _StrT_co: ...


@set_module("optype")
@runtime_checkable
class CanFormat(Protocol[_StrT_contra, _StrT_co]):
    """
    Each `object` has a `__format__: (CanFormat[X, Y], X) -> Y` method, with
    `-X` *contra*variant, and `+Y` *co*variant. Both `X` and `Y` can be `str`
    or `str` subtypes. Note that `format()` *does not* upcast `Y` to `str`.
    """

    @override
    def __format__(self, fmt: _StrT_contra, /) -> _StrT_co: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]


# Iteration


@set_module("optype")
@runtime_checkable
class CanNext(Protocol[_V_co]):
    """
    Similar to `collections.abc.Iterator[V]`, but without the requirement to
    also have a `__iter__` method, which isn't needed in most cases (at least
    not in cpython).
    """

    def __next__(self, /) -> _V_co: ...


_CanNextT_co = TypeVar("_CanNextT_co", bound=CanNext[object], covariant=True)


@set_module("optype")
@runtime_checkable
class CanIter(Protocol[_CanNextT_co]):
    """Like `collections.abc.Iterable[V]`, but with a flexible return type."""

    def __iter__(self, /) -> _CanNextT_co: ...


@set_module("optype")
@runtime_checkable
class CanIterSelf(CanNext[_V_co], CanIter[CanNext[_V_co]], Protocol[_V_co]):
    """Like `collections.abc.Iterator[V]`, but without the `abc` nonsense."""

    @override
    def __iter__(self, /) -> Self: ...


# Async Iteration


@set_module("optype")
@runtime_checkable
class CanANext(Protocol[_V_co]):
    def __anext__(self, /) -> _V_co: ...


_CanANextT_co = TypeVar("_CanANextT_co", bound=CanANext[object], covariant=True)


@set_module("optype")
@runtime_checkable
class CanAIter(Protocol[_CanANextT_co]):
    def __aiter__(self, /) -> _CanANextT_co: ...


@set_module("optype")
@runtime_checkable
class CanAIterSelf(CanAIter["CanAIterSelf[_V_co]"], CanANext[_V_co], Protocol[_V_co]):
    """Like `collections.abc.AsyncIterator[T]`, but without the `abc` nonsense."""

    @override
    def __aiter__(self, /) -> Self: ...


# Rich comparison operands


@set_module("optype")
@runtime_checkable
class CanEq(Protocol[_ObjectT_contra, _AnyBoolT_co]):  # noqa: PLW1641
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
    def __eq__(self, rhs: _ObjectT_contra, /) -> _AnyBoolT_co: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@set_module("optype")
@runtime_checkable
class CanNe(Protocol[_ObjectT_contra, _AnyBoolT_co]):
    """
    Just like `__eq__`, the `__ne__` method is incorrectly annotated in
    typeshed. Refer to `CanEq` for why this is.
    """

    @override
    def __ne__(self, rhs: _ObjectT_contra, /) -> _AnyBoolT_co: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@set_module("optype")
@runtime_checkable
class CanLt(Protocol[_AnyT_contra, _AnyBoolT_co]):
    def __lt__(self, rhs: _AnyT_contra, /) -> _AnyBoolT_co: ...


@set_module("optype")
@runtime_checkable
class CanLe(Protocol[_AnyT_contra, _AnyBoolT_co]):
    def __le__(self, rhs: _AnyT_contra, /) -> _AnyBoolT_co: ...


@set_module("optype")
@runtime_checkable
class CanGt(Protocol[_AnyT_contra, _AnyBoolT_co]):
    def __gt__(self, rhs: _AnyT_contra, /) -> _AnyBoolT_co: ...


@set_module("optype")
@runtime_checkable
class CanGe(Protocol[_AnyT_contra, _AnyBoolT_co]):
    def __ge__(self, rhs: _AnyT_contra, /) -> _AnyBoolT_co: ...


# Callables

# this should (but can't) be contravariant
_P = ParamSpec("_P", default=...)


@set_module("optype")
@runtime_checkable
class CanCall(Protocol[_P, _AnyT_co]):
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _AnyT_co: ...  # type: ignore[no-any-explicit]


# Dynamic attribute access


@set_module("optype")
@runtime_checkable
class CanGetattr(Protocol[_StrT_contra, _AnyT_co]):
    def __getattr__(self, name: _StrT_contra, /) -> _AnyT_co: ...  # type: ignore[misc]


@set_module("optype")
@runtime_checkable
class CanGetattribute(Protocol[_StrT_contra, _AnyT_co]):
    """Note that `isinstance(x, CanGetattribute)` is always `True`."""

    @override
    def __getattribute__(self, name: _StrT_contra, /) -> _AnyT_co: ...  # type: ignore[misc]  # pyright: ignore[reportIncompatibleMethodOverride]


@set_module("optype")
@runtime_checkable
class CanSetattr(Protocol[_StrT_contra, _AnyT_contra]):
    """Note that `isinstance(x, CanSetattr)` is always true."""

    @override
    def __setattr__(self, name: _StrT_contra, value: _AnyT_contra, /) -> _Ignored: ...  # type: ignore[misc]  # pyright: ignore[reportIncompatibleMethodOverride]


@set_module("optype")
@runtime_checkable
class CanDelattr(Protocol[_StrT_contra]):
    @override
    def __delattr__(self, name: _StrT_contra, /) -> _Ignored: ...  # pyright: ignore[reportIncompatibleMethodOverride]


_AnyStrIterT_co = TypeVar(
    "_AnyStrIterT_co",
    bound=CanIter[CanNext[object]],
    covariant=True,
    default=CanIter[CanIterSelf[str]],
)


@set_module("optype")
@runtime_checkable
class CanDir(Protocol[_AnyStrIterT_co]):
    @override
    def __dir__(self, /) -> _AnyStrIterT_co: ...  # pyright: ignore[reportIncompatibleMethodOverride]


# Descriptors


@set_module("optype")
@runtime_checkable
class CanGet(Protocol[_T_contra, _V_co, _VV_co]):
    @overload
    def __get__(self, obj: _T_contra, cls: type | None = ..., /) -> _V_co: ...
    @overload
    def __get__(self, obj: None, cls: type[_T_contra], /) -> _VV_co: ...


@set_module("optype")
@runtime_checkable
class CanSet(Protocol[_T_contra, _V_contra]):
    def __set__(self, owner: _T_contra, value: _V_contra, /) -> _Ignored: ...


@set_module("optype")
@runtime_checkable
class CanDelete(Protocol[_T_contra]):
    def __delete__(self, owner: _T_contra, /) -> _Ignored: ...


@set_module("optype")
@runtime_checkable
class CanSetName(Protocol[_T_contra, _StrT_contra]):
    def __set_name__(self, cls: type[_T_contra], name: _StrT_contra, /) -> _Ignored: ...


# Collection type operands.


@set_module("optype")
@runtime_checkable
class CanLen(Protocol[_IntT_co]):
    def __len__(self, /) -> _IntT_co: ...


@set_module("optype")
@runtime_checkable
class CanLengthHint(Protocol[_IntT_co]):
    def __length_hint__(self, /) -> _IntT_co: ...


@set_module("optype")
@runtime_checkable
class CanGetitem(Protocol[_K_contra, _V_co]):
    def __getitem__(self, key: _K_contra, /) -> _V_co: ...


@set_module("optype")
@runtime_checkable
class CanSetitem(Protocol[_K_contra, _V_contra]):
    def __setitem__(self, key: _K_contra, value: _V_contra, /) -> _Ignored: ...


@set_module("optype")
@runtime_checkable
class CanDelitem(Protocol[_K_contra]):
    def __delitem__(self, key: _K_contra, /) -> None: ...


@set_module("optype")
@runtime_checkable
class CanReversed(Protocol[_T_co]):
    # `builtin.reversed` can return anything, but in practice it's always
    # something that can be iterated over (e.g. iterable or sequence-like)
    def __reversed__(self, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanContains(Protocol[_ObjectT_contra, _BoolT_co]):
    # usually the key is required to also be a hashable object, but this
    # isn't strictly required
    def __contains__(self, key: _ObjectT_contra, /) -> _BoolT_co: ...


@set_module("optype")
@runtime_checkable
class CanMissing(Protocol[_K_contra, _V_co]):
    def __missing__(self, key: _K_contra, /) -> _V_co: ...


@set_module("optype")
@runtime_checkable
class CanGetMissing(
    CanGetitem[_K_contra, _V_co],
    CanMissing[_K_contra, _V_co],
    Protocol[_K_contra, _V_co, _VV_co],
): ...


_IndexT_contra = TypeVar("_IndexT_contra", bound=CanIndex | slice, contravariant=True)


@set_module("optype")
@runtime_checkable
class CanSequence(
    CanGetitem[_IndexT_contra, _V_co],
    CanLen[_IntT_co],
    Protocol[_IndexT_contra, _V_co, _IntT_co],
):
    """
    A sequence is an object with a __len__ method and a
    __getitem__ method that takes int(-like) argument as key (the index).
    Additionally, it is expected to be 0-indexed (the first element is at
    index 0) and "dense" (i.e. the indices are consecutive integers, and are
    obtainable with e.g. `range(len(_))`).
    """


# Arithmetic operands


@set_module("optype")
@runtime_checkable
class CanAdd(Protocol[_T_contra, _T_co]):
    def __add__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanSub(Protocol[_T_contra, _T_co]):
    def __sub__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanMul(Protocol[_T_contra, _T_co]):
    def __mul__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanMatmul(Protocol[_T_contra, _T_co]):
    def __matmul__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanTruediv(Protocol[_T_contra, _T_co]):
    def __truediv__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanFloordiv(Protocol[_T_contra, _T_co]):
    def __floordiv__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanMod(Protocol[_T_contra, _T_co]):
    def __mod__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanDivmod(Protocol[_T_contra, _T_co]):
    def __divmod__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanPow2(Protocol[_T_contra, _T_co]):
    @overload
    def __pow__(self, exp: _T_contra, /) -> _T_co: ...
    @overload
    def __pow__(self, exp: _T_contra, mod: None = ..., /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanPow3(Protocol[_T_contra, _V_contra, _AnyIntT_co]):
    def __pow__(self, exp: _T_contra, mod: _V_contra, /) -> _AnyIntT_co: ...


@set_module("optype")
@runtime_checkable
class CanPow(
    CanPow2[_T_contra, _T_co],
    CanPow3[_T_contra, _V_contra, _AnyIntT_co],
    Protocol[_T_contra, _V_contra, _T_co, _AnyIntT_co],
):
    @overload
    @override
    def __pow__(self, exp: _T_contra, /) -> _T_co: ...
    @overload
    @override
    def __pow__(self, exp: _T_contra, mod: None = ..., /) -> _T_co: ...
    @overload
    @override
    def __pow__(self, exp: _T_contra, mod: _V_contra, /) -> _AnyIntT_co: ...


@set_module("optype")
@runtime_checkable
class CanLshift(Protocol[_T_contra, _T_co]):
    def __lshift__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRshift(Protocol[_T_contra, _T_co]):
    def __rshift__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanAnd(Protocol[_T_contra, _T_co]):
    def __and__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanXor(Protocol[_T_contra, _T_co]):
    def __xor__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanOr(Protocol[_T_contra, _T_co]):
    def __or__(self, rhs: _T_contra, /) -> _T_co: ...


# Reflected arithmetic operands


@set_module("optype")
@runtime_checkable
class CanRAdd(Protocol[_T_contra, _T_co]):
    def __radd__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRSub(Protocol[_T_contra, _T_co]):
    def __rsub__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRMul(Protocol[_T_contra, _T_co]):
    def __rmul__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRMatmul(Protocol[_T_contra, _T_co]):
    def __rmatmul__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRTruediv(Protocol[_T_contra, _T_co]):
    def __rtruediv__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRFloordiv(Protocol[_T_contra, _T_co]):
    def __rfloordiv__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRMod(Protocol[_T_contra, _T_co]):
    def __rmod__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRDivmod(Protocol[_T_contra, _T_co]):
    # can return anything, but is almost always a 2-tuple
    def __rdivmod__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRPow(Protocol[_T_contra, _T_co]):
    def __rpow__(self, x: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRLshift(Protocol[_T_contra, _T_co]):
    def __rlshift__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRRshift(Protocol[_T_contra, _T_co]):
    def __rrshift__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRAnd(Protocol[_T_contra, _T_co]):
    def __rand__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanRXor(Protocol[_T_contra, _T_co]):
    def __rxor__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanROr(Protocol[_T_contra, _T_co]):
    def __ror__(self, rhs: _T_contra, /) -> _T_co: ...


# Augmented arithmetic operands


@set_module("optype")
@runtime_checkable
class CanIAdd(Protocol[_T_contra, _T_co]):
    def __iadd__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIAddSelf(CanIAdd[_T_contra, "CanIAddSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __iadd__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanISub(Protocol[_T_contra, _T_co]):
    def __isub__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanISubSelf(CanISub[_T_contra, "CanISubSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __isub__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIMul(Protocol[_T_contra, _T_co]):
    def __imul__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIMulSelf(CanIMul[_T_contra, "CanIMulSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __imul__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIMatmul(Protocol[_T_contra, _T_co]):
    def __imatmul__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIMatmulSelf(
    CanIMatmul[_T_contra, "CanIMatmulSelf[_T_contra]"],
    Protocol[_T_contra],
):
    @override
    def __imatmul__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanITruediv(Protocol[_T_contra, _T_co]):
    def __itruediv__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanITruedivSelf(
    CanITruediv[_T_contra, "CanITruedivSelf[_T_contra]"],
    Protocol[_T_contra],
):
    @override
    def __itruediv__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIFloordiv(Protocol[_T_contra, _T_co]):
    def __ifloordiv__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIFloordivSelf(
    CanIFloordiv[_T_contra, "CanIFloordivSelf[_T_contra]"],
    Protocol[_T_contra],
):
    @override
    def __ifloordiv__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIMod(Protocol[_T_contra, _T_co]):
    def __imod__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIModSelf(CanIMod[_T_contra, "CanIModSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __imod__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIPow(Protocol[_T_contra, _T_co]):
    # no augmented pow/3 exists
    def __ipow__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIPowSelf(CanIPow[_T_contra, "CanIPowSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __ipow__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanILshift(Protocol[_T_contra, _T_co]):
    def __ilshift__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanILshiftSelf(
    CanILshift[_T_contra, "CanILshiftSelf[_T_contra]"],
    Protocol[_T_contra],
):
    @override
    def __ilshift__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIRshift(Protocol[_T_contra, _T_co]):
    def __irshift__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIRshiftSelf(
    CanIRshift[_T_contra, "CanIRshiftSelf[_T_contra]"],
    Protocol[_T_contra],
):
    @override
    def __irshift__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIAnd(Protocol[_T_contra, _T_co]):
    def __iand__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIAndSelf(CanIAnd[_T_contra, "CanIAndSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __iand__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIXor(Protocol[_T_contra, _T_co]):
    def __ixor__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIXorSelf(CanIXor[_T_contra, "CanIXorSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __ixor__(self, rhs: _T_contra, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanIOr(Protocol[_T_contra, _T_co]):
    def __ior__(self, rhs: _T_contra, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanIOrSelf(CanIOr[_T_contra, "CanIOrSelf[_T_contra]"], Protocol[_T_contra]):
    @override
    def __ior__(self, rhs: _T_contra, /) -> Self: ...


# Unary arithmetic ops


@set_module("optype")
@runtime_checkable
class CanNeg(Protocol[_T_co]):
    def __neg__(self, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanNegSelf(CanNeg["CanNegSelf"], Protocol):
    @override
    def __neg__(self, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanPos(Protocol[_T_co]):
    def __pos__(self, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanPosSelf(CanPos["CanPosSelf"], Protocol):
    @override
    def __pos__(self, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanAbs(Protocol[_T_co]):
    def __abs__(self, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanAbsSelf(CanAbs["CanAbsSelf"], Protocol):
    @override
    def __abs__(self, /) -> Self: ...


@set_module("optype")
@runtime_checkable
class CanInvert(Protocol[_T_co]):
    def __invert__(self, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanInvertSelf(CanInvert["CanInvertSelf"], Protocol):
    @override
    def __invert__(self, /) -> Self: ...


# Rounding


@set_module("optype")
@runtime_checkable
class CanRound1(Protocol[_AnyIntT_co]):
    @overload
    def __round__(self, /) -> _AnyIntT_co: ...
    @overload
    def __round__(self, /, ndigits: None = ...) -> _AnyIntT_co: ...


@set_module("optype")
@runtime_checkable
class CanRound2(Protocol[_AnyIntT_contra, _AnyFloatT_co]):
    def __round__(self, /, ndigits: _AnyIntT_contra) -> _AnyFloatT_co: ...


@set_module("optype")
@runtime_checkable
class CanRound(
    CanRound1[_AnyIntT_co],
    CanRound2[_AnyIntT_contra, _AnyFloatT_co],
    Protocol[_AnyIntT_contra, _AnyIntT_co, _AnyFloatT_co],
):
    @overload
    @override
    def __round__(self, /) -> _AnyIntT_co: ...
    @overload
    @override
    def __round__(self, /, ndigits: None = ...) -> _AnyIntT_co: ...
    @overload
    @override
    def __round__(self, /, ndigits: _AnyIntT_contra) -> _AnyFloatT_co: ...


@set_module("optype")
@runtime_checkable
class CanTrunc(Protocol[_AnyIntT_co]):
    def __trunc__(self, /) -> _AnyIntT_co: ...


@set_module("optype")
@runtime_checkable
class CanFloor(Protocol[_AnyIntT_co]):
    def __floor__(self, /) -> _AnyIntT_co: ...


@set_module("optype")
@runtime_checkable
class CanCeil(Protocol[_AnyIntT_co]):
    def __ceil__(self, /) -> _AnyIntT_co: ...


# Context managers


@set_module("optype")
@runtime_checkable
class CanEnter(Protocol[_T_co]):
    def __enter__(self, /) -> _T_co: ...


@set_module("optype")
@runtime_checkable
class CanEnterSelf(CanEnter["CanEnterSelf"], Protocol):
    @override
    def __enter__(self, /) -> Self: ...  # pyright: ignore[reportMissingSuperCall]


_ExcT = TypeVar("_ExcT", bound=BaseException)


@set_module("optype")
@runtime_checkable
class CanExit(Protocol[_AnyNoneT_co]):
    @overload
    def __exit__(self, exc_type: None, exc: None, tb: None, /) -> None: ...
    @overload
    def __exit__(
        self,
        exc_type: type[_ExcT],
        exc: _ExcT,
        tb: TracebackType,
        /,
    ) -> _AnyNoneT_co: ...


@set_module("optype")
@runtime_checkable
class CanWith(
    CanEnter[_T_co],
    CanExit[_AnyNoneT_co],
    Protocol[_T_co, _AnyNoneT_co],
): ...


@set_module("optype")
@runtime_checkable
class CanWithSelf(CanEnterSelf, CanExit[_AnyNoneT_co], Protocol[_AnyNoneT_co]): ...


# Async context managers


@set_module("optype")
@runtime_checkable
class CanAEnter(Protocol[_T_co]):
    def __aenter__(self, /) -> CanAwait[_T_co]: ...


@set_module("optype")
@runtime_checkable
class CanAEnterSelf(CanAEnter["CanAEnterSelf"], Protocol):
    @override
    def __aenter__(self, /) -> CanAwait[Self]: ...


@set_module("optype")
@runtime_checkable
class CanAExit(Protocol[_AnyNoneT_co]):
    @overload
    def __aexit__(self, exc_type: None, exc: None, tb: None, /) -> CanAwait[None]: ...
    @overload
    def __aexit__(
        self,
        exc_type: type[_ExcT],
        exc_: _ExcT,
        tb: TracebackType,
        /,
    ) -> CanAwait[_AnyNoneT_co]: ...


@set_module("optype")
@runtime_checkable
class CanAsyncWith(
    CanAEnter[_T_co],
    CanAExit[_AnyNoneT_co],
    Protocol[_T_co, _AnyNoneT_co],
): ...


@set_module("optype")
@runtime_checkable
class CanAsyncWithSelf(
    CanAEnterSelf,
    CanAExit[_AnyNoneT_co],
    Protocol[_AnyNoneT_co],
): ...


# Buffer protocol


@set_module("optype")
@runtime_checkable
class CanBuffer(Protocol[_IntT_contra]):
    def __buffer__(self, buffer: _IntT_contra, /) -> memoryview: ...


@set_module("optype")
@runtime_checkable
class CanReleaseBuffer(Protocol):
    def __release_buffer__(self, buffer: memoryview, /) -> None: ...


# Awaitables

# This should be `asyncio.Future[typing.Any] | None`. But that would make this
# incompatible with `collections.abc.Awaitable` -- it (annoyingly) uses `Any`:
# https://github.com/python/typeshed/blob/587ad6/stdlib/asyncio/futures.pyi#L51
_FutureOrNone: TypeAlias = object
_AsyncGen: TypeAlias = "Generator[_FutureOrNone, None, _T]"


@set_module("optype")
@runtime_checkable
class CanAwait(Protocol[_T_co]):
    # Technically speaking, this can return any
    # `CanNext[None | asyncio.Future[object]]`. But in theory, the return value
    # of generators are currently impossible to type, because the return value
    # of a `yield from _` is # piggybacked using a `raise StopIteration(value)`
    # from `__next__`. So that also makes `__await__` theoretically
    # impossible to type. In practice, typecheckers work around that, by
    # accepting the lie called `collections.abc.Generator`...
    @overload
    def __await__(self: CanAwait[_T_co], /) -> _AsyncGen[_T_co]: ...
    @overload
    def __await__(self: CanAwait[None], /) -> CanNext[_FutureOrNone]: ...
