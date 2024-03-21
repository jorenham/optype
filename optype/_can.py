# ruff: noqa: PYI034
from collections.abc import Generator  # sadge :(
from types import TracebackType
from typing import (
    Any,
    Protocol,
    Self,
    overload,
    override,
    runtime_checkable,
)


# Iterator types
# https://docs.python.org/3/library/stdtypes.html#iterator-types


@runtime_checkable
class CanNext[V](Protocol):
    """
    Similar to `collections.abc.Iterator`, but without the (often redundant)
    requirement to also have a `__iter__` method.
    """
    def __next__(self) -> V: ...


@runtime_checkable
class CanIter[Vs: CanNext[Any]](Protocol):
    """Similar to `collections.abc.Iterable`, but more flexible."""
    def __iter__(self) -> Vs: ...


@runtime_checkable
class CanIterSelf[V](CanNext[V], CanIter[CanNext[V]], Protocol):
    """
    Equivalent to `collections.abc.Iterator[T]`, minus the `abc` nonsense.
    """
    @override
    def __iter__(self) -> Self: ...


# 3.3.1. Basic customization
# https://docs.python.org/3/reference/datamodel.html#basic-customization


@runtime_checkable
class CanRepr[Y: str](Protocol):
    """
    Each `object` has a *co*variant `__repr__: (CanRepr[Y]) -> Y` method.
    That means that if `__repr__` returns an instance of a custom `str`
    subtype `Y <: str`, then `repr()` will also return `Y` (i.e. no upcasting).
    """
    @override
    def __repr__(self) -> Y: ...


@runtime_checkable
class CanStr[Y: str](Protocol):
    """
    Each `object` has a *co*variant `__str__: (CanStr[Y]) -> Y` method on `+Y`.
    That means that if `__str__()` returns an instance of a custom `str`
    subtype `Y <: str`, then `str()` will also return `Y` (i.e. no upcasting).
    """
    @override
    def __str__(self) -> Y: ...


@runtime_checkable
class CanBytes[Y: bytes](Protocol):
    """
    The `__bytes__: (CanBytes[Y]) -> Y` method is *co*variant on `+Y`.
    So if `__bytes__` returns an instance of a custom `bytes` subtype
    `Y <: bytes`, then `bytes()` will also return `Y` (i.e. no upcasting).
    """
    def __bytes__(self) -> Y: ...


@runtime_checkable
class CanFormat[X: str, Y: str](Protocol):
    """
    Each `object` has a `__format__: (CanFormat[X, Y], X) -> Y` method, with
    `-X` *contra*variant, and `+Y` *co*variant. Both `X` and `Y` can be `str`
    or `str` subtypes. Note that `format()` *does not* upcast `Y` to `str`.
    """
    @override
    def __format__(self, __x: X) -> Y: ...   # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanBool(Protocol):
    def __bool__(self) -> bool: ...


@runtime_checkable
class CanHash(Protocol):
    @override
    def __hash__(self) -> int: ...


# 3.3.1. Basic customization - Rich comparison method
# https://docs.python.org/3/reference/datamodel.html#object.__lt__


@runtime_checkable
class CanLt[X, Y](Protocol):
    def __lt__(self, __x: X) -> Y: ...


@runtime_checkable
class CanLe[X, Y](Protocol):
    def __le__(self, __x: X) -> Y: ...


@runtime_checkable
class CanEq[X, Y](Protocol):
    """
    Unfortunately, `typeshed` incorrectly annotates `object.__eq__` as
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
    def __eq__(self, __x: X, /) -> Y: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanNe[X, Y](Protocol):
    """
    Just like `__eq__`, The `__ne__` method is incorrectly annotated in
    `typeshed`. See `CanEq` for why this is, and how `optype` fixes this.
    """
    @override
    def __ne__(self, __x: X) -> Y: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanGt[X, Y](Protocol):
    def __gt__(self, __x: X) -> Y: ...


@runtime_checkable
class CanGe[X, Y](Protocol):
    def __ge__(self, __x: X) -> Y: ...


# 3.3.2. Customizing attribute access
# https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access

@runtime_checkable
class CanGetattr[K: str, V](Protocol):
    def __getattr__(self, __k: K) -> V: ...


@runtime_checkable
class CanGetattribute[K: str, V](Protocol):
    """Note that `isinstance(x, CanGetattribute)` is always true."""
    @override
    def __getattribute__(self, __k: K) -> V: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanSetattr[K: str, V](Protocol):
    """Note that `isinstance(x, CanSetattr)` is always true."""
    @override
    def __setattr__(self, __k: K, __v: V) -> Any: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanDelattr[K: str](Protocol):
    @override
    def __delattr__(self, __k: K) -> Any: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanDir[Vs: CanIter[Any]](Protocol):
    @override
    def __dir__(self) -> Vs: ...


# 3.3.2.2. Implementing Descriptors
# https://docs.python.org/3/reference/datamodel.html#implementing-descriptors

@runtime_checkable
class CanGet[T, U, V](Protocol):
    @overload
    def __get__(self, __obj: None, __cls: type[T]) -> U: ...
    @overload
    def __get__(self, __obj: T, __cls: type[T] | None = ...) -> V: ...


@runtime_checkable
class CanSet[T: object, V](Protocol):
    def __set__(self, __obj: T, __v: V) -> Any: ...


@runtime_checkable
class CanDelete[T: object](Protocol):
    def __delete__(self, __obj: T) -> Any: ...


# 3.3.3. Customizing class creation
# https://docs.python.org/3/reference/datamodel.html#customizing-class-creation

@runtime_checkable
class CanSetName[T](Protocol):
    def __set_name__(self, __cls: type[T], __name: str) -> Any: ...


# 3.3.6. Emulating callable objects
# https://docs.python.org/3/reference/datamodel.html#emulating-callable-objects

@runtime_checkable
class CanCall[**Xs, Y](Protocol):
    def __call__(self, *__xs: Xs.args, **__kxs: Xs.kwargs) -> Y: ...


# 3.3.7. Emulating container types
# https://docs.python.org/3/reference/datamodel.html#emulating-container-types

@runtime_checkable
class CanLen(Protocol):
    def __len__(self) -> int: ...


@runtime_checkable
class CanLengthHint(Protocol):
    def __length_hint__(self) -> int: ...


@runtime_checkable
class CanGetitem[K, V](Protocol):
    def __getitem__(self, __k: K) -> V: ...


@runtime_checkable
class CanSetitem[K, V](Protocol):
    def __setitem__(self, __k: K, __v: V) -> None: ...


@runtime_checkable
class CanDelitem[K](Protocol):
    def __delitem__(self, __k: K) -> None: ...


@runtime_checkable
class CanReversed[Y](Protocol):
    def __reversed__(self) -> Y: ...


@runtime_checkable
class CanContains[K](Protocol):
    def __contains__(self, __k: K) -> bool: ...


@runtime_checkable
class CanMissing[K, V](Protocol):
    def __missing__(self, __k: K) -> V: ...


@runtime_checkable
class CanGetMissing[K, V, M](CanGetitem[K, V], CanMissing[K, M], Protocol): ...


@runtime_checkable
class CanSequence[I: 'CanIndex', V](CanLen, CanGetitem[I, V], Protocol):
    """
    A sequence is an object with a __len__ method and a
    __getitem__ method that takes int(-like) argument as key (the index).
    Additionally, it is expected to be 0-indexed (the first element is at
    index 0) and "dense" (i.e. the indices are consecutive integers, and are
    obtainable with e.g. `range(len(_))`).
    """


# 3.3.8. Emulating numeric types
# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

@runtime_checkable
class CanAdd[X, Y](Protocol):
    def __add__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanSub[X, Y](Protocol):
    def __sub__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanMul[X, Y](Protocol):
    def __mul__(self, __x: X) -> Y: ...


@runtime_checkable
class CanMatmul[X, Y](Protocol):
    def __matmul__(self, __x: X) -> Y: ...


@runtime_checkable
class CanTruediv[X, Y](Protocol):
    def __truediv__(self, __x: X) -> Y: ...


@runtime_checkable
class CanFloordiv[X, Y](Protocol):
    def __floordiv__(self, __x: X) -> Y: ...


@runtime_checkable
class CanMod[X, Y](Protocol):
    def __mod__(self, __x: X) -> Y: ...


@runtime_checkable
class CanDivmod[X, Y](Protocol):
    def __divmod__(self, __x: X) -> Y: ...


@runtime_checkable
class CanPow2[X, Y2](Protocol):
    def __pow__(self, __x: X) -> Y2: ...


@runtime_checkable
class CanPow3[X, M, Y3](Protocol):
    def __pow__(self, __x: X, __m: M) -> Y3: ...


@runtime_checkable
class CanPow[X, M, Y2, Y3](CanPow2[X, Y2], CanPow3[X, M, Y3], Protocol):
    @overload
    def __pow__(self, __x: X) -> Y2: ...
    @overload
    def __pow__(self, __x: X, __m: M) -> Y3: ...


@runtime_checkable
class CanLshift[X, Y](Protocol):
    def __lshift__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanRshift[X, Y](Protocol):
    def __rshift__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanAnd[X, Y](Protocol):
    def __and__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanXor[X, Y](Protocol):
    def __xor__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanOr[X, Y](Protocol):
    def __or__(self, __x: X, /) -> Y: ...


# reflected

@runtime_checkable
class CanRAdd[X, Y](Protocol):
    def __radd__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanRSub[X, Y](Protocol):
    def __rsub__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanRMul[X, Y](Protocol):
    def __rmul__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRMatmul[X, Y](Protocol):
    def __rmatmul__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRTruediv[X, Y](Protocol):
    def __rtruediv__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRFloordiv[X, Y](Protocol):
    def __rfloordiv__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRMod[X, Y](Protocol):
    def __rmod__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRDivmod[X, Y](Protocol):
    def __rdivmod__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRPow[X, Y](Protocol):
    def __rpow__(self, __x: X) -> Y: ...


@runtime_checkable
class CanRLshift[X, Y](Protocol):
    def __rlshift__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanRRshift[X, Y](Protocol):
    def __rrshift__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanRAnd[X, Y](Protocol):
    def __rand__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanRXor[X, Y](Protocol):
    def __rxor__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanROr[X, Y](Protocol):
    def __ror__(self, __x: X, /) -> Y: ...


# augmented / in-place

@runtime_checkable
class CanIAdd[X, Y](Protocol):
    def __iadd__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanISub[X, Y](Protocol):
    def __isub__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanIMul[X, Y](Protocol):
    def __imul__(self, __x: X) -> Y: ...


@runtime_checkable
class CanIMatmul[X, Y](Protocol):
    def __imatmul__(self, __x: X) -> Y: ...


@runtime_checkable
class CanITruediv[X, Y](Protocol):
    def __itruediv__(self, __x: X) -> Y: ...


@runtime_checkable
class CanIFloordiv[X, Y](Protocol):
    def __ifloordiv__(self, __x: X) -> Y: ...


@runtime_checkable
class CanIMod[X, Y](Protocol):
    def __imod__(self, __x: X) -> Y: ...


@runtime_checkable
class CanIPow[X, Y](Protocol):
    # no augmented pow/3 exists
    def __ipow__(self, __x: X) -> Y: ...


@runtime_checkable
class CanILshift[X, Y](Protocol):
    def __ilshift__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanIRshift[X, Y](Protocol):
    def __irshift__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanIAnd[X, Y](Protocol):
    def __iand__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanIXor[X, Y](Protocol):
    def __ixor__(self, __x: X, /) -> Y: ...


@runtime_checkable
class CanIOr[X, Y](Protocol):
    def __ior__(self, __x: X, /) -> Y: ...


# unary arithmetic

@runtime_checkable
class CanNeg[Y](Protocol):
    def __neg__(self) -> Y: ...


@runtime_checkable
class CanPos[Y](Protocol):
    def __pos__(self) -> Y: ...


@runtime_checkable
class CanAbs[Y](Protocol):
    def __abs__(self) -> Y: ...


@runtime_checkable
class CanInvert[Y](Protocol):
    def __invert__(self) -> Y: ...


# numeric conversion

@runtime_checkable
class CanComplex(Protocol):
    def __complex__(self) -> complex: ...


@runtime_checkable
class CanFloat(Protocol):
    def __float__(self) -> float: ...


@runtime_checkable
class CanInt(Protocol):
    def __int__(self) -> int: ...


@runtime_checkable
class CanIndex(Protocol):
    def __index__(self) -> int: ...

# rounding


@runtime_checkable
class CanRound1[Y](Protocol):
    @overload
    def __round__(self) -> Y: ...
    @overload
    def __round__(self, __n: None = ...) -> Y: ...


@runtime_checkable
class CanRound2[N, Y](Protocol):
    def __round__(self, __n: N) -> Y: ...


@runtime_checkable
class CanRound[N, Y1, Y2](CanRound1[Y1], CanRound2[N, Y2], Protocol):
    @overload
    def __round__(self) -> Y1: ...
    @overload
    def __round__(self, __n: None = ...) -> Y1: ...
    @overload
    def __round__(self, __n: N) -> Y2: ...


@runtime_checkable
class CanTrunc[Y](Protocol):
    def __trunc__(self) -> Y: ...


@runtime_checkable
class CanFloor[Y](Protocol):
    def __floor__(self) -> Y: ...


@runtime_checkable
class CanCeil[Y](Protocol):
    def __ceil__(self) -> Y: ...


# 3.3.9. With Statement Context Managers
# https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers

@runtime_checkable
class CanEnter[V](Protocol):
    def __enter__(self) -> V: ...


@runtime_checkable
class CanExit[R](Protocol):
    @overload
    def __exit__(self, __tp: None, __ex: None, __tb: None) -> None: ...
    @overload
    def __exit__[E: BaseException](
        self,
        __tp: type[E],
        __ex: E,
        __tb: TracebackType,
    ) -> R: ...


@runtime_checkable
class CanWith[V, R](CanEnter[V], CanExit[R], Protocol): ...


# 3.3.11. Emulating buffer types
# https://docs.python.org/3/reference/datamodel.html#emulating-buffer-types

@runtime_checkable
class CanBuffer[B: int](Protocol):
    def __buffer__(self, __b: B) -> memoryview: ...


@runtime_checkable
class CanReleaseBuffer(Protocol):
    def __release_buffer__(self, __v: memoryview) -> None: ...


# 3.4.1. Awaitable Objects
# https://docs.python.org/3/reference/datamodel.html#awaitable-objects


# This should be `None | asyncio.Future[Any]`. But that would make this
# incompatible with `collections.abc.Awaitable`, because it (annoyingly)
# uses `Any`...
type _MaybeFuture = Any


@runtime_checkable
class CanAwait[V](Protocol):
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
    def __await__(self: 'CanAwait[V]') -> Generator[_MaybeFuture, None, V]: ...


# 3.4.3. Asynchronous Iterators
# https://docs.python.org/3/reference/datamodel.html#asynchronous-iterators

@runtime_checkable
class CanANext[V](Protocol):
    def __anext__(self) -> V: ...


@runtime_checkable
class CanAIter[Y: CanANext[Any]](Protocol):
    def __aiter__(self) -> Y: ...


@runtime_checkable
class CanAIterSelf[V](CanANext[V], CanAIter[CanANext[V]], Protocol):
    """A less inflexible variant of `collections.abc.AsyncIterator[T]`."""
    @override
    def __aiter__(self) -> 'CanAIterSelf[V]': ...  # `Self` doesn't work here?


# 3.4.4. Asynchronous Context Managers
# https://docs.python.org/3/reference/datamodel.html#asynchronous-context-managers

@runtime_checkable
class CanAEnter[V](Protocol):
    def __aenter__(self) -> CanAwait[V]: ...


@runtime_checkable
class CanAExit[R](Protocol):
    @overload
    def __aexit__(
        self,
        __tp: None,
        __ex: None,
        __tb: None,
    ) -> CanAwait[None]: ...
    @overload
    def __aexit__[E: BaseException](
        self,
        __tp: type[E],
        __ex: E,
        __tb: TracebackType,
    ) -> CanAwait[R]: ...


@runtime_checkable
class CanAsyncWith[V, R](CanAEnter[V], CanAExit[R], Protocol): ...


# `copy` stdlib
# https://docs.python.org/3.13/library/copy.html


@runtime_checkable
class CanCopy[T](Protocol):
    """Support for creating shallow copies through `copy.copy`."""
    def __copy__(self) -> T: ...


@runtime_checkable
class CanDeepcopy[T](Protocol):
    """Support for creating deep copies through `copy.deepcopy`."""
    def __deepcopy__(self, memo: dict[int, Any], /) -> T: ...


@runtime_checkable
class CanReplace[T, V](Protocol):
    """Support for `copy.replace` in Python 3.13+."""
    def __replace__(self, /, **changes: V) -> T: ...


@runtime_checkable
class CanCopySelf(CanCopy['CanCopySelf'], Protocol):
    """Variant of `CanCopy` that returns `Self` (as it should)."""
    @override
    def __copy__(self) -> Self: ...


class CanDeepcopySelf(CanDeepcopy['CanDeepcopySelf'], Protocol):
    """Variant of `CanDeepcopy` that returns `Self` (as it should)."""
    @override
    def __deepcopy__(self, memo: dict[int, Any], /) -> Self: ...


@runtime_checkable
class CanReplaceSelf[V](CanReplace['CanReplaceSelf[Any]', V], Protocol):
    """Variant of `CanReplace` that returns `Self`."""
    @override
    def __replace__(self, /, **changes: V) -> Self: ...


# `pickle` stdlib
# https://docs.python.org/3.13/library/pickle.html


@runtime_checkable
class CanReduce[R: str | tuple[Any, ...]](Protocol):
    @override
    def __reduce__(self) -> R: ...


@runtime_checkable
class CanReduceEx[R: str | tuple[Any, ...]](Protocol):
    @override
    def __reduce_ex__(self, protocol: CanIndex, /) -> R: ...


@runtime_checkable
class CanGetstate[S: object](Protocol):
    @override
    def __getstate__(self) -> S: ...


@runtime_checkable
class CanSetstate[S: object](Protocol):
    def __setstate__(self, state: S, /) -> None: ...


@runtime_checkable
class CanGetnewargs[*Args](Protocol):
    def __new__(cls, *__args: *Args) -> Self: ...
    def __getnewargs__(self) -> tuple[*Args]: ...


@runtime_checkable
class CanGetnewargsEx[*Args, Kw](Protocol):
    def __new__(cls, *__args: *Args, **__kwargs: Kw) -> Self: ...
    def __getnewargs_ex__(self) -> tuple[tuple[*Args], dict[str, Kw]]: ...
