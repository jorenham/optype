# ruff: noqa: PYI034
from collections.abc import Generator  # sadge :(
from types import TracebackType
from typing import (
    Any,
    Protocol,
    overload,
    override,
    runtime_checkable,
)


# Iterator types
# https://docs.python.org/3/library/stdtypes.html#iterator-types

@runtime_checkable
class CanNext[V](Protocol):
    def __next__(self) -> V: ...


@runtime_checkable
class CanIter[Vs: CanNext[Any]](Protocol):
    def __iter__(self) -> Vs: ...

# 3.3.1. Basic customization
# https://docs.python.org/3/reference/datamodel.html#basic-customization


@runtime_checkable
class CanRepr[Y: str](Protocol):
    @override
    def __repr__(self) -> Y: ...


@runtime_checkable
class CanStr[Y: str](Protocol):
    """By default, each `object` has a `__str__` method."""
    @override
    def __str__(self) -> Y: ...


@runtime_checkable
class CanBytes[Y: bytes](Protocol):
    def __bytes__(self) -> Y: ...


@runtime_checkable
class CanFormat[X: str, Y: str](Protocol):
    @override
    def __format__(self, __x: X) -> Y: ...   # pyright:ignore[reportIncompatibleMethodOverride]


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
    @override
    def __eq__(self, __x: X, /) -> Y: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanNe[X, Y](Protocol):
    @override
    def __ne__(self, __x: X) -> Y: ...  # pyright:ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class CanGt[X, Y](Protocol):
    def __gt__(self, __x: X) -> Y: ...


@runtime_checkable
class CanGe[X, Y](Protocol):
    def __ge__(self, __x: X) -> Y: ...


@runtime_checkable
class CanHash(Protocol):
    @override
    def __hash__(self) -> int: ...


@runtime_checkable
class CanBool(Protocol):
    def __bool__(self) -> bool: ...


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


# path-like
# TODO: CanFspath[P: str | bytes]


# standard library `copy`
# https://docs.python.org/3/library/copy.html
# TODO: CanCopy
# TODO: CanDeepCopy
# TODO: CanReplace (py313+)


# standard library `pickle`
# https://docs.python.org/3/library/pickle.html#pickling-class-instances
# TODO: CanGetnewargsEx
# TODO: CanGetnewargs
# TODO: CanGetstate
# TODO: CanSetstate
# TODO: CanReduce
# TODO: CanReduceEx


# 3rd-party library `numpy`
# https://numpy.org/devdocs/reference/arrays.classes.html
# https://numpy.org/devdocs/user/basics.subclassing.html
# TODO: __array__
# TODO: __array_ufunc__ (this one is pretty awesome)
# TODO: __array_function__
# TODO: __array_finalize__
# TODO (maybe): __array_prepare__
# TODO (maybe): __array_priority__
# TODO (maybe): __array_wrap__
# https://numpy.org/doc/stable/reference/arrays.interface.html
# TODO: __array_interface__
# TODO (maybe): __array_struct__


# Array API
# https://data-apis.org/array-api/latest/API_specification/array_object.html
# TODO: __array_namespace__
# TODO: __dlpack__
# TODO: __dlpack_device__


# Dataframe API
# https://data-apis.org/dataframe-api/draft/API_specification/index.html
# TODO: __dataframe_namespace__
# TODO: __column_namespace__
