from typing import Any, Protocol, Self, final, overload, override

import optype._can as _c


# iteration

class _CanIterNext[V](
    _c.CanIter['_CanIterNext[Any]'],
    _c.CanNext[V],
    Protocol,
):
    @override
    def __iter__(self) -> Self: ...


@final
class DoesNext(Protocol):
    # https://docs.python.org/3/library/functions.html#next
    @overload
    def __call__[V](self, __vs: _c.CanNext[V], /) -> V: ...
    @overload
    def __call__[V, V0](self, __vs: _c.CanNext[V], __v0: V0, /) -> V | V0: ...


@final
class DoesIter(Protocol):
    # https://docs.python.org/3/library/functions.html#iter
    @overload
    def __call__[Vs: _c.CanNext[Any]](self, __vs: _c.CanIter[Vs], /) -> Vs: ...
    @overload
    def __call__[V](
        self, __vs: _c.CanGetitem[_c.CanIndex, V], /,
    ) -> _CanIterNext[V]: ...
    @overload
    def __call__[V, S: object | None](
        self, __f: _c.CanCall[[], V | S], __s: S, /,
    ) -> _CanIterNext[V]: ...


# type conversion

@final
class DoesBool(Protocol):
    # https://docs.python.org/3/library/functions.html#bool
    def __call__(self, __o: _c.CanBool, /) -> bool: ...


@final
class DoesInt(Protocol):
    # https://docs.python.org/3/library/functions.html#int
    def __call__(self, __o: _c.CanInt, /) -> int: ...


@final
class DoesFloat(Protocol):
    # https://docs.python.org/3/library/functions.html#float
    def __call__(self, __o: _c.CanFloat, /) -> float: ...


@final
class DoesComplex(Protocol):
    # https://docs.python.org/3/library/functions.html#complex
    def __call__(self, __o: _c.CanComplex, /) -> complex: ...


@final
class DoesStr(Protocol):
    # https://docs.python.org/3/library/functions.html#func-str
    def __call__[Y: str](self, __o: _c.CanStr[Y], /) -> Y: ...


@final
class DoesBytes(Protocol):
    # https://docs.python.org/3/library/functions.html#func-bytes
    def __call__[Y: bytes](self, __o: _c.CanBytes[Y], /) -> Y: ...


# formatting

@final
class DoesRepr(Protocol):
    # https://docs.python.org/3/library/functions.html#repr
    def __call__[Y: str](self, __o: _c.CanRepr[Y], /) -> Y: ...


@final
class DoesFormat(Protocol):
    # https://docs.python.org/3/library/functions.html#format
    def __call__[X: str, Y: str](
        self, __o: _c.CanFormat[X, Y], __x: X = ..., /,
    ) -> Y: ...


# rich comparison

@final
class DoesLt(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanLt[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanGt[X, Y], /) -> Y: ...


@final
class DoesLe(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanLe[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanGe[X, Y], /) -> Y: ...


@final
class DoesEq(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanEq[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanEq[X, Y], /) -> Y: ...


@final
class DoesNe(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanNe[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanNe[X, Y], /) -> Y: ...


@final
class DoesGt(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanGt[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanLt[X, Y], /) -> Y: ...


@final
class DoesGe(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanGe[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanLe[X, Y], /) -> Y: ...


# attributes

type _CanGetAttr[K: str, V] = _c.CanGetattr[K, V] | _c.CanGetattribute[K, V]


@final
class DoesGetattr(Protocol):
    # https://docs.python.org/3/library/functions.html#getattr
    @overload
    def __call__[K: str, V](self, __o: _CanGetAttr[K, V], __k: K, /) -> V: ...
    @overload
    def __call__[K: str, V, V0](
        self, __o: _CanGetAttr[K, V], __k: K, __v0: V0, /,
    ) -> V | V0: ...


@final
class DoesSetattr(Protocol):
    # https://docs.python.org/3/library/functions.html#setattr
    def __call__[K: str, V](
        self, __o: _c.CanSetattr[K, V], __k: K, __v: V, /,
    ) -> None: ...


@final
class DoesDelattr(Protocol):
    # https://docs.python.org/3/library/functions.html#delattr
    def __call__[K: str](self, __o: _c.CanDelattr[K], __k: K, /) -> None: ...


@final
class DoesDir(Protocol):
    # https://docs.python.org/3/library/functions.html#dir
    @overload
    def __call__(self, /) -> list[str]: ...
    @overload
    def __call__[Vs: _c.CanIter[Any]](self, __o: _c.CanDir[Vs], /) -> Vs: ...


# callables

@final
class DoesCall(Protocol):
    def __call__[**Xs, Y](
        self,
        __f: _c.CanCall[Xs, Y],
        *__xs: Xs.args,
        **__kxs: Xs.kwargs,
    ) -> Y: ...


# containers and subscritable types

type _CanSubscript[K, V, M] = _c.CanGetitem[K, V] | _c.CanGetMissing[K, V, M]


@final
class DoesGetitem(Protocol):
    def __call__[K, V, M](
        self, __o: _CanSubscript[K, V, M], __k: K, /,
    ) -> V | M: ...


@final
class DoesSetitem(Protocol):
    def __call__[K, V](
        self, __o: _c.CanSetitem[K, V], __k: K, __v: V, /,
    ) -> None: ...


@final
class DoesDelitem(Protocol):
    def __call__[K](self, __o: _c.CanDelitem[K], __k: K, /) -> None: ...


@final
class DoesContains(Protocol):
    def __call__[K](self, __o: _c.CanContains[K], __k: K, /) -> bool: ...


# binary infix operators

@final
class DoesAdd(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanAdd[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRAdd[X, Y], /) -> Y: ...


@final
class DoesSub(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanSub[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRSub[X, Y], /) -> Y: ...


@final
class DoesMul(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanMul[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRMul[X, Y], /) -> Y: ...


@final
class DoesMatmul(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanMatmul[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRMatmul[X, Y], /) -> Y: ...


@final
class DoesTruediv(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanTruediv[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRTruediv[X, Y], /) -> Y: ...


@final
class DoesFloordiv(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanFloordiv[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRFloordiv[X, Y], /) -> Y: ...


@final
class DoesMod(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanMod[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRMod[X, Y], /) -> Y: ...


@final
class DoesDivmod(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanDivmod[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRDivmod[X, Y], /) -> Y: ...


@final
class DoesPow(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanPow2[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, M, Y](
        self, __o: _c.CanPow3[X, M, Y], __x: X, __mod: M, /,
    ) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRPow[X, Y], /) -> Y: ...


@final
class DoesLshift(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanLshift[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRLshift[X, Y], /) -> Y: ...


@final
class DoesRshift(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanRshift[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRRshift[X, Y], /) -> Y: ...


@final
class DoesAnd(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanAnd[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRAnd[X, Y], /) -> Y: ...


@final
class DoesXor(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanXor[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanRXor[X, Y], /) -> Y: ...


@final
class DoesOr(Protocol):
    @overload
    def __call__[X, Y](self, __o: _c.CanOr[X, Y], __x: X, /) -> Y: ...
    @overload
    def __call__[X, Y](self, __x: X, __o: _c.CanROr[X, Y], /) -> Y: ...


# binary reflected operators

@final
class DoesRAdd(Protocol):
    def __call__[X, Y](self, __o: _c.CanRAdd[X, Y], __x: X, /) -> Y: ...


@final
class DoesRSub(Protocol):
    def __call__[X, Y](self, __o: _c.CanRSub[X, Y], __x: X, /) -> Y: ...


@final
class DoesRMul(Protocol):
    def __call__[X, Y](self, __o: _c.CanRMul[X, Y], __x: X, /) -> Y: ...


@final
class DoesRMatmul(Protocol):
    def __call__[X, Y](self, __o: _c.CanRMatmul[X, Y], __x: X, /) -> Y: ...


@final
class DoesRTruediv(Protocol):
    def __call__[X, Y](self, __o: _c.CanRTruediv[X, Y], __x: X, /) -> Y: ...


@final
class DoesRFloordiv(Protocol):
    def __call__[X, Y](self, __o: _c.CanRFloordiv[X, Y], __x: X, /) -> Y: ...


@final
class DoesRMod(Protocol):
    def __call__[X, Y](self, __o: _c.CanRMod[X, Y], __x: X, /) -> Y: ...


@final
class DoesRDivmod(Protocol):
    def __call__[X, Y](self, __o: _c.CanRDivmod[X, Y], __x: X, /) -> Y: ...


@final
class DoesRPow(Protocol):
    def __call__[X, Y](self, __o: _c.CanRPow[X, Y], __x: X, /) -> Y: ...


@final
class DoesRLshift(Protocol):
    def __call__[X, Y](self, __o: _c.CanRLshift[X, Y], __x: X, /) -> Y: ...


@final
class DoesRRshift(Protocol):
    def __call__[X, Y](self, __o: _c.CanRRshift[X, Y], __x: X, /) -> Y: ...


@final
class DoesRAnd(Protocol):
    def __call__[X, Y](self, __o: _c.CanRAnd[X, Y], __x: X, /) -> Y: ...


@final
class DoesRXor(Protocol):
    def __call__[X, Y](self, __o: _c.CanRXor[X, Y], __x: X, /) -> Y: ...


@final
class DoesROr(Protocol):
    def __call__[X, Y](self, __o: _c.CanROr[X, Y], __x: X, /) -> Y: ...


# augmented / in-place operators

@final
class DoesIAdd(Protocol):
    def __call__[X, Y](self, __o: _c.CanIAdd[X, Y], __x: X, /) -> Y: ...


@final
class DoesISub(Protocol):
    def __call__[X, Y](self, __o: _c.CanISub[X, Y], __x: X, /) -> Y: ...


@final
class DoesIMul(Protocol):
    def __call__[X, Y](self, __o: _c.CanIMul[X, Y], __x: X, /) -> Y: ...


@final
class DoesIMatmul(Protocol):
    def __call__[X, Y](self, __o: _c.CanIMatmul[X, Y], __x: X, /) -> Y: ...


@final
class DoesITruediv(Protocol):
    def __call__[X, Y](self, __o: _c.CanITruediv[X, Y], __x: X, /) -> Y: ...


@final
class DoesIFloordiv(Protocol):
    def __call__[X, Y](self, __o: _c.CanIFloordiv[X, Y], __x: X, /) -> Y: ...


@final
class DoesIMod(Protocol):
    def __call__[X, Y](self, __o: _c.CanMod[X, Y], __x: X, /) -> Y: ...


@final
class DoesIPow(Protocol):
    def __call__[X, Y](self, __o: _c.CanIPow[X, Y], __x: X, /) -> Y: ...


@final
class DoesILshift(Protocol):
    def __call__[X, Y](self, __o: _c.CanILshift[X, Y], __x: X, /) -> Y: ...


@final
class DoesIRshift(Protocol):
    def __call__[X, Y](self, __o: _c.CanIRshift[X, Y], __x: X, /) -> Y: ...


@final
class DoesIAnd(Protocol):
    def __call__[X, Y](self, __o: _c.CanIAnd[X, Y], __x: X, /) -> Y: ...


@final
class DoesIXor(Protocol):
    def __call__[X, Y](self, __o: _c.CanIXor[X, Y], __x: X, /) -> Y: ...


@final
class DoesIOr(Protocol):
    def __call__[X, Y](self, __o: _c.CanIOr[X, Y], __x: X, /) -> Y: ...


# unary arithmetic

@final
class DoesNeg(Protocol):
    def __call__[Y](self, __o: _c.CanNeg[Y], /) -> Y: ...


@final
class DoesPos(Protocol):
    def __call__[Y](self, __o: _c.CanPos[Y], /) -> Y: ...


@final
class DoesAbs(Protocol):
    def __call__[Y](self, __o: _c.CanAbs[Y], /) -> Y: ...


@final
class DoesInvert(Protocol):
    def __call__[Y](self, __o: _c.CanInvert[Y], /) -> Y: ...


# fingerprinting

@final
class DoesHash(Protocol):
    def __call__[Y](self, __o: _c.CanHash, /) -> int: ...


@final
class DoesIndex(Protocol):
    def __call__[Y](self, __o: _c.CanIndex, /) -> int: ...


# rounding

@final
class DoesRound(Protocol):
    @overload
    def __call__[Y](self, __o: _c.CanRound1[Y], /) -> Y: ...
    @overload
    def __call__[Y](self, __o: _c.CanRound1[Y], __n: None = ..., /) -> Y: ...
    @overload
    def __call__[N, Y](self, __o: _c.CanRound2[N, Y], __n: N, /) -> Y: ...


@final
class DoesTrunc(Protocol):
    def __call__[Y](self, __o: _c.CanTrunc[Y], /) -> Y: ...


@final
class DoesFloor(Protocol):
    def __call__[Y](self, __o: _c.CanFloor[Y], /) -> Y: ...


@final
class DoesCeil(Protocol):
    def __call__[Y](self, __o: _c.CanCeil[Y], /) -> Y: ...


# async iteration

@final
class DoesANext(Protocol):
    @overload
    def __call__[V](self, __o: _c.CanANext[V], /) -> V: ...
    @overload
    async def __call__[V, V0](
        self, __o: _c.CanANext[_c.CanAwait[V]], __v0: V0, /,
    ) -> V | V0: ...


@final
class DoesAIter(Protocol):
    def __call__[Y: _c.CanANext[Any]](self, __o: _c.CanAIter[Y], /) -> Y: ...
