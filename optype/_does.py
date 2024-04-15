from typing import (
    Any,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

import optype._can as _c


_V = TypeVar('_V')
_D = TypeVar('_D')
_V_next = TypeVar('_V_next', bound=_c.CanNext[Any])
_V_anext = TypeVar('_V_anext', bound=_c.CanANext[Any])

_X = TypeVar('_X')
_X_str = TypeVar('_X_str', bound=str)
_X_optional = TypeVar('_X_optional', bound=object | None)

_Y = TypeVar('_Y')
_Y_str = TypeVar('_Y_str', bound=str)
_Y_bytes = TypeVar('_Y_bytes', bound=bytes)

_Xss = ParamSpec('_Xss')


# iteration


@final
class DoesNext(Protocol):
    # https://docs.python.org/3/library/functions.html#next
    @overload
    def __call__(self, __vs: _c.CanNext[_V], /) -> _V: ...
    @overload
    def __call__(self, __vs: _c.CanNext[_V], __v0: _D, /) -> _V | _D: ...


@final
class DoesIter(Protocol):
    # https://docs.python.org/3/library/functions.html#iter
    @overload
    def __call__(self, __vs: _c.CanIter[_V_next], /) -> _V_next: ...
    @overload
    def __call__(
        self,
        __vs: _c.CanGetitem[_c.CanIndex, _V],
        /,
    ) -> _c.CanIterSelf[_V]: ...
    @overload
    def __call__(
        self,
        __f: _c.CanCall[[], _V | _X_optional],
        __s: _X_optional,
        /,
    ) -> _c.CanIterSelf[_V]: ...


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
    def __call__(self, __o: _c.CanStr[_Y_str], /) -> _Y_str: ...


@final
class DoesBytes(Protocol):
    # https://docs.python.org/3/library/functions.html#func-bytes
    def __call__(self, __o: _c.CanBytes[_Y_bytes], /) -> _Y_bytes: ...


# formatting

@final
class DoesRepr(Protocol):
    # https://docs.python.org/3/library/functions.html#repr
    def __call__(self, __o: _c.CanRepr[_Y_str], /) -> _Y_str: ...


@final
class DoesFormat(Protocol):
    # https://docs.python.org/3/library/functions.html#format
    def __call__(
        self,
        __o: _c.CanFormat[_X_str, _Y_str],
        __x: _X_str = ...,
        /,
    ) -> _Y_str: ...


# rich comparison

@final
class DoesLt(Protocol):
    @overload
    def __call__(self, __o: _c.CanLt[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanGt[_X, _Y], /) -> _Y: ...


@final
class DoesLe(Protocol):
    @overload
    def __call__(self, __o: _c.CanLe[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanGe[_X, _Y], /) -> _Y: ...


@final
class DoesEq(Protocol):
    @overload
    def __call__(self, __o: _c.CanEq[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanEq[_X, _Y], /) -> _Y: ...


@final
class DoesNe(Protocol):
    @overload
    def __call__(self, __o: _c.CanNe[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanNe[_X, _Y], /) -> _Y: ...


@final
class DoesGt(Protocol):
    @overload
    def __call__(self, __o: _c.CanGt[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanLt[_X, _Y], /) -> _Y: ...


@final
class DoesGe(Protocol):
    @overload
    def __call__(self, __o: _c.CanGe[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanLe[_X, _Y], /) -> _Y: ...


# attributes
_CanGetAttr: TypeAlias = (
    _c.CanGetattr[_X_str, _V]
    | _c.CanGetattribute[_X_str, _V]
)


@final
class DoesGetattr(Protocol):
    # https://docs.python.org/3/library/functions.html#getattr
    @overload
    def __call__(self, __o: _CanGetAttr[_X_str, _V], __k: _X_str, /) -> _V: ...
    @overload
    def __call__(
        self,
        __o: _CanGetAttr[_X_str, _V],
        __k: _X_str,
        __v0: _D,
        /,
    ) -> _V | _D: ...


@final
class DoesSetattr(Protocol):
    # https://docs.python.org/3/library/functions.html#setattr
    def __call__(
        self,
        __o: _c.CanSetattr[_X_str, _V],
        __k: _X_str,
        __v: _V,
        /,
    ) -> None: ...


@final
class DoesDelattr(Protocol):
    # https://docs.python.org/3/library/functions.html#delattr
    def __call__(self, __o: _c.CanDelattr[_X_str], __k: _X_str, /) -> None: ...


_Vss = TypeVar('_Vss', bound=_c.CanIter[Any])


@final
class DoesDir(Protocol):
    # https://docs.python.org/3/library/functions.html#dir
    @overload
    def __call__(self, /) -> list[str]: ...
    @overload
    def __call__(self, __o: _c.CanDir[_Vss], /) -> _Vss: ...


# callables

@final
class DoesCall(Protocol):
    def __call__(
        self,
        __f: _c.CanCall[_Xss, _Y],
        *__xs: _Xss.args,
        **__kxs: _Xss.kwargs,
    ) -> _Y: ...


# containers and subscritable types


@final
class DoesLen(Protocol):
    def __call__(self, __o: _c.CanLen, /) -> int: ...


@final
class DoesLengthHint(Protocol):
    def __call__(self, __o: _c.CanLengthHint, /) -> int: ...


_CanSubscript: TypeAlias = _c.CanGetitem[_X, _V] | _c.CanGetMissing[_X, _V, _D]


@final
class DoesGetitem(Protocol):
    def __call__(
        self,
        __o: _CanSubscript[_X, _V, _D],
        __k: _X,
        /,
    ) -> _V | _D: ...


@final
class DoesSetitem(Protocol):
    def __call__(
        self,
        __o: _c.CanSetitem[_X, _V],
        __k: _X,
        __v: _V,
        /,
    ) -> None: ...


@final
class DoesDelitem(Protocol):
    def __call__(self, __o: _c.CanDelitem[_X], __k: _X, /) -> None: ...


@final
class DoesMissing(Protocol):
    def __call__(self, __o: _c.CanMissing[_X, _V], __k: _X, /) -> _V: ...


@final
class DoesContains(Protocol):
    def __call__(self, __o: _c.CanContains[_X], __k: _X, /) -> bool: ...


@final
class DoesReversed(Protocol):
    """
    This is correct type of `builtins.reversed`.

    Note that typeshed's annotations for `reversed` are completely wrong:
    https://github.com/python/typeshed/issues/11645
    """
    @overload
    def __call__(self, __o: _c.CanReversed[_V], /) -> _V: ...
    @overload
    def __call__(self, __o: _c.CanSequence[Any, _V], /) -> 'reversed[_V]': ...


# binary infix operators

@final
class DoesAdd(Protocol):
    @overload
    def __call__(self, __o: _c.CanAdd[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRAdd[_X, _Y], /) -> _Y: ...


@final
class DoesSub(Protocol):
    @overload
    def __call__(self, __o: _c.CanSub[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRSub[_X, _Y], /) -> _Y: ...


@final
class DoesMul(Protocol):
    @overload
    def __call__(self, __o: _c.CanMul[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRMul[_X, _Y], /) -> _Y: ...


@final
class DoesMatmul(Protocol):
    @overload
    def __call__(self, __o: _c.CanMatmul[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRMatmul[_X, _Y], /) -> _Y: ...


@final
class DoesTruediv(Protocol):
    @overload
    def __call__(self, __o: _c.CanTruediv[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRTruediv[_X, _Y], /) -> _Y: ...


@final
class DoesFloordiv(Protocol):
    @overload
    def __call__(self, __o: _c.CanFloordiv[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRFloordiv[_X, _Y], /) -> _Y: ...


@final
class DoesMod(Protocol):
    @overload
    def __call__(self, __o: _c.CanMod[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRMod[_X, _Y], /) -> _Y: ...


@final
class DoesDivmod(Protocol):
    @overload
    def __call__(self, __o: _c.CanDivmod[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRDivmod[_X, _Y], /) -> _Y: ...


@final
class DoesPow(Protocol):
    @overload
    def __call__(self, __o: _c.CanPow2[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(
        self,
        __o: _c.CanPow3[_X, _D, _Y],
        __x: _X,
        __mod: _D,
        /,
    ) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRPow[_X, _Y], /) -> _Y: ...


@final
class DoesLshift(Protocol):
    @overload
    def __call__(self, __o: _c.CanLshift[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRLshift[_X, _Y], /) -> _Y: ...


@final
class DoesRshift(Protocol):
    @overload
    def __call__(self, __o: _c.CanRshift[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRRshift[_X, _Y], /) -> _Y: ...


@final
class DoesAnd(Protocol):
    @overload
    def __call__(self, __o: _c.CanAnd[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRAnd[_X, _Y], /) -> _Y: ...


@final
class DoesXor(Protocol):
    @overload
    def __call__(self, __o: _c.CanXor[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanRXor[_X, _Y], /) -> _Y: ...


@final
class DoesOr(Protocol):
    @overload
    def __call__(self, __o: _c.CanOr[_X, _Y], __x: _X, /) -> _Y: ...
    @overload
    def __call__(self, __x: _X, __o: _c.CanROr[_X, _Y], /) -> _Y: ...


# binary reflected operators

@final
class DoesRAdd(Protocol):
    def __call__(self, __o: _c.CanRAdd[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRSub(Protocol):
    def __call__(self, __o: _c.CanRSub[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRMul(Protocol):
    def __call__(self, __o: _c.CanRMul[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRMatmul(Protocol):
    def __call__(self, __o: _c.CanRMatmul[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRTruediv(Protocol):
    def __call__(self, __o: _c.CanRTruediv[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRFloordiv(Protocol):
    def __call__(self, __o: _c.CanRFloordiv[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRMod(Protocol):
    def __call__(self, __o: _c.CanRMod[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRDivmod(Protocol):
    def __call__(self, __o: _c.CanRDivmod[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRPow(Protocol):
    def __call__(self, __o: _c.CanRPow[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRLshift(Protocol):
    def __call__(self, __o: _c.CanRLshift[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRRshift(Protocol):
    def __call__(self, __o: _c.CanRRshift[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRAnd(Protocol):
    def __call__(self, __o: _c.CanRAnd[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesRXor(Protocol):
    def __call__(self, __o: _c.CanRXor[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesROr(Protocol):
    def __call__(self, __o: _c.CanROr[_X, _Y], __x: _X, /) -> _Y: ...


# augmented / in-place operators

@final
class DoesIAdd(Protocol):
    def __call__(self, __o: _c.CanIAdd[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesISub(Protocol):
    def __call__(self, __o: _c.CanISub[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIMul(Protocol):
    def __call__(self, __o: _c.CanIMul[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIMatmul(Protocol):
    def __call__(self, __o: _c.CanIMatmul[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesITruediv(Protocol):
    def __call__(self, __o: _c.CanITruediv[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIFloordiv(Protocol):
    def __call__(self, __o: _c.CanIFloordiv[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIMod(Protocol):
    def __call__(self, __o: _c.CanIMod[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIPow(Protocol):
    def __call__(self, __o: _c.CanIPow[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesILshift(Protocol):
    def __call__(self, __o: _c.CanILshift[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIRshift(Protocol):
    def __call__(self, __o: _c.CanIRshift[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIAnd(Protocol):
    def __call__(self, __o: _c.CanIAnd[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIXor(Protocol):
    def __call__(self, __o: _c.CanIXor[_X, _Y], __x: _X, /) -> _Y: ...


@final
class DoesIOr(Protocol):
    def __call__(self, __o: _c.CanIOr[_X, _Y], __x: _X, /) -> _Y: ...


# unary arithmetic

@final
class DoesNeg(Protocol):
    def __call__(self, __o: _c.CanNeg[_Y], /) -> _Y: ...


@final
class DoesPos(Protocol):
    def __call__(self, __o: _c.CanPos[_Y], /) -> _Y: ...


@final
class DoesAbs(Protocol):
    def __call__(self, __o: _c.CanAbs[_Y], /) -> _Y: ...


@final
class DoesInvert(Protocol):
    def __call__(self, __o: _c.CanInvert[_Y], /) -> _Y: ...


# fingerprinting

@final
class DoesHash(Protocol):
    def __call__(self, __o: _c.CanHash, /) -> int: ...


@final
class DoesIndex(Protocol):
    def __call__(self, __o: _c.CanIndex, /) -> int: ...


# rounding

@final
class DoesRound(Protocol):
    @overload
    def __call__(self, __o: _c.CanRound1[_Y], /) -> _Y: ...
    @overload
    def __call__(self, __o: _c.CanRound1[_Y], __n: None = ..., /) -> _Y: ...
    @overload
    def __call__(self, __o: _c.CanRound2[_X, _Y], __n: _X, /) -> _Y: ...


@final
class DoesTrunc(Protocol):
    def __call__(self, __o: _c.CanTrunc[_Y], /) -> _Y: ...


@final
class DoesFloor(Protocol):
    def __call__(self, __o: _c.CanFloor[_Y], /) -> _Y: ...


@final
class DoesCeil(Protocol):
    def __call__(self, __o: _c.CanCeil[_Y], /) -> _Y: ...


# async iteration

@final
class DoesANext(Protocol):
    @overload
    def __call__(self, __o: _c.CanANext[_V], /) -> _V: ...
    @overload
    async def __call__(
        self,
        __o: _c.CanANext[_c.CanAwait[_V]],
        __v0: _D,
        /,
    ) -> _V | _D: ...


@final
class DoesAIter(Protocol):
    def __call__(self, __o: _c.CanAIter[_V_anext], /) -> _V_anext: ...
