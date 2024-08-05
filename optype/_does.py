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


_KeyT = TypeVar('_KeyT')
_ValueT = TypeVar('_ValueT')
_AttrT = TypeVar('_AttrT')
_LeftT = TypeVar('_LeftT')
_RightT = TypeVar('_RightT')
_ModT = TypeVar('_ModT')
_NDigitsT = TypeVar('_NDigitsT')
_OutT = TypeVar('_OutT')
_DefaultT = TypeVar('_DefaultT')
_SentinelT = TypeVar('_SentinelT')
_IteratorT = TypeVar('_IteratorT', bound='_c.CanNext[Any]')
_AIteratorT = TypeVar('_AIteratorT', bound='_c.CanANext[Any]')
_IterT = TypeVar('_IterT', bound='_c.CanIter[Any]')
_BoolT = TypeVar('_BoolT', _JustTrue, _JustFalse, bool)
_IntT = TypeVar('_IntT', bound=int)
_StrT = TypeVar('_StrT', bound=str)
_FormatT = TypeVar('_FormatT', bound=str)
_BytesT = TypeVar('_BytesT', bound=bytes)
_ParamsT = ParamSpec('_ParamsT')

# iteration


@set_module('optype')
@final
class DoesNext(Protocol):
    @overload
    def __call__(self, iterator: _c.CanNext[_ValueT], /) -> _ValueT: ...
    @overload
    def __call__(
        self,
        iterator: _c.CanNext[_ValueT],
        default: _DefaultT,
        /,
    ) -> _ValueT | _DefaultT: ...


@set_module('optype')
@final
class DoesANext(Protocol):
    @overload
    def __call__(self, aiterator: _c.CanANext[_ValueT], /) -> _ValueT: ...
    @overload
    async def __call__(
        self,
        aiterator: _c.CanANext[_c.CanAwait[_ValueT]],
        default: _DefaultT,
        /,
    ) -> _ValueT | _DefaultT: ...


@set_module('optype')
@final
class DoesIter(Protocol):
    @overload
    def __call__(self, iterable: _c.CanIter[_IteratorT], /) -> _IteratorT: ...
    @overload
    def __call__(
        self,
        sequence: _c.CanGetitem[_c.CanIndex, _ValueT],
        /,
    ) -> _c.CanIterSelf[_ValueT]: ...
    @overload
    def __call__(
        self,
        callable_: _c.CanCall[[], _ValueT | None],
        sentinel: None,
        /,
    ) -> _c.CanIterSelf[_ValueT]: ...
    @overload
    def __call__(
        self,
        callable_: _c.CanCall[[], _ValueT | _SentinelT],
        sentinel: _SentinelT,
        /,
    ) -> _c.CanIterSelf[_ValueT]: ...


@set_module('optype')
@final
class DoesAIter(Protocol):
    def __call__(
        self,
        aiterable: _c.CanAIter[_AIteratorT],
        /,
    ) -> _AIteratorT: ...


# type conversion

@set_module('optype')
@final
class DoesComplex(Protocol):
    def __call__(self, obj: _c.CanComplex, /) -> complex: ...


@set_module('optype')
@final
class DoesFloat(Protocol):
    def __call__(self, obj: _c.CanFloat, /) -> float: ...


@set_module('optype')
@final
class DoesInt(Protocol):
    def __call__(self, obj: _c.CanInt[_IntT], /) -> _IntT: ...


@set_module('optype')
@final
class DoesBool(Protocol):
    @overload
    def __call__(self, obj: _c.CanBool[_BoolT], /) -> _BoolT: ...
    @overload
    def __call__(self, obj: _c.CanLen[_Just0], /) -> _JustFalse: ...
    @overload
    def __call__(self, obj: _c.CanLen[_PosInt], /) -> _JustTrue: ...
    @overload
    def __call__(self, obj: object, /) -> bool: ...


@set_module('optype')
@final
class DoesStr(Protocol):
    def __call__(self, obj: _c.CanStr[_StrT], /) -> _StrT: ...


@set_module('optype')
@final
class DoesBytes(Protocol):
    def __call__(self, obj: _c.CanBytes[_BytesT], /) -> _BytesT: ...


# formatting


@set_module('optype')
@final
class DoesRepr(Protocol):
    def __call__(self, obj: _c.CanRepr[_StrT], /) -> _StrT: ...


@set_module('optype')
@final
class DoesFormat(Protocol):
    def __call__(
        self,
        obj: _c.CanFormat[_FormatT, _StrT],
        format_spec: _FormatT = ...,
        /,
    ) -> _StrT: ...


# rich comparison


@set_module('optype')
@final
class DoesLt(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanLt[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanGt[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesLe(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanLe[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanGe[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesEq(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanEq[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(  # pyright: ignore[reportOverlappingOverload]
        self,
        lhs: _LeftT,
        rhs: _c.CanEq[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesNe(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanNe[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(  # pyright: ignore[reportOverlappingOverload]
        self,
        lhs: _LeftT,
        rhs: _c.CanNe[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesGt(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanGt[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanLt[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesGe(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanGe[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanLe[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


# dynamic attribute access


@set_module('optype')
@final
class DoesGetattr(Protocol):
    @overload
    def __call__(
        self,
        obj: _c.CanGetattr[_StrT, _AttrT],
        name: _StrT,
        /,
    ) -> _AttrT: ...
    @overload
    def __call__(
        self,
        obj: _c.CanGetattribute[_StrT, _AttrT],
        name: _StrT,
        /,
    ) -> _AttrT: ...
    @overload
    def __call__(
        self,
        obj: _c.CanGetattr[_StrT, _AttrT],
        name: _StrT,
        default: _DefaultT,
        /,
    ) -> _AttrT | _DefaultT: ...
    @overload
    def __call__(
        self,
        obj: _c.CanGetattribute[_StrT, _AttrT],
        name: _StrT,
        default: _DefaultT,
        /,
    ) -> _AttrT | _DefaultT: ...


@set_module('optype')
@final
class DoesSetattr(Protocol):
    def __call__(
        self,
        obj: _c.CanSetattr[_StrT, _AttrT],
        name: _StrT,
        value: _AttrT,
        /,
    ) -> None: ...


@set_module('optype')
@final
class DoesDelattr(Protocol):
    def __call__(self, obj: _c.CanDelattr[_StrT], name: _StrT, /) -> None: ...


@set_module('optype')
@final
class DoesDir(Protocol):
    @overload
    def __call__(self, /) -> list[str]: ...
    @overload
    def __call__(self, obj: _c.CanDir[_IterT], /) -> _IterT: ...


# callables


@set_module('optype')
@final
class DoesCall(Protocol):
    def __call__(
        self,
        callable_: Callable[_ParamsT, _OutT],
        /,
        *args: _ParamsT.args,
        **kwargs: _ParamsT.kwargs,
    ) -> _OutT: ...


# containers and subscriptable types


@set_module('optype')
@final
class DoesLen(Protocol):
    def __call__(self, obj: _c.CanLen[_IntT], /) -> _IntT: ...


@set_module('optype')
@final
class DoesLengthHint(Protocol):
    def __call__(self, obj: _c.CanLengthHint[_IntT], /) -> _IntT: ...


@set_module('optype')
@final
class DoesGetitem(Protocol):
    def __call__(
        self,
        obj: (
            _c.CanGetitem[_KeyT, _ValueT]
            | _c.CanGetMissing[_KeyT, _ValueT, _DefaultT]
        ),
        key: _KeyT,
        /,
    ) -> _ValueT | _DefaultT: ...


@set_module('optype')
@final
class DoesSetitem(Protocol):
    def __call__(
        self,
        obj: _c.CanSetitem[_KeyT, _ValueT],
        key: _KeyT,
        value: _ValueT,
        /,
    ) -> None: ...


@set_module('optype')
@final
class DoesDelitem(Protocol):
    def __call__(self, obj: _c.CanDelitem[_KeyT], key: _KeyT, /) -> None: ...


@set_module('optype')
@final
class DoesMissing(Protocol):
    def __call__(
        self,
        obj: _c.CanMissing[_KeyT, _DefaultT],
        key: _KeyT,
        /,
    ) -> _DefaultT: ...


@set_module('optype')
@final
class DoesContains(Protocol):
    def __call__(
        self,
        obj: _c.CanContains[_KeyT, _BoolT],
        key: _KeyT,
        /,
    ) -> _BoolT: ...


@set_module('optype')
@final
class DoesReversed(Protocol):
    """
    This is correct type of `builtins.reversed`.

    Note that typeshed's annotations for `reversed` are completely wrong:
    https://github.com/python/typeshed/issues/11645
    """
    @overload
    def __call__(self, reversible: _c.CanReversed[_OutT], /) -> _OutT: ...
    @overload
    def __call__(
        self,
        sequence: _c.CanSequence[_c.CanIndex, _ValueT],
        /,
    ) -> reversed[_ValueT]: ...


# binary infix operators


@set_module('optype')
@final
class DoesAdd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanAdd[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRAdd[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesSub(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanSub[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRSub[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesMul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanMul[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRMul[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesMatmul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanMatmul[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRMatmul[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesTruediv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanTruediv[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRTruediv[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesFloordiv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanFloordiv[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRFloordiv[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesMod(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanMod[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRMod[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesDivmod(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanDivmod[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRDivmod[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesPow(Protocol):
    @overload
    def __call__(
        self,
        base: _c.CanPow2[_RightT, _OutT],
        exp: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        base: _c.CanPow3[_RightT, _ModT, _OutT],
        exp: _RightT,
        mod: _ModT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        base: _LeftT,
        exp: _c.CanRPow[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesLshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanLshift[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRLshift[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanRshift[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRRshift[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesAnd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanAnd[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRAnd[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesXor(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanXor[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRXor[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesOr(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanOr[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanROr[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


# binary reflected operators


@set_module('optype')
@final
class DoesRAdd(Protocol):
    def __call__(
        self,
        rhs: _c.CanRAdd[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRSub(Protocol):
    def __call__(
        self,
        rhs: _c.CanRSub[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRMul(Protocol):
    def __call__(
        self,
        rhs: _c.CanRMul[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRMatmul(Protocol):
    def __call__(
        self,
        rhs: _c.CanRMatmul[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRTruediv(Protocol):
    def __call__(
        self,
        rhs: _c.CanRTruediv[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRFloordiv(Protocol):
    def __call__(
        self,
        rhs: _c.CanRFloordiv[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRMod(Protocol):
    def __call__(
        self,
        rhs: _c.CanRMod[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


class DoesRDivmod(Protocol):
    def __call__(
        self,
        rhs: _c.CanRDivmod[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRPow(Protocol):
    def __call__(
        self,
        rhs: _c.CanRPow[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRLshift(Protocol):
    def __call__(
        self,
        rhs: _c.CanRLshift[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRRshift(Protocol):
    def __call__(
        self,
        rhs: _c.CanRRshift[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRAnd(Protocol):
    def __call__(
        self,
        rhs: _c.CanRAnd[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesRXor(Protocol):
    def __call__(
        self,
        rhs: _c.CanRXor[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesROr(Protocol):
    def __call__(
        self,
        rhs: _c.CanROr[_LeftT, _OutT],
        lhs: _LeftT,
        /,
    ) -> _OutT: ...


# augmented / in-place operators


@set_module('optype')
@final
class DoesIAdd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIAdd[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanAdd[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRAdd[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesISub(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanISub[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanSub[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRSub[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIMul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIMul[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanMul[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRMul[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIMatmul(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIMatmul[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanMatmul[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRMatmul[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesITruediv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanITruediv[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanTruediv[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRTruediv[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIFloordiv(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIFloordiv[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanFloordiv[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRFloordiv[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIMod(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIMod[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanMod[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRMod[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIPow(Protocol):
    @overload
    def __call__(self,
        lhs: _c.CanIPow[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(self,
        lhs: _c.CanPow2[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(self,
        lhs: _LeftT,
        rhs: _c.CanRPow[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesILshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanILshift[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanLshift[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRLshift[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIRshift(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIRshift[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanRshift[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRRshift[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIAnd(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIAnd[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanAnd[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRAnd[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIXor(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIXor[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanXor[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanRXor[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesIOr(Protocol):
    @overload
    def __call__(
        self,
        lhs: _c.CanIOr[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _c.CanOr[_RightT, _OutT],
        rhs: _RightT,
        /,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        lhs: _LeftT,
        rhs: _c.CanROr[_LeftT, _OutT],
        /,
    ) -> _OutT: ...


# unary arithmetic


@set_module('optype')
@final
class DoesNeg(Protocol):
    def __call__(self, obj: _c.CanNeg[_OutT], /) -> _OutT: ...


@set_module('optype')
@final
class DoesPos(Protocol):
    def __call__(self, obj: _c.CanPos[_OutT], /) -> _OutT: ...


@set_module('optype')
@final
class DoesAbs(Protocol):
    def __call__(self, obj: _c.CanAbs[_OutT], /) -> _OutT: ...


@set_module('optype')
@final
class DoesInvert(Protocol):
    def __call__(self, obj: _c.CanInvert[_OutT], /) -> _OutT: ...


# object identification


@set_module('optype')
@final
class DoesIndex(Protocol):
    def __call__(self, obj: _c.CanIndex[_IntT], /) -> _IntT: ...


@set_module('optype')
@final
class DoesHash(Protocol):
    def __call__(self, obj: _c.CanHash, /) -> int: ...


# rounding


@set_module('optype')
@final
class DoesRound(Protocol):
    @overload
    def __call__(self, obj: _c.CanRound1[_OutT], /) -> _OutT: ...
    @overload
    def __call__(
        self,
        obj: _c.CanRound1[_OutT],
        /,
        ndigits: None = None,
    ) -> _OutT: ...
    @overload
    def __call__(
        self,
        obj: _c.CanRound2[_NDigitsT, _OutT],
        /,
        ndigits: _NDigitsT,
    ) -> _OutT: ...


@set_module('optype')
@final
class DoesTrunc(Protocol):
    def __call__(self, obj: _c.CanTrunc[_OutT], /) -> _OutT: ...


@set_module('optype')
@final
class DoesFloor(Protocol):
    def __call__(self, obj: _c.CanFloor[_OutT], /) -> _OutT: ...


@set_module('optype')
@final
class DoesCeil(Protocol):
    def __call__(self, obj: _c.CanCeil[_OutT], /) -> _OutT: ...
