from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar, Protocol


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import CodeType

    from ._can import CanIter, CanNext

if sys.version_info >= (3, 13):
    from typing import (
        LiteralString,
        ParamSpec,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        LiteralString,
        ParamSpec,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        override,
        runtime_checkable,
    )


from ._utils import set_module


###


_Ts = TypeVarTuple("_Ts")
_Tss = ParamSpec("_Tss")

_T_co = TypeVar("_T_co", covariant=True)
_ObjectT_co = TypeVar("_ObjectT_co", default=object, covariant=True)

_DictT = TypeVar("_DictT", bound="Mapping[str, object]", default=dict[str, object])
_DictT_co = TypeVar(
    "_DictT_co",
    bound="Mapping[str, object]",
    default=dict[str, object],
    covariant=True,
)
_NameT = TypeVar("_NameT", bound=str, default=str)
_QNameT = TypeVar("_QNameT", bound=str, default=str)
_StrT_co = TypeVar("_StrT_co", bound=str, default=str, covariant=True)


###


@set_module("optype")
@runtime_checkable
class HasMatchArgs(Protocol):
    __match_args__: ClassVar[tuple[LiteralString, ...] | list[LiteralString]]


@set_module("optype")
@runtime_checkable
class HasSlots(Protocol):
    __slots__: ClassVar[LiteralString | CanIter[CanNext[LiteralString]]]  # type: ignore[assignment]


@set_module("optype")
@runtime_checkable
class HasDict(Protocol[_DictT]):  # type: ignore[misc]
    # the typeshed annotations for `builtins.object.__dict__` too narrow
    __dict__: _DictT  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleVariableOverride]


@set_module("optype")
@runtime_checkable
class HasClass(Protocol):
    @property  # type: ignore[explicit-override]  # (basedmypy bug?)
    @override
    def __class__(self) -> type[Self]: ...
    @__class__.setter
    @override
    def __class__(self, cls: type[Self], /) -> None:
        """Don't."""


@set_module("optype")
@runtime_checkable
class HasModule(Protocol[_StrT_co]):
    __module__: _StrT_co


@set_module("optype")
@runtime_checkable
class HasName(Protocol[_NameT]):
    __name__: _NameT


@set_module("optype")
@runtime_checkable
class HasQualname(Protocol[_NameT]):  # pyright: ignore[reportInvalidTypeVarUse]
    __qualname__: _NameT


@set_module("optype")
@runtime_checkable
class HasNames(HasName[_NameT], HasQualname[_QNameT], Protocol[_NameT, _QNameT]): ...  # pyright: ignore[reportInvalidTypeVarUse]


# docs and type hints


@set_module("optype")
@runtime_checkable
class HasDoc(Protocol[_StrT_co]):
    # note that docstrings are stripped if ran with e.g. `python -OO`
    __doc__: _StrT_co | None


@set_module("optype")
@runtime_checkable
class HasAnnotations(Protocol[_DictT_co]):  # pyright: ignore[reportInvalidTypeVarUse]
    __annotations__: _DictT_co  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleVariableOverride]


# TODO(jorenham): https://github.com/jorenham/optype/issues/244
@set_module("optype")
@runtime_checkable
class HasTypeParams(Protocol[Unpack[_Ts]]):
    # Note that `*Ps: (TypeVar, ParamSpec, TypeVarTuple)` should hold
    __type_params__: tuple[Unpack[_Ts]]


# functions and methods


@set_module("optype")
@runtime_checkable
class HasFunc(Protocol[_Tss, _T_co]):
    @property
    def __func__(self, /) -> Callable[_Tss, _T_co]: ...


@set_module("optype")
@runtime_checkable
class HasWrapped(Protocol[_Tss, _T_co]):
    @property
    def __wrapped__(self, /) -> Callable[_Tss, _T_co]: ...


@set_module("optype")
@runtime_checkable
class HasSelf(Protocol[_ObjectT_co]):
    @property
    def __self__(self, /) -> _ObjectT_co: ...


@set_module("optype")
@runtime_checkable
class HasCode(Protocol):
    __code__: CodeType
