from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar, Protocol


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


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import CodeType

    from ._can import CanIter, CanNext


@set_module("optype")
@runtime_checkable
class HasMatchArgs(Protocol):
    __match_args__: ClassVar[tuple[LiteralString, ...] | list[LiteralString]]


@set_module("optype")
@runtime_checkable
class HasSlots(Protocol):
    __slots__: ClassVar[LiteralString | CanIter[CanNext[LiteralString]]]  # type: ignore[assignment]


_DictT = TypeVar("_DictT", bound="Mapping[str, object]", default=dict[str, object])


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


_ModuleT_co = TypeVar("_ModuleT_co", covariant=True, bound=str, default=str)


@set_module("optype")
@runtime_checkable
class HasModule(Protocol[_ModuleT_co]):
    __module__: _ModuleT_co


_NameT = TypeVar("_NameT", bound=str, default=str)


@set_module("optype")
@runtime_checkable
class HasName(Protocol[_NameT]):
    __name__: _NameT


_QualnameT = TypeVar("_QualnameT", bound=str, default=str)


@set_module("optype")
@runtime_checkable
class HasQualname(Protocol[_QualnameT]):  # pyright: ignore[reportInvalidTypeVarUse]
    __qualname__: _QualnameT


@set_module("optype")
@runtime_checkable
class HasNames(  # pyright: ignore[reportInvalidTypeVarUse]
    HasName[_NameT],
    HasQualname[_QualnameT],
    Protocol[_NameT, _QualnameT],
): ...


# docs and type hints

_DocT_co = TypeVar("_DocT_co", covariant=True, bound=str, default=str)


@set_module("optype")
@runtime_checkable
class HasDoc(Protocol[_DocT_co]):
    # note that docstrings are stripped if ran with e.g. `python -OO`
    __doc__: _DocT_co | None


_AnnotationsT_co = TypeVar(
    "_AnnotationsT_co",
    covariant=True,
    bound=dict[str, object],
    default=dict[str, object],
)


@set_module("optype")
@runtime_checkable
class HasAnnotations(Protocol[_AnnotationsT_co]):  # pyright: ignore[reportInvalidTypeVarUse]
    __annotations__: _AnnotationsT_co  # pyright: ignore[reportIncompatibleVariableOverride]


# should be one of `(TypeVar, TypeVarTuple, ParamSpec)`
_TypeParamsTs = TypeVarTuple("_TypeParamsTs")


@set_module("optype")
@runtime_checkable
class HasTypeParams(Protocol[Unpack[_TypeParamsTs]]):
    # Note that `*Ps: (TypeVar, ParamSpec, TypeVarTuple)` should hold
    __type_params__: tuple[Unpack[_TypeParamsTs]]


# functions and methods

_FuncTss = ParamSpec("_FuncTss")
_FuncT_co = TypeVar("_FuncT_co", covariant=True)


@set_module("optype")
@runtime_checkable
class HasFunc(Protocol[_FuncTss, _FuncT_co]):
    @property
    def __func__(self) -> Callable[_FuncTss, _FuncT_co]: ...


_WrappedTss = ParamSpec("_WrappedTss")
_WrappedT_co = TypeVar("_WrappedT_co", covariant=True)


@set_module("optype")
@runtime_checkable
class HasWrapped(Protocol[_WrappedTss, _WrappedT_co]):
    @property
    def __wrapped__(self) -> Callable[_WrappedTss, _WrappedT_co]: ...


_SelfT_co = TypeVar("_SelfT_co", covariant=True, default=object)


@set_module("optype")
@runtime_checkable
class HasSelf(Protocol[_SelfT_co]):
    @property
    def __self__(self) -> _SelfT_co: ...


@set_module("optype")
@runtime_checkable
class HasCode(Protocol):
    __code__: CodeType
