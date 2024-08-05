from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias


if sys.version_info >= (3, 13):
    from typing import (
        LiteralString,
        ParamSpec,
        Protocol,
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
        Protocol,
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


_Ignored: TypeAlias = Any


@set_module('optype')
@runtime_checkable
class HasMatchArgs(Protocol):
    __match_args__: ClassVar[tuple[LiteralString, ...] | list[LiteralString]]


@set_module('optype')
@runtime_checkable
class HasSlots(Protocol):
    __slots__: ClassVar[LiteralString | CanIter[CanNext[LiteralString]]]


_DictT = TypeVar('_DictT', bound='Mapping[str, Any]', default=dict[str, Any])


@set_module('optype')
@runtime_checkable
class HasDict(Protocol[_DictT]):
    # the typeshed annotations for `builtins.object.__dict__` too narrow
    __dict__: _DictT  # pyright: ignore[reportIncompatibleVariableOverride]


@set_module('optype')
@runtime_checkable
class HasClass(Protocol):
    @property
    @override
    def __class__(self) -> type[Self]: ...
    @__class__.setter
    def __class__(self, cls: type[Self], /) -> _Ignored:
        """Don't."""


_ModuleT_co = TypeVar('_ModuleT_co', covariant=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class HasModule(Protocol[_ModuleT_co]):
    __module__: _ModuleT_co


_NameT_co = TypeVar('_NameT_co', covariant=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class HasName(Protocol[_NameT_co]):
    __name__: Final[_NameT_co]


_QualnameT_co = TypeVar(
    '_QualnameT_co',
    covariant=True,
    bound=str,
    default=str,
)


@set_module('optype')
@runtime_checkable
class HasQualname(Protocol[_QualnameT_co]):
    __qualname__: _QualnameT_co


@set_module('optype')
@runtime_checkable
class HasNames(
    HasName[_NameT_co],
    HasQualname[_QualnameT_co],
    Protocol[_NameT_co, _QualnameT_co],
): ...


# docs and type hints

_DocT_co = TypeVar('_DocT_co', covariant=True, bound=str, default=str)


@set_module('optype')
@runtime_checkable
class HasDoc(Protocol[_DocT_co]):
    # note that docstrings are stripped if ran with e.g. `python -OO`
    __doc__: _DocT_co | None


_AnnotationsT_co = TypeVar(
    '_AnnotationsT_co',
    covariant=True,
    bound='Mapping[str, Any]',
    default=dict[str, Any],
)


@set_module('optype')
@runtime_checkable
class HasAnnotations(Protocol[_AnnotationsT_co]):
    __annotations__: Final[_AnnotationsT_co]  # pyright: ignore[reportIncompatibleVariableOverride]


# should be one of `(TypeVar, TypeVarTuple, ParamSpec)`
_TypeParamsTs = TypeVarTuple('_TypeParamsTs')


@set_module('optype')
@runtime_checkable
class HasTypeParams(Protocol[Unpack[_TypeParamsTs]]):
    # Note that `*Ps: (TypeVar, ParamSpec, TypeVarTuple)` should hold
    __type_params__: tuple[Unpack[_TypeParamsTs]]


# functions and methods

_FuncTss = ParamSpec('_FuncTss')
_FuncT_co = TypeVar('_FuncT_co', covariant=True)


@set_module('optype')
@runtime_checkable
class HasFunc(Protocol[_FuncTss, _FuncT_co]):
    @property
    def __func__(self) -> Callable[_FuncTss, _FuncT_co]: ...


_WrappedTss = ParamSpec('_WrappedTss')
_WrappedT_co = TypeVar('_WrappedT_co', covariant=True)


@set_module('optype')
@runtime_checkable
class HasWrapped(Protocol[_WrappedTss, _WrappedT_co]):
    @property
    def __wrapped__(self) -> Callable[_WrappedTss, _WrappedT_co]: ...


_SelfT_co = TypeVar('_SelfT_co', covariant=True, default=object)


@set_module('optype')
@runtime_checkable
class HasSelf(Protocol[_SelfT_co]):
    @property
    def __self__(self) -> _SelfT_co: ...


@set_module('optype')
@runtime_checkable
class HasCode(Protocol):
    __code__: CodeType
