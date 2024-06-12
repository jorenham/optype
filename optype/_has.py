from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Protocol,
    TypeAlias,
    runtime_checkable,
)


if sys.version_info >= (3, 13):
    from typing import (
        Never,
        ParamSpec,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        override,
    )
else:
    from typing_extensions import (
        Never,
        ParamSpec,
        Self,  # noqa: TCH002
        TypeVar,
        TypeVarTuple,
        Unpack,
        override,
    )

import optype._can as _c


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from dataclasses import Field as _Field
    from types import CodeType, ModuleType


_Ignored: TypeAlias = Any


_V_match_args = TypeVar(
    '_V_match_args',
    infer_variance=True,
    bound=tuple[str, ...] | list[str],
    default=Any,
)


@runtime_checkable
class HasMatchArgs(Protocol[_V_match_args]):
    __match_args__: ClassVar[tuple[str, ...] | list[str]]


_V_slots = TypeVar(
    '_V_slots',
    infer_variance=True,
    bound=str | _c.CanIter[_c.CanNext[str]],
    default=Any,
)


@runtime_checkable
class HasSlots(Protocol[_V_slots]):
    __slots__: Final[_V_slots]


_V_dict = TypeVar(
    '_V_dict',
    infer_variance=True,
    bound='Mapping[str, Any]',
    default=dict[str, Any],
)


@runtime_checkable
class HasDict(Protocol[_V_dict]):
    # the typeshed annotations for `builtins.object.__dict__` too narrow
    __dict__: _V_dict  # pyright: ignore[reportIncompatibleVariableOverride]


_V_class_set = TypeVar(
    '_V_class_set',
    infer_variance=True,
    bound=object,
    default=Never,
)


@runtime_checkable
class HasClass(Protocol[_V_class_set]):
    @property
    @override
    def __class__(self) -> type[Self | _V_class_set]: ...
    @__class__.setter
    def __class__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        cls: type[_V_class_set],
        /,
    ) -> _Ignored:
        """Don't."""


_V_module = TypeVar('_V_module', infer_variance=True, bound=str, default=str)


@runtime_checkable
class HasModule(Protocol[_V_module]):
    __module__: _V_module


_V_name = TypeVar('_V_name', infer_variance=True, bound=str, default=str)


@runtime_checkable
class HasName(Protocol[_V_name]):
    __name__: _V_name


_V_qualname = TypeVar(
    '_V_qualname',
    infer_variance=True,
    bound=str,
    default=str,
)


@runtime_checkable
class HasQualname(Protocol[_V_qualname]):
    __qualname__: _V_qualname


_V_names_name = TypeVar(
    '_V_names_name',
    infer_variance=True,
    bound=str,
    default=str,
)
_V_names_qualname = TypeVar(
    '_V_names_qualname',
    infer_variance=True,
    bound=str,
    default=_V_names_name,
)


@runtime_checkable
class HasNames(
    HasName[_V_names_name],
    HasQualname[_V_names_qualname],
    Protocol[_V_names_name, _V_names_qualname],
):
    __name__: _V_names_name
    __qualname__: _V_names_qualname


# docs and type hints

_V_doc = TypeVar(
    '_V_doc',
    infer_variance=True,
    bound=str,
    default=str,
)


@runtime_checkable
class HasDoc(Protocol[_V_doc]):
    # note that docstrings are stripped if ran with e.g. `python -OO`
    __doc__: _V_doc | None


_V_annotations = TypeVar(
    '_V_annotations',
    infer_variance=True,
    bound='Mapping[str, Any]',
    default=dict[str, Any],
)


@runtime_checkable
class HasAnnotations(Protocol[_V_annotations]):
    __annotations__: _V_annotations  # pyright: ignore[reportIncompatibleVariableOverride]


# should be one of `(TypeVar, TypeVarTuple, ParamSpec)`
_Ps_type_params = TypeVarTuple('_Ps_type_params')


@runtime_checkable
class HasTypeParams(Protocol[Unpack[_Ps_type_params]]):
    # Note that `*Ps: (TypeVar, ParamSpec, TypeVarTuple)` should hold
    __type_params__: tuple[Unpack[_Ps_type_params]]


# functions and methods

_Pss_func = ParamSpec('_Pss_func')
_V_func = TypeVar('_V_func', infer_variance=True)


@runtime_checkable
class HasFunc(Protocol[_Pss_func, _V_func]):
    __func__: Callable[_Pss_func, _V_func]


_Pss_wrapped = ParamSpec('_Pss_wrapped')
_V_wrapped = TypeVar('_V_wrapped', infer_variance=True)


@runtime_checkable
class HasWrapped(Protocol[_Pss_wrapped, _V_wrapped]):
    __wrapped__: Callable[_Pss_wrapped, _V_wrapped]


_V_self = TypeVar(
    '_V_self',
    infer_variance=True,
    bound='ModuleType | object',
    default=object,
)


@runtime_checkable
class HasSelf(Protocol[_V_self]):
    @property
    def __self__(self) -> _V_self: ...


@runtime_checkable
class HasCode(Protocol):
    __code__: CodeType


# Module `dataclasses`
# https://docs.python.org/3/library/dataclasses.html

_V_dataclass_fields = TypeVar(
    '_V_dataclass_fields',
    infer_variance=True,
    bound='Mapping[str, _Field[Any]]',
    default=dict[str, '_Field[Any]'],
)


@runtime_checkable
class HasDataclassFields(Protocol[_V_dataclass_fields]):
    """Can be used to check whether a type or instance is a dataclass."""
    __dataclass_fields__: _V_dataclass_fields
