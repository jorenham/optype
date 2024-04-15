# ruff: noqa: PLW3201
"""
Elementary interfaces for special "dunder" attributes.
"""
import sys
from collections.abc import Callable
from dataclasses import Field as _Field
from types import CodeType, ModuleType
from typing import (
    Any,
    ClassVar,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    TypeVarTuple,
    runtime_checkable,
)


if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

from ._can import CanIter as _CanIter


_V = TypeVar('_V')
_V_match_args = TypeVar('_V_match_args', bound=tuple[str, ...] | list[str])
_V_slots = TypeVar('_V_slots', bound=str | _CanIter[Any])

_Xs = TypeVarTuple('_Xs')
_Xss = ParamSpec('_Xss')
_Y = TypeVar('_Y')


# Instances

@runtime_checkable
class HasMatchArgs(Protocol[_V_match_args]):
    __match_args__: _V_match_args


@runtime_checkable
class HasSlots(Protocol[_V_slots]):
    __slots__: _V_slots


@runtime_checkable
class HasDict(Protocol[_V]):
    __dict__: dict[str, _V]


@runtime_checkable
class HasClass(Protocol):
    @property
    @override
    def __class__(self) -> type[Self]: ...
    @__class__.setter
    def __class__(self, __cls: type[Self]) -> None:
        """Don't."""


@runtime_checkable
class HasModule(Protocol):
    __module__: str


# __name__ and __qualname__ generally are a package deal

@runtime_checkable
class HasName(Protocol):
    __name__: str


@runtime_checkable
class HasQualname(Protocol):
    __qualname__: str


@runtime_checkable
class HasNames(HasName, HasQualname, Protocol): ...


# docs and type hints

@runtime_checkable
class HasDoc(Protocol):
    __doc__: str | None


@runtime_checkable
class HasAnnotations(Protocol[_V]):
    __annotations__: dict[str, _V]


@runtime_checkable
class HasTypeParams(Protocol[*_Xs]):
    # Note that `*Ps: (TypeVar, ParamSpec, TypeVarTuple)` should hold
    __type_params__: tuple[*_Xs]


# functions and methods

@runtime_checkable
class HasFunc(Protocol[_Xss, _Y]):
    __func__: Callable[_Xss, _Y]


@runtime_checkable
class HasWrapped(Protocol[_Xss, _Y]):
    __wrapped__: Callable[_Xss, _Y]


_T_self_co = TypeVar('_T_self_co', bound=object | ModuleType, covariant=True)


@runtime_checkable
class HasSelf(Protocol[_T_self_co]):
    @property
    def __self__(self) -> _T_self_co: ...


@runtime_checkable
class HasCode(Protocol):
    __code__: CodeType


# Module `dataclasses`
# https://docs.python.org/3/library/dataclasses.html

@runtime_checkable
class HasDataclassFields(Protocol):
    """Can be used to check whether a type or instance is a dataclass."""
    __dataclass_fields__: ClassVar[dict[str, _Field[Any]]]
