# ruff: noqa: PLW3201
"""
Elementary interfaces for special "dunder" attributes.
"""
from types import CodeType, ModuleType
from typing import Any, Protocol, Self, override, runtime_checkable

from ._can import (
    CanCall as _CanCall,
    CanIter as _CanIter,
)


# Instances

@runtime_checkable
class HasMatchArgs[Ks: tuple[str, ...] | list[str]](Protocol):
    __match_args__: Ks


@runtime_checkable
class HasSlots[S: str | _CanIter[Any]](Protocol):
    __slots__: S


@runtime_checkable
class HasDict[V](Protocol):
    __dict__: dict[str, V]


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
class HasAnnotations[V](Protocol):
    __annotations__: dict[str, V]


@runtime_checkable
class HasTypeParams[*Ps](Protocol):
    # Note that `*Ps: (TypeVar, ParamSpec, TypeVarTuple)` should hold
    __type_params__: tuple[*Ps]


# functions and methods

@runtime_checkable
class HasFunc[**Xs, Y](Protocol):
    __func__: _CanCall[Xs, Y]


@runtime_checkable
class HasWrapped[**Xs, Y](Protocol):
    __wrapped__: _CanCall[Xs, Y]


@runtime_checkable
class HasSelf[T: object | ModuleType](Protocol):
    @property
    def __self__(self) -> T: ...


@runtime_checkable
class HasCode(Protocol):
    __code__: CodeType


# Module `dataclasses`
# https://docs.python.org/3/library/dataclasses.html

# TODO: HasDataclassFields
