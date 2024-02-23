# ruff: noqa: PLW3201
"""
Elementary interfaces for special "dunder" attributes.
"""
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable


if TYPE_CHECKING:
    import weakref as _weakref


# special attributes

@runtime_checkable
class HasDict[V](Protocol):
    __dict__: dict[str, V]


@runtime_checkable
class _HasDocAttr(Protocol):
    __doc__: str | None


@runtime_checkable
class _HasDocProp(Protocol):
    @property
    def __doc__(self) -> str | None: ...  # type: ignore[override]


HasDoc = _HasDocAttr | _HasDocProp


@runtime_checkable
class _HasNameAttr(Protocol):
    __name__: str


@runtime_checkable
class _HasNameProp(Protocol):
    @property
    def __name__(self) -> str: ...


HasName = _HasNameAttr | _HasNameProp


@runtime_checkable
class HasQualname(Protocol):
    __qualname__: str


@runtime_checkable
class _HasModuleAttr(Protocol):
    __module__: str


@runtime_checkable
class _HasModuleProp(Protocol):
    @property
    def __module__(self) -> str: ...  # type: ignore[override]


HasModule = _HasModuleAttr | _HasModuleProp


@runtime_checkable
class HasAnnotations[V](Protocol):
    """Note that the `V` type is hard to accurately define; blame PEP 563."""
    __annotations__: dict[str, V]


@runtime_checkable
class HasMatchArgs[Ks: tuple[str, ...] | list[str]](Protocol):
    __match_args__: Ks


# Module `dataclasses`
# https://docs.python.org/3/library/dataclasses.html

# TODO: HasDataclassFields


# Module `weakref`
# https://docs.python.org/3/library/weakref.html

@runtime_checkable
class HasWeakReference(Protocol):
    """An object referenced by a `weakref.ReferenceType[Self]`."""
    __weakref__: '_weakref.ReferenceType[Self]'


@runtime_checkable
class HasWeakCallableProxy[**Xs, Y](Protocol):
    """A callable referenced by a `weakref.CallableProxyType[Self]`."""
    __weakref__: '_weakref.CallableProxyType[Self]'

    def __call__(self, *__args: Xs.args, **__kwargs: Xs.kwargs) -> Y: ...

@runtime_checkable
class _HasWeakProxy(Protocol):
    __weakref__: '_weakref.ProxyType[Self]'


type HasWeakProxy = HasWeakCallableProxy[..., Any] | _HasWeakProxy
"""An object referenced by a `weakref.proxy` (not the proxy itself)."""
