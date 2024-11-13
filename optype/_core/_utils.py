from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol


if sys.version_info >= (3, 13):
    from typing import LiteralString, TypeVar, is_protocol
else:
    from typing_extensions import LiteralString, TypeVar, is_protocol

if TYPE_CHECKING:
    from types import ModuleType


__all__ = "get_callables", "set_module"


def get_callables(
    module: ModuleType,
    /,
    *,
    protocols: bool = False,
    private: bool = False,
) -> frozenset[str]:
    """
    Return the public callables (types are callables too) in the given module,
    except for `typing.Protocol`.
    """
    exclude = frozenset({"typing", "typing_extensions", "optype._core._utils"})
    return frozenset({
        name
        for name in dir(module)
        if callable(cls := getattr(module, name))  # pyright: ignore[reportAny]
        and (private or not name.startswith("_"))
        and (protocols or not (isinstance(cls, type) and is_protocol(cls)))
        and cls.__module__ not in exclude
    })


# cannot reuse `optype._has._HasModule` due to circular imports
class _HasModule(Protocol):
    __module__: str


_HasModuleT = TypeVar("_HasModuleT", bound=_HasModule)


class _DoesSetModule(Protocol):
    def __call__(self, obj: _HasModuleT, /) -> _HasModuleT: ...


def set_module(module: LiteralString, /) -> _DoesSetModule:
    """
    Private decorator for overriding the `__module__` of a function or a class.

    If used on a function that `typing.overload`s, then apply `@set_module`
    to each overload *first*, to avoid breaking `typing.get_overloads`.
    For example:

    ```python
    from typing import overload


    @overload
    @set_module("spamlib")
    def process(response: None, /) -> None: ...


    @overload
    @set_module("spamlib")
    def process(response: bytes, /) -> str: ...


    @set_module("spamlib")
    def process(response: byes | None, /) -> str | None:
        pass  # implementation here
    ```
    """
    assert module
    assert all(map(str.isidentifier, module.split(".")))

    def do_set_module(has_module: _HasModuleT, /) -> _HasModuleT:
        assert hasattr(has_module, "__module__")
        has_module.__module__ = module
        return has_module

    return do_set_module
