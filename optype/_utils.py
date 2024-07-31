from __future__ import annotations

import sys


if sys.version_info >= (3, 13):
    from typing import LiteralString, Protocol, TypeVar, final
else:
    from typing_extensions import (  # noqa: TCH002
        LiteralString,
        Protocol,
        TypeVar,
        final,
    )


__all__ = ('set_module',)


# cannot reuse `optype._has._HasModule` due to circular imports
class _HasModule(Protocol):
    __module__: str


_HasModuleT = TypeVar('_HasModuleT', bound=_HasModule)


@final
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
    @set_module('spamlib')
    def process(response: None, /) -> None: ...

    @overload
    @set_module('spamlib')
    def process(response: bytes, /) -> str: ...

    @set_module('spamlib')
    def process(response: byes | None, /) -> str | None:
        ...  # implementation here
    ```
    """
    assert module
    assert all(map(str.isidentifier, module.split('.')))

    def do_set_module(has_module: _HasModuleT, /) -> _HasModuleT:
        assert hasattr(has_module, '__module__')
        has_module.__module__ = module
        return has_module

    return do_set_module
