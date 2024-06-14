from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Protocol as _typing_Protocol, cast


if sys.version_info >= (3, 13):
    from typing import Protocol
else:
    from typing_extensions import Protocol


if TYPE_CHECKING:
    from types import ModuleType

    from optype import CanBool, CanLt


__all__ = (
    'get_callable_members',
    'get_protocol_members',
    'get_protocols',
    'is_dunder',
    'is_protocol',
    'pascamel_to_snake',
)


def is_protocol(cls: type, /) -> bool:
    """Based on `typing_extensions.is_protocol`."""
    return (
        isinstance(cls, type)
        and cls is not Protocol
        and cls is not _typing_Protocol
        and (issubclass(cls, Protocol) or issubclass(cls, _typing_Protocol))
        and getattr(cls, '_is_protocol', False)
    )


def is_runtime_protocol(cls: type, /) -> bool:
    """Check if `cls` is a `@runtime_checkable` `typing.Protocol`."""
    return is_protocol(cls) and getattr(cls, '_is_runtime_protocol', False)


def get_protocol_members(cls: type, /) -> frozenset[str]:
    """
    A variant of `typing_extensions.get_protocol_members()` that doesn't
    hide e.g. `__dict__` and `__annotations__`, or adds `__hash__` if there's
    an `__eq__` method.
    Does not return method names of base classes defined in another module.
    """
    assert is_protocol(cls)

    module = cls.__module__
    annotations = cls.__annotations__

    members = annotations.keys() | {
        name for name, v in vars(cls).items()
        if (
            name != '__new__'
            and callable(v)
            and (
                v.__module__ == module
                or (
                    # Fun fact: Each `@overload` returns the same dummy
                    # function; so there's no reference your wrapped method :).
                    # Oh and BTW; `typing.get_overloads` only works on the
                    # non-overloaded method...
                    # Oh, you mean the one that # you shouldn't define within
                    # a `typing.Protocol`?
                    # Yes exactly! Anyway, good luck searching for the
                    # undocumented and ever-changing dark corner of the
                    # `typing` internals. I'm sure it must be there somewhere!
                    # Oh yea if you can't find it, try `typing_extensions`.
                    # Oh, still can't find it? Did you try ALL THE VERSIONS?
                    #
                    # ...anyway, the only thing we know here, is the name of
                    # an overloaded method. But we have no idea how many of
                    # them there *were*, let alone their signatures.
                    v.__module__.startswith('typing')
                    and v.__name__ == '_overload_dummy'
                )
            )
        ) or (
            isinstance(v, property)
            and v.fget
            and v.fget.__module__ == module
        )
    }

    # this hack here is plagiarized from the (often incorrect)
    # `typing_extensions.get_protocol_members`.
    # Maybe the `typing.get_protocol_member`s` that's coming in 3.13 will
    # won't be as broken. I have little hope though...
    members |= cast(
        set[str],
        getattr(cls, '__protocol_attrs__', None) or set(),
    )

    # sometimes __protocol_attrs__ hallicunates some non-existing dunders.
    # the `getattr_static` avoids potential descriptor magic
    members = {
        member for member in members
        if member in annotations
        or inspect.getattr_static(cls, member) is not None
        # or getattr(cls, member) is not None
    }

    # also include any of the parents
    for supercls in cls.mro()[1:]:
        if is_protocol(supercls):
            members |= get_protocol_members(supercls)

    return frozenset(members)


def get_protocols(module: ModuleType) -> frozenset[type]:
    """Return the public protocol types within the given module."""
    return frozenset({
        cls for name in dir(module)
        if not name.startswith('_')
        and is_protocol(cls := getattr(module, name))
    })


def get_callable_members(module: ModuleType) -> frozenset[str]:
    """Return the public protocol types within the given module."""
    return frozenset({
        name for name in dir(module)
        if not name.startswith('_')
        and callable(cls := getattr(module, name))
        and not is_protocol(cls)
        and getattr(module, name).__module__ != 'typing'
    })


def pascamel_to_snake(
    pascamel: str,
    start: CanLt[int, CanBool] = 0,
    /,
) -> str:
    """Converts 'CamelCase' or 'pascalCase' to 'snake_case'."""
    assert pascamel.isidentifier()

    snake = ''.join(
        f'_{char}' if i > start and char.isupper() else char
        for i, char in enumerate(pascamel)
    ).lower()
    assert snake.isidentifier()
    assert snake[0] != '_'
    assert snake[-1] != '_'

    return snake


def is_dunder(name: str, /) -> bool:
    """Whether the name is a valid `__dunder_name__`."""
    return (
        len(name) > 4
        and name.isidentifier()
        and name.islower()
        and name[:2] == name[-2:] == '__'
        and name[2].isalpha()
        and name[-3].isalpha()
    )
