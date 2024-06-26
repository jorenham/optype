import sys
from types import ModuleType

from optype import CanBool, CanLt


if sys.version_info >= (3, 13):
    from typing import is_protocol
else:
    from typing_extensions import is_protocol


__all__ = (
    'get_callable_members',
    'is_dunder',
    'is_protocol',
    'pascamel_to_snake',
)


def get_callable_members(module: ModuleType, /) -> frozenset[str]:
    """
    Return the public callables of a module, that aren't protocols, and
    """
    module_blacklist = {'typing', 'typing_extensions'}
    return frozenset({
        name for name in dir(module)
        if not name.startswith('_')
        and callable(cls := getattr(module, name))
        and not is_protocol(cls)
        and getattr(module, name).__module__ not in module_blacklist
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
