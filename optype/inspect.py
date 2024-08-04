# ruff: noqa: A005
from __future__ import annotations

import inspect
import sys
from types import GenericAlias, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    get_args as _get_args,
)


if sys.version_info >= (3, 13):
    from typing import (
        TypeAliasType,
        TypeIs,
        is_protocol,
        overload,
    )
else:
    from typing_extensions import (
        TypeAliasType,
        TypeIs,  # noqa: TCH002
        is_protocol,
        overload,
    )


if TYPE_CHECKING:
    from collections.abc import Callable as CanCall
    from types import ModuleType

    from .types import WrappedFinalType
    from .typing import AnyIterable

from ._can import CanGetitem, CanIter, CanLen
from .types import AnnotatedAlias, GenericType, LiteralAlias, UnionAlias


__all__ = (
    'get_args',
    'get_protocol_members',
    'get_protocols',
    'is_final',
    'is_generic_alias',
    'is_iterable',
    'is_protocol',
    'is_runtime_protocol',
    'is_union_type',
)


def is_iterable(obj: object, /) -> TypeIs[AnyIterable]:
    """
    Check whether the object can be iterated over, i.e. if it can be used in
    a `for` loop, or if it can be passed to `builtins.iter`.

    Note:
        Contrary to popular *belief*, this isn't limited to objects that
        implement `__iter___`, as suggested by the name of
        `collections.abc.Iterable`.

        Sequence-like objects that implement `__getitem__` for consecutive
        `int` keys that start at `0` (or raise `IndexEeror` if out of bounds),
        can also be used in a `for` loop.
        In fact, all builtins that accept "iterables" also accept these
        sequence-likes at runtime.

    See also:
        - [`optype.types.Iterable`][optype.types.Iterable]
    """
    if isinstance(obj, type):
        # workaround the false-positive bug in `@runtime_checkable` for types
        return False

    if isinstance(obj, CanIter):
        return True
    if isinstance(obj, CanGetitem):
        # check if obj is a sequence-like
        if isinstance(obj, CanLen):
            return True

        # not all sequence-likes implement __len__, e.g. `ctypes.pointer`
        try:
            obj[0]
        except (IndexError, StopIteration):
            pass
        except (KeyError, ValueError, TypeError):
            return False

        return True

    return False


@overload
def is_final(final_cls_or_method: WrappedFinalType, /) -> Literal[True]: ...
@overload
def is_final(cls: type, /) -> bool: ...
@overload
def is_final(method: CanCall[..., Any], /) -> bool: ...
@overload
def is_final(prop: property, /) -> bool: ...
@overload
def is_final(
    clsmethod: classmethod[Any, ..., Any] | staticmethod[..., Any],
    /,
) -> bool: ...
def is_final(
    arg: (
        WrappedFinalType
        | type
        | CanCall[..., Any]
        | property
        | classmethod[Any, ..., Any]
        | staticmethod[..., Any]
    ),
    /,
) -> bool:
    """
    Check if the type, method, classmethod, staticmethod, or property, is
    decorated with `@typing.final` or `@typing_extensions.final`.

    IMPORTANT: A final `@property` won't be recognized unless `@final` is
    applied before the `@property` decorator, i.e. directly on the method.

    Do this:

    ```python
    class Europe:
        @property
        @final
        def countdown(self): ...
    ```

    Don't do this:

    ```python
    class Europe:
        @final
        @property
        def countdown(self): ...
    ```

    The reason for this is that `builtins.property.__final__` cannot be
    written to, so `@final` won't be able to set `__final__ = True` on it.
    Only the getter method of a `@property` is checked.

    NOTE: Accessing a `classmethod` or `staticmethod` of a class can't be done
    through regular attribute access, but should be done with
    `inspect.getattr_static`.
    But unlike `@property`, it doesn't matter in which order you use the
    `@classmethod` and `@staticmethod` decorators and `@final`.
    """
    if callable(arg):  # classes are also callable
        if getattr(arg, '__final__', False):
            return True
        return isinstance(arg, staticmethod) and is_final(arg.__wrapped__)
    if isinstance(arg, property) and arg.fget is not None:
        return is_final(arg.fget)
    if isinstance(arg, classmethod):
        return getattr(arg, '__final__', False) or is_final(arg.__wrapped__)

    return False


def is_runtime_protocol(cls: type, /) -> bool:
    """
    Check if `cls` is a `typing[_extensions].Protocol` that's decorated with
    `typing[_extensions].runtime_checkable`.
    """
    return is_protocol(cls) and getattr(cls, '_is_runtime_protocol', False)


def is_union_type(tp: Any, /) -> TypeIs[UnionType | UnionAlias]:
    return isinstance(tp, UnionType | UnionAlias)


def is_generic_alias(tp: Any, /) -> TypeIs[GenericType | GenericAlias]:
    return isinstance(tp, GenericType | GenericAlias)


def _get_alias(tp: Any, /) -> Any:
    seen: set[TypeAliasType] = set()
    for _ in range(sys.getrecursionlimit()):
        if isinstance(tp, TypeAliasType):
            seen.add(tp)
            tp = tp.__value__
            if tp in seen:
                raise RecursionError('type alias of itself')
            continue
        if isinstance(tp, AnnotatedAlias):
            assert len(tp.__args__) == 1
            tp = tp.__args__[0]
            continue
        return tp

    raise RecursionError


def get_args(tp: Any, /) -> tuple[Any, ...]:
    """
    A less broken implementation of `typing[_extensions].get_args()` that

    - unpacks `Annotated` and `TypeAliasType`,
    - recursively flattens unions / nested `Literal`s, and
    - raises `TypeError` if `tp` if isn't a generic type (alias).
    """
    if isinstance(tp, str):
        raise NotImplementedError('str')

    _raise = True
    if isinstance(tp, AnnotatedAlias | TypeAliasType):
        tp = _get_alias(tp)
        _raise = False

    if isinstance(tp, UnionType | UnionAlias | LiteralAlias):
        args: list[Any] = []
        for arg in tp.__args__:
            if isinstance(arg, TypeAliasType | AnnotatedAlias):
                arg = _get_alias(arg)  # noqa: PLW2901
            if isinstance(arg, UnionType | UnionAlias | LiteralAlias):
                args.extend(get_args(arg))
            else:
                args.append(arg)
        return tuple(args)

    if hasattr(tp, '__origin__') and hasattr(tp, '__args__'):
        return _get_args(tp)

    if _raise and not isinstance(tp, type):
        raise TypeError(repr(tp))

    return ()


def get_protocol_members(cls: type, /) -> frozenset[str]:
    """
    A variant of `typing[_extensions].get_protocol_members()` that

    - doesn't hide `__dict__` or `__annotations__`,
    - doesn't add a `__hash__` if there's an `__eq__` method, and
    - doesn't include methods of base types from different module.
    """
    if not is_protocol(cls):
        msg = f'{cls!r} is not a protocol'
        raise TypeError(msg)

    module_blacklist = {'typing', 'typing_extensions'}
    annotations, module = cls.__annotations__, cls.__module__
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
                    v.__module__ in module_blacklist
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


def get_protocols(
    module: ModuleType,
    /,
    private: bool = False,
) -> frozenset[type]:
    """Return the protocol types within the given module."""
    if private:
        members = dir(module)
    elif hasattr(module, '__all__'):
        members = module.__all__
    else:
        members = (k for k in dir(module) if not k.startswith('_'))

    return frozenset({
        cls for name in members
        if is_protocol(cls := getattr(module, name))
    })
