import functools
import inspect
from types import ModuleType
from typing import Protocol, cast

import pytest

import optype
import optype._can
import optype._do
import optype._does
import optype._has


def _is_protocol(cls: type) -> bool:
    """Based on `typing_extensions.is_protocol`."""
    return (
        isinstance(cls, type)
        and cls is not Protocol
        and issubclass(cls, Protocol)
        and getattr(cls, '_is_protocol', False)
    )


def _is_runtime_protocol(cls: type) -> bool:
    """Check if `cls` is a `@runtime_checkable` `typing.Protocol`."""
    return _is_protocol(cls) and getattr(cls, '_is_runtime_protocol', False)


@functools.cache
def _get_protocol_members(cls: type) -> frozenset[str]:
    """
    A variant of `typing_extensions.get_protocol_members()` that doesn't
    hide e.g. `__dict__` and `__annotations`, or adds `__hash__` if there's an
    `__eq__` method.
    Does not return method names of base classes defined in another module.
    """
    assert _is_protocol(cls)

    module = cls.__module__
    annotations = cls.__annotations__

    members = annotations.keys() | {
        name for name, v in vars(cls).items()
        if (
            callable(v) and (
                v.__module__ == module or (
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
        if _is_protocol(supercls):
            members |= _get_protocol_members(supercls)

    return frozenset(members)


def _get_protocols(module: ModuleType) -> frozenset[type]:
    """Return the public protocol types within the given module."""
    return frozenset({
        cls for name in dir(module)
        if not name.startswith('_')
        and _is_protocol(cls := getattr(module, name))
    })


def _get_callable_members(module: ModuleType) -> frozenset[str]:
    """Return the public protocol types within the given module."""
    return frozenset({
        name for name in dir(module)
        if not name.startswith('_')
        and callable(getattr(module, name))
    })


def test_all_public():
    """
    Ensure all of protocols in `optype._can` and `optype._has` are in
    `optype.__all__`.
    """
    protocols_all = _get_protocols(optype)
    protocols_can = _get_protocols(optype._can)
    protocols_has = _get_protocols(optype._has)
    protocols_does = _get_protocols(optype._does)

    assert protocols_can | protocols_has | protocols_does == protocols_all


@pytest.mark.parametrize('cls', _get_protocols(optype._can))
def test_can_runtime_checkable(cls: type):
    """Ensure that all `Can*` protocols are `@runtime_checkable`."""
    assert _is_runtime_protocol(cls)


@pytest.mark.parametrize('cls', _get_protocols(optype._has))
def test_has_runtime_checkable(cls: type):
    """Ensure that all `Has*` protocols are `@runtime_checkable`."""
    assert _is_runtime_protocol(cls)


@pytest.mark.parametrize('cls', _get_protocols(optype._does))
def test_does_not_runtime_checkable(cls: type):
    """Ensure that all `Does*` protocols are **not** `@runtime_checkable`."""
    assert not _is_runtime_protocol(cls)


def test_num_does_eq_num_do():
    num_does = len(_get_protocols(optype._does))
    num_do = len(_get_callable_members(optype._do))
    assert num_does == num_do


@pytest.mark.parametrize('cls', _get_protocols(optype._does))
def test_does_has_do(cls: type):
    """Ensure that all `Does*` protocols have a corresponding `do_` op."""
    name = cls.__name__.removeprefix('Does')
    assert name != cls.__name__

    do_name = f'do_{name.lower()}'
    do_op = getattr(optype._do, do_name, None)
    assert do_op is not None, do_name
    assert callable(do_op), do_name


@pytest.mark.parametrize(
    'cls',
    _get_protocols(optype._can) | _get_protocols(optype._has),
)
def test_name_matches_dunder(cls: type):
    """
    Ensure that each single-member `Can*` and `Has*` name matches its member,
    and that each multi-member optype does not have more members than it has
    super optypes.
    """
    prefix = cls.__module__.rsplit('.', 1)[1].removeprefix('_').title()
    assert prefix in {'Can', 'Has'}

    name = cls.__name__
    assert name.startswith(prefix)

    members = _get_protocol_members(cls)
    assert members

    member_count = len(members)
    super_count = sum(map(_is_protocol, cls.mro()[1:-1]))

    if member_count > 1:
        assert super_count == member_count
        return

    # convert CamelCase to to snake_case (ignoring the first char, which
    # could be an async (A), augmented (I), or reflected (R) binop name prefix)
    member_expect = ''.join(
        f'_{c}' if i > 1 and c.isupper() else c
        for i, c in enumerate(name.removeprefix(prefix))
    ).lower()
    # sanity checks (a bit out-of-scope, but humankind will probably survive)
    assert member_expect.isidentifier()
    assert '__' not in member_expect
    assert member_expect[0] != '_'
    assert member_expect[-1] != '_'

    # remove potential trailing arity digit
    if member_expect[-1].isdigit():
        member_expect = member_expect[:-1]
    # another misplaced check (ah well, let's hope the extinction event is fun)
    assert not member_expect[-1].isdigit()

    member = next(iter(members))

    if member[:2] == member[-2:] == '__':
        # add some thunder... or was is döner...? wait, no; dunder!.
        member_expect = f'__{member_expect}__'

    assert member == member_expect
