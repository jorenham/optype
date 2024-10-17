# mypy: disable-error-code="no-any-explicit"
import sys
from typing import cast

import pytest

import optype as opt
import optype._can
import optype._do
import optype._does
import optype._has
from optype._utils import get_callables
from optype.inspect import (
    get_protocol_members,
    get_protocols,
    is_protocol,
    is_runtime_protocol,
)


if sys.version_info >= (3, 13):
    from typing import is_protocol
else:
    from typing_extensions import is_protocol


def _is_dunder(name: str, /) -> bool:
    """Whether the name is a valid `__dunder_name__`."""
    return (
        len(name) > 4
        and name[:2] == name[-2:] == "__"
        and name[2] != "_"
        and name[-3] != "_"
        and name[2:-2].isidentifier()
        and (name.islower() or name.isupper())
    )


def _pascamel_to_snake(
    pascamel: str,
    start: opt.CanLt[int, opt.CanBool] = 0,
    /,
) -> str:
    """Converts 'CamelCase' or 'pascalCase' to 'snake_case'."""
    assert pascamel.isidentifier()

    snake = "".join(
        f"_{char}" if i > start and char.isupper() else char
        for i, char in enumerate(pascamel)
    ).lower()
    assert snake.isidentifier()
    assert snake[0] != "_"
    assert snake[-1] != "_"

    return snake


def test_all_public() -> None:
    """
    Ensure all of protocols from `optype._can`, `optype._has`, and
    `optype._does` are in `optype.__all__`.
    """
    protocols_all = get_protocols(opt)
    protocols_can = get_protocols(optype._can)
    protocols_has = get_protocols(optype._has)
    protocols_does = get_protocols(optype._does)

    assert protocols_can | protocols_has | protocols_does == protocols_all


@pytest.mark.parametrize("cls", get_protocols(optype._can))
def test_can_runtime_checkable(cls: type) -> None:
    """Ensure that all `Can*` protocols are `@runtime_checkable`."""
    assert is_runtime_protocol(cls)


@pytest.mark.parametrize("cls", get_protocols(optype._has))
def test_has_runtime_checkable(cls: type) -> None:
    """Ensure that all `Has*` protocols are `@runtime_checkable`."""
    assert is_runtime_protocol(cls)


@pytest.mark.parametrize("cls", get_protocols(optype._does))
def test_does_not_runtime_checkable(cls: type) -> None:
    """Ensure that all `Does*` protocols are **not** `@runtime_checkable`."""
    assert not is_runtime_protocol(cls)


def test_num_does_eq_num_do() -> None:
    num_does = len(get_protocols(optype._does))
    num_do = len(get_callables(optype._do))
    assert num_does == num_do


@pytest.mark.parametrize("cls", get_protocols(optype._does))
def test_does_has_do(cls: type) -> None:
    """Ensure that all `Does*` protocols have a corresponding `do_` op."""
    name = cls.__name__.removeprefix("Does")
    assert name != cls.__name__

    do_name = f"do_{_pascamel_to_snake(name, 1)}"
    do_op: opt.CanCall[..., object] | None = getattr(optype._do, do_name, None)
    assert do_op is not None, do_name
    assert callable(do_op), do_name


@pytest.mark.parametrize(
    "cls",
    get_protocols(optype._can) | get_protocols(optype._has),
)
def test_name_matches_dunder(cls: type) -> None:
    """
    Ensure that each single-member `Can*` and `Has*` name matches the name of
    its member, and that each multi-member optype does not have more members
    than it has super optypes. I.e. require at most 1 member (attr, prop or
    method) for each **concrete** Protocol.
    """
    assert cls.__module__ == "optype"

    prefix = cls.__qualname__[:3]
    assert prefix in {"Can", "Has"}

    name = cls.__name__
    assert name.startswith(prefix)

    members = get_protocol_members(cls)
    assert members

    own_members: frozenset[str]
    parents = [
        parent
        for parent in cls.mro()[1:]
        if not parent.__name__.endswith("Self")
        and is_protocol(parent)
    ]
    if parents:
        overridden = {
            member
            for member in members
            if callable(f := cast(object, getattr(cls, member)))
            and getattr(f, "__override__", False)
        }
        own_members = members - overridden
    else:
        own_members = members

    member_count = len(members)
    own_member_count = len(own_members)

    # this test should probably be split up...

    if member_count > min(1, own_member_count):
        # ensure len(parent protocols) == len(members) (including inherited)
        assert member_count == len(parents), own_members

        members_concrete = set(members)
        for parent in parents:
            members_concrete.difference_update(get_protocol_members(parent))

        assert not members_concrete
    else:
        # remove the `Can`, `Has`, or `Does` prefix
        stem = name.removeprefix(prefix)
        # strip the arity digit if exists
        if stem[-1].isdigit():
            stem = stem[:-1]
            assert stem[-1].isalpha()

        # the `1` arg ensures that any potential leading `A`, `I` or `R` chars
        # won't have a `_` directly after (i.e. considers `stem[:2].lower()`).
        member_predict = _pascamel_to_snake(stem, 1)
        member_expect = next(iter(members))

        # prevent comparing apples with oranges: paint the apples orange!
        if _is_dunder(member_expect):
            member_predict = f"__{member_predict}__"

        assert member_predict == member_expect
