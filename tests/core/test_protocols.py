import sys
import types
from typing import Any, cast

if sys.version_info >= (3, 13):
    from typing import is_protocol
else:
    from typing_extensions import is_protocol

import pytest

import optype as op
from optype import _utils
from optype._core import _can, _do, _does, _has, _just
from optype.inspect import (
    get_protocol_members,
    get_protocols,
    is_protocol,
    is_runtime_protocol,
)


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


def _pascamel_to_snake(pascamel: str, start: op.CanLt[int, op.CanBool] = 0, /) -> str:
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


def _get_protocols_with_suffix(
    module: types.ModuleType,
    suffix: str,
    /,
    private: bool = False,
) -> frozenset[type]:
    """Get all protocols in a module with a specific suffix."""
    return frozenset(
        cls for cls in get_protocols(module, private) if cls.__name__.endswith(suffix)
    )


def test_all_public() -> None:
    """
    Verify that all of protocols from the private `_core.*` submodules are expected
    in `optype.__all__`.
    """
    protos_all = get_protocols(op)
    protos_can = get_protocols(_can)
    protos_has = get_protocols(_has)
    protos_does = get_protocols(_does)
    protos_just = get_protocols(_just)

    assert protos_can | protos_has | protos_does | protos_just == protos_all


@pytest.mark.parametrize("cls", get_protocols(_can))
def test_can_runtime_checkable(cls: type) -> None:
    """Ensure that all `Can*` protocols are `@runtime_checkable`."""
    assert is_runtime_protocol(cls)


@pytest.mark.parametrize("cls", _get_protocols_with_suffix(_can, "Same"))
def test_can_same_self(cls: type) -> None:
    """
    Ensure that for each `Can{}Same` protocol there also exist the `Can{}Self`, `Can{}`,
    `CanR{}Self`, `CanR{}`, `CanI{}Self`, and `CanI{}` protocols, where `{}` is the
    camelcase operation name.
    The `Can{}Same`, `Can{}Self`, and `Can{}` protocols should also have the same single
    member method.
    """
    name = cls.__name__
    assert name.startswith("Can")
    assert name.endswith("Same")

    stem = name.removeprefix("Can").removesuffix("Same")
    assert hasattr(op, f"Can{stem}")
    assert hasattr(op, f"CanR{stem}")
    assert hasattr(op, f"CanI{stem}")
    assert hasattr(op, f"Can{stem}Self")
    assert hasattr(op, f"CanR{stem}Self")
    assert hasattr(op, f"CanI{stem}Self")
    assert hasattr(op, f"Can{stem}Same")
    assert not hasattr(op, f"CanR{stem}Same")
    assert not hasattr(op, f"CanI{stem}Same")

    members_same = get_protocol_members(cls)
    assert len(members_same) == 1, members_same

    members_base = get_protocol_members(getattr(op, f"Can{stem}"))
    members_self = get_protocol_members(getattr(op, f"Can{stem}Self"))

    assert members_same == members_self
    assert members_same == members_base


def test_can_add_same_int() -> None:
    """Ensure that `builtins.int` is assignable to `CanAddSame`."""
    assert issubclass(int, op.CanAddSame)

    x: int = 42
    assert isinstance(x, op.CanAddSame)

    y0: op.CanAddSame = x
    y1: op.CanAddSame[Any] = x
    y2: op.CanAddSame[int] = x

    z1: op.CanAddSelf[Any] = x
    z2: op.CanAddSelf[int] = x


@pytest.mark.parametrize("cls", get_protocols(_has))
def test_has_runtime_checkable(cls: type) -> None:
    """Ensure that all `Has*` protocols are `@runtime_checkable`."""
    assert is_runtime_protocol(cls)


@pytest.mark.parametrize("cls", get_protocols(_does))
def test_does_not_runtime_checkable(cls: type) -> None:
    """Ensure that all `Does*` protocols are **not** `@runtime_checkable`."""
    assert not is_runtime_protocol(cls)


def test_num_does_eq_num_do() -> None:
    num_does = len(set_does := get_protocols(_does))
    num_do = len(set_do := _utils.get_callables(_do))

    assert not {t.__name__ for t in set_does if not t.__name__.startswith("Does")}
    assert not {k for k in set_do if not k.startswith("do_")}

    assert num_does == num_do, {k[3:] for k in set_do} - {
        t.__name__[4:].lower() for t in set_does
    }


@pytest.mark.parametrize("cls", get_protocols(_does))
def test_does_has_do(cls: type) -> None:
    """Ensure that all `Does*` protocols have a corresponding `do_` op."""
    name = cls.__name__.removeprefix("Does")
    assert name != cls.__name__

    do_name = f"do_{_pascamel_to_snake(name, 1)}"
    do_op = cast("op.CanCall[..., object] | None", getattr(_do, do_name, None))
    assert do_op is not None, do_name
    assert callable(do_op), do_name


@pytest.mark.parametrize("cls", get_protocols(_can) | get_protocols(_has))
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
    parents = [parent for parent in cls.mro()[1:] if is_protocol(parent)]
    if parents:
        overridden = {
            member
            for member in members
            if callable(f := getattr(cls, member)) and getattr(f, "__override__", False)
        }
        own_members = members - overridden
    else:
        own_members = members

    member_count = len(members)
    own_member_count = len(own_members)

    # this test should probably be split up...

    if member_count > min(1, own_member_count):
        # ensure len(parent protocols) == len(members) (including inherited)
        assert member_count == len(parents), (own_members, parents)

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

        if prefix == "Can":
            # strip the `Self` or `Same` suffix
            stem = stem.removesuffix("Self").removesuffix("Same")

        # the `1` arg ensures that any potential leading `A`, `I` or `R` chars
        # won't have a `_` directly after (i.e. considers `stem[:2].lower()`).
        member_predict = _pascamel_to_snake(stem, 1)
        member_expect = next(iter(members))

        # prevent comparing apples with oranges: paint the apples orange!
        if _is_dunder(member_expect):
            member_predict = f"__{member_predict}__"

        assert member_predict == member_expect
