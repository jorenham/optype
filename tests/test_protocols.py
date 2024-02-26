import pytest

import optype
import optype._can
import optype._do
import optype._does
import optype._has
from .helpers import (
    get_callable_members,
    get_protocol_members,
    get_protocols,
    is_dunder,
    is_protocol,
    is_runtime_protocol,
    pascamel_to_snake,
)


def test_all_public():
    """
    Ensure all of protocols from `optype._can`, `optype._has`, and
    `optype._does` are in `optype.__all__`.
    """
    protocols_all = get_protocols(optype)
    protocols_can = get_protocols(optype._can)
    protocols_has = get_protocols(optype._has)
    protocols_does = get_protocols(optype._does)

    assert protocols_can | protocols_has | protocols_does == protocols_all


@pytest.mark.parametrize('cls', get_protocols(optype._can))
def test_can_runtime_checkable(cls: type):
    """Ensure that all `Can*` protocols are `@runtime_checkable`."""
    assert is_runtime_protocol(cls)


@pytest.mark.parametrize('cls', get_protocols(optype._has))
def test_has_runtime_checkable(cls: type):
    """Ensure that all `Has*` protocols are `@runtime_checkable`."""
    assert is_runtime_protocol(cls)


@pytest.mark.parametrize('cls', get_protocols(optype._does))
def test_does_not_runtime_checkable(cls: type):
    """Ensure that all `Does*` protocols are **not** `@runtime_checkable`."""
    assert not is_runtime_protocol(cls)


def test_num_does_eq_num_do():
    num_does = len(get_protocols(optype._does))
    num_do = len(get_callable_members(optype._do))
    assert num_does == num_do


@pytest.mark.parametrize('cls', get_protocols(optype._does))
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
    get_protocols(optype._can) | get_protocols(optype._has),
)
def test_name_matches_dunder(cls: type):
    """
    Ensure that each single-member `Can*` and `Has*` name matches the name of
    its member, and that each multi-member optype does not have more members
    than it has super optypes. I.e. require at most 1 member (attr, prop or
    method) for each **concrete** Protocol.
    """
    prefix = cls.__module__.rsplit('.', 1)[1].removeprefix('_').title()
    assert prefix in {'Can', 'Has'}

    name = cls.__name__
    assert name.startswith(prefix)

    members = get_protocol_members(cls)
    assert members

    member_count = len(members)
    parents = list(filter(is_protocol, cls.mro()[1:]))

    # this test should probably be split up...

    if member_count > 1:
        # ensure #parent protocols == #members (including inherited)
        assert len(parents) == member_count

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
        member_predict = pascamel_to_snake(stem, 1)
        member_expect = next(iter(members))

        # prevent comparing apples with oranges: paint the apples orange!
        if is_dunder(member_expect):
            member_predict = f'__{member_predict}__'

        assert member_predict == member_expect
