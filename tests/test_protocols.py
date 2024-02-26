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
    is_protocol,
    is_runtime_protocol,
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
    super_count = sum(map(is_protocol, cls.mro()[1:-1]))

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
