"""
This type-tests for "true positive" and "false negative" (i.e. type I and
II errors), assuming that i.f.f. there's a typing error, pyright catches
it.

The "trick" here, is the use `type: ignore` comments, which are *only*
allowed when there's an actual error (shoutout to ruff).
This is a simple and effective way to test for false positive, by
temporarily that invalid `optype` use will actually cause the typechecker
(we only consider (based)pyright at the moment) to complain.
"""

import types
from collections.abc import Collection, Iterable, Iterator
from typing import Any, TypeAlias, TypeVar

import pytest

import optype as op
from optype._core import _can
from optype.inspect import get_protocol_members, get_protocols

_T_ReIter = TypeVar("_T_ReIter")
CanReIter: TypeAlias = op.CanIter[op.CanIterSelf[_T_ReIter]]


def test_can_add_self_int() -> None:
    """Ensure that `builtins.int` is assignable to `CanAddSelf`."""
    x: int = 42
    assert isinstance(x, op.CanAddSelf)
    assert issubclass(int, op.CanAddSelf)

    a1: op.CanAddSelf[Any] = x
    a2: op.CanAddSelf[int] = x

    r0: op.CanAddSelf[float] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    r1: op.CanAddSelf[object] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_can_add_same_int() -> None:
    """Ensure that `builtins.int` is assignable to `CanAddSame`."""
    x: int = 42
    assert isinstance(x, op.CanAddSame)
    assert issubclass(int, op.CanAddSame)

    a0: op.CanAddSame = x
    a1: op.CanAddSame[Any] = x
    a2: op.CanAddSame[int] = x
    a3: op.CanAddSame[bool] = x

    r0: op.CanAddSame[float] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    r1: op.CanAddSame[object] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_iadd() -> None:
    """
    The `builtins.list` type is the only builtin collection that implements
    `__iadd__`.
    """
    some_list: list[int] = [42]

    x_iadd: op.CanIAdd[Iterable[int], list[int]] = some_list
    x_iadd_wrong_in: op.CanIAdd[str, list[int]] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_iadd_wrong_in_val: op.CanIAdd[CanReIter[str], list[int]] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_iadd_wrong_out: op.CanIAdd[CanReIter[int], list[str]] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, op.CanIAdd)

    x_iadd_self: op.CanIAddSelf[Iterable[int]] = some_list
    x_iadd_self_wrong: op.CanIAddSelf[str] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_iadd_self_wrong_val: op.CanIAddSelf[CanReIter[str]] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, op.CanIAddSelf)


def test_imul() -> None:
    """
    The `builtins.list` type is the only builtin collection that implements
    `__imul__`.
    """
    some_list: list[int] = [42]

    x_imul: op.CanIMul[int, list[int]] = some_list
    x_imul_wrong_in: op.CanIMul[str, list[int]] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_imul_wrong_out: op.CanIMul[int, list[str]] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, op.CanIMul)

    x_imul_self: op.CanIMulSelf[int] = some_list
    x_imul_self_wrong: op.CanIMulSelf[str] = some_list  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, op.CanIMulSelf)


def test_isub() -> None:
    """
    The `builtins.set` type is the only builtin collection that implements
    `__isub__`.
    """
    some_set: set[int] = {42}

    x_isub: op.CanISub[set[int], set[int]] = some_set
    x_isub_wrong_in: op.CanISub[int, set[int]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_isub_wrong_out: op.CanISub[set[int], set[str]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanISub)

    x_isub_self: op.CanISubSelf[set[int]] = some_set
    x_isub_self_wrong: op.CanISubSelf[str] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanISubSelf)


def test_iand() -> None:
    """
    The `builtins.set` type is the only builtin collection that implements
    `__iand__`.

    Note that the `set.__iand__` and `set.__isub__` method signatures are
    equivalent, namely `(Self, collections.abc.Set[object]) -> Self`.

    But unfortunately, there is currently no *clean* way to avoid code
    duplication between these tests.
    """
    some_set: set[int] = {42}

    x_iand: op.CanIAnd[set[int], set[int]] = some_set
    x_iand_wrong_in: op.CanIAnd[int, set[int]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_iand_wrong_out: op.CanIAnd[set[int], set[str]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanIAnd)

    x_iand_self: op.CanIAndSelf[set[int]] = some_set
    x_iand_self_wrong: op.CanIAndSelf[str] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanIAndSelf)


def test_ixor() -> None:
    """
    The `builtins.set` type is the only builtin collection that implements
    `__ixor__`.
    Its method signature is `(Self[~T], collections.abc.Set[~T]) -> Self[~T]`.
    """
    some_set: set[bool] = {False, True}

    x_ixor: op.CanIXor[set[bool], set[bool]] = some_set
    x_ixor_wrong_in: op.CanIXor[set[str], set[bool]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_ixor_wrong_out: op.CanIXor[set[bool], set[str]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanIXor)

    x_ixor_self: op.CanIXorSelf[set[bool]] = some_set
    x_ixor_self_wrong: op.CanIXorSelf[set[str]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanIXorSelf)


def test_ior() -> None:
    """
    Both the `builtins.set` and the `builtins.dict` standard collection types
    implement `__ior__` method.
    """
    some_set: set[float] = {1 / 137}

    x_ior: op.CanIOr[set[float], set[float]] = some_set
    x_ior_wrong_in: op.CanIOr[set[complex], set[float]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_ior_wrong_out: op.CanIOr[set[float], set[int]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanIOr)

    x_ior_self: op.CanIOrSelf[set[float]] = some_set
    x_ior_self_wrong: op.CanIOrSelf[set[str]] = some_set  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, op.CanIOrSelf)

    some_dict: dict[bytes, int] = {b"answer": 0x2A}

    y_ior: op.CanIOr[dict[bytes, int], dict[bytes, int]] = some_dict
    y_ior_wrong_in: op.CanIOr[dict[str, int], dict[bytes, int]] = some_dict  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    y_ior_wrong_out: op.CanIOr[dict[bytes, int], dict[str, int]] = some_dict  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    assert isinstance(some_dict, op.CanIOr)

    y_ior_self: op.CanIOrSelf[dict[bytes, int]] = some_dict
    y_ior_self_wrong: op.CanIOrSelf[dict[str, int]] = some_dict  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_dict, op.CanIOrSelf)


def test_can_iadd_same_list_accept() -> None:
    """Ensure that `builtins.list` is assignable to `CanAddSame`."""
    # acceptance tests (true negatives)
    x: list[int] = [42]
    assert isinstance(x, op.CanIAddSame)
    assert issubclass(list, op.CanIAddSame)

    a0: op.CanIAddSame = x
    a1: op.CanIAddSame[Any] = x
    a2: op.CanIAddSame[list[int]] = x
    a3: op.CanIAddSame[bytes] = x


def test_can_iadd_same_list_reject() -> None:
    """Ensure that `builtins.int` is **not** assignable to `CanIAddSame`."""
    x: int = 42
    assert not isinstance(x, op.CanIAddSame)
    assert not issubclass(int, op.CanIAddSame)

    r0: op.CanIAddSame = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    r1: op.CanIAddSame[Any] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    r2: op.CanIAddSame[int] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_can_iter_int() -> None:
    # one can *not* iter and int (pun intended)
    value: int = 42 * 1337

    iter_next_int: op.CanIter[op.CanNext[int]] = value  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    iter_self_int: op.CanIterSelf[int] = value  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    assert not isinstance(value, op.CanIter)
    assert not isinstance(value, op.CanNext)
    assert not isinstance(value, op.CanIterSelf)


@pytest.mark.parametrize(
    "value",
    [
        "spam",
        (("spam", "ham"),),
        (["spam", "ham"],),
        ({"spam", "ham"},),
        ({"spam": "food", "ham": 0xF00D},),
    ],
)
def test_can_iter_collection_str(
    value: (str | tuple[str, ...] | list[str] | set[str] | dict[str, object]),
) -> None:
    # sanity checks
    assert isinstance(value, Collection)
    assert not isinstance(value, Iterator)

    value_iter_next: op.CanIter[op.CanNext[str]] = value
    value_iter_next_wrong: op.CanIter[op.CanNext[int]] = value  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    value_iter_self: op.CanIterSelf[str] = value  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    value_iter_iter_self: op.CanIter[op.CanIterSelf[str]] = value

    # strings are iterables of strings; making them infinitely nested
    value_2_iter_next: op.CanIter[op.CanNext[op.CanIter[op.CanNext[str]]]] = value
    value_3_iter_next: op.CanIter[
        op.CanNext[op.CanIter[op.CanNext[op.CanIter[op.CanNext[str]]]]]
    ] = value

    assert isinstance(value, op.CanIter)
    assert not isinstance(value, op.CanNext)
    assert not isinstance(value, op.CanIterSelf)

    ivalue = iter(value)
    ivalue_iter_next: op.CanIter[op.CanNext[str]] = ivalue
    ivalue_iter_next_wrong: op.CanIter[op.CanNext[int]] = ivalue  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    ivalue_iter_self_str: op.CanIterSelf[str] = ivalue
    ivalue_iter_iter_self: op.CanIter[op.CanIterSelf[str]] = ivalue

    assert isinstance(ivalue, op.CanIter)
    assert isinstance(ivalue, op.CanNext)
    assert isinstance(ivalue, op.CanIterSelf)


class UnsliceableSequence:
    def __getitem__(self, index: int, /) -> str:
        # note how `index` is an invariant `int`.
        return str(index)

    def __len__(self, /) -> int:
        # wishful thinking; this will raise a `ValueError` at runtime
        return int("inf")


def test_unsliceable_sequence() -> None:
    seq_int_str: op.CanSequence[int, str] = UnsliceableSequence()
    seq_wrong_str: op.CanSequence[slice, str] = UnsliceableSequence()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(UnsliceableSequence, op.CanSequence)


@pytest.mark.parametrize(
    "x",
    [
        "spam ham",
        (("spam", "ham"),),
        (["spam", "ham"],),
        UnsliceableSequence(),
    ],
)
def test_can_sequence_sequence_str(
    x: str | tuple[str, ...] | list[str] | UnsliceableSequence,
) -> None:
    x_sequence_int_str: op.CanSequence[int, str] = x
    x_sequence_wrong1: op.CanSequence[slice, str] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    x_sequence_wrong2: op.CanSequence[int, bytes] = x  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    assert isinstance(x, op.CanLen)
    assert isinstance(x, op.CanGetitem)
    assert isinstance(x, op.CanSequence)


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

    base = name.removesuffix("Same")
    assert hasattr(op, f"{base}")
    assert hasattr(op, f"{base}Self")
    assert hasattr(op, f"{base}Same")

    stem = base.removeprefix("Can").removesuffix("Same")
    if iop := (stem[0] == "I" and stem[1].isupper()):
        stem = stem[1:]

    assert hasattr(op, f"Can{stem}")
    assert hasattr(op, f"CanI{stem}")
    assert hasattr(op, f"CanR{stem}")
    assert hasattr(op, f"Can{stem}Self")
    assert hasattr(op, f"CanI{stem}Self")
    assert hasattr(op, f"CanR{stem}Self")
    assert hasattr(op, f"Can{stem}Same")
    assert hasattr(op, f"CanI{stem}Same")
    assert not hasattr(op, f"CanR{stem}Same")

    members_same = get_protocol_members(cls)
    assert len(members_same) == 1, members_same

    members_base = get_protocol_members(getattr(op, f"{base}"))
    members_self = get_protocol_members(getattr(op, f"{base}Self"))

    assert members_same == members_self
    assert members_same == members_base
