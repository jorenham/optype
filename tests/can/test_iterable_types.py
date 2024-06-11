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

# pyright: reportUnusedVariable=false
# ruff: noqa: TCH002, F841
from collections.abc import Collection, Iterator
from typing import Any

import pytest

from optype import (
    CanGetitem,
    CanIter,
    CanIterSelf,
    CanLen,
    CanNext,
    CanSequence,
)


def test_can_iter_int():
    # one can *not* iter and int (pun intended)
    value: int = 42 * 1337

    iter_any: CanIter[Any] = value  # pyright: ignore[reportAssignmentType]
    iter_next_any: CanIter[CanNext[Any]] = value  # pyright: ignore[reportAssignmentType]
    iter_next_int: CanIter[CanNext[int]] = value  # pyright: ignore[reportAssignmentType]
    iter_self_int: CanIterSelf[int] = value  # pyright: ignore[reportAssignmentType]

    assert not isinstance(value, CanIter)
    assert not isinstance(value, CanNext)
    assert not isinstance(value, CanIterSelf)


@pytest.mark.parametrize(
    'value',
    [
        'spam',
        (('spam', 'ham'),),
        (['spam', 'ham'],),
        ({'spam', 'ham'},),
        ({'spam': 'food', 'ham': 0xf00d},),
    ],
)
def test_can_iter_collection_str(
    value: (
        str
        | tuple[str, ...]
        | list[str]
        | set[str]
        | dict[str, Any]
    ),
):
    # sanity checks
    assert isinstance(value, Collection)
    assert not isinstance(value, Iterator)

    value_iter_any: CanIter[Any] = value
    value_iter_next_any: CanIter[CanNext[Any]] = value
    value_iter_next: CanIter[CanNext[str]] = value
    value_iter_next_wrong: CanIter[CanNext[int]] = value  # pyright: ignore[reportAssignmentType]
    value_iter_self: CanIterSelf[str] = value  # pyright: ignore[reportAssignmentType]
    value_iter_iter_self: CanIter[CanIterSelf[str]] = value

    # strings are iterables of strings; making them infinitely nested
    value_2_iter_next: CanIter[CanNext[
        CanIter[CanNext[str]]
    ]] = value
    value_3_iter_next: CanIter[CanNext[
        CanIter[CanNext[
            CanIter[CanNext[str]]
        ]]
    ]] = value

    assert isinstance(value, CanIter)
    assert not isinstance(value, CanNext)
    assert not isinstance(value, CanIterSelf)

    ivalue = iter(value)
    ivalue_iter_any: CanIter[Any] = ivalue
    ivalue_iter_next_any: CanIter[CanNext[Any]] = ivalue
    ivalue_iter_next: CanIter[CanNext[str]] = ivalue
    ivalue_iter_next_wrong: CanIter[CanNext[int]] = ivalue  # pyright: ignore[reportAssignmentType]
    ivalue_iter_self_str: CanIterSelf[str] = ivalue
    ivalue_iter_iter_self: CanIter[CanIterSelf[str]] = ivalue

    assert isinstance(ivalue, CanIter)
    assert isinstance(ivalue, CanNext)
    assert isinstance(ivalue, CanIterSelf)


class UnsliceableSequence:
    def __getitem__(self, index: int, /) -> str:
        # note how `index` is an invariant `int`.
        return str(index)

    def __len__(self, /) -> int:
        # wishful thinking; this will raise a `ValueError` at runtime
        return int('inf')


def test_unsliceable_sequence():
    seq_int_str: CanSequence[int, str] = UnsliceableSequence()
    seq_wrong_str: CanSequence[slice, str] = UnsliceableSequence()  # pyright: ignore[reportAssignmentType]
    assert isinstance(UnsliceableSequence, CanSequence)


@pytest.mark.parametrize(
    'x',
    [
        'spam ham',
        (('spam', 'ham'),),
        (['spam', 'ham'],),
        UnsliceableSequence(),
    ],
)
def test_can_sequence_sequence_str(
    x: str | tuple[str, ...] | list[str] | UnsliceableSequence,
):
    x_sequence_any_any: CanSequence[Any, Any] = x
    x_sequence_any_str: CanSequence[Any, str] = x
    x_sequence_int_any: CanSequence[int, Any] = x
    x_sequence_int_str: CanSequence[int, str] = x

    x_sequence_wrong1: CanSequence[slice, str] = x  # pyright: ignore[reportAssignmentType]
    x_sequence_wrong2: CanSequence[int, bytes] = x  # pyright: ignore[reportAssignmentType]

    assert isinstance(x, CanLen)
    assert isinstance(x, CanGetitem)
    assert isinstance(x, CanSequence)
