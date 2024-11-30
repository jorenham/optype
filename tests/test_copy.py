import sys
from datetime import date
from fractions import Fraction
from typing import Final

import pytest

import optype as op
from optype.inspect import get_protocol_members, is_runtime_protocol


require_py313: Final = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="requires python>=3.13",
)


def get_type_params(cls: type) -> tuple[object, ...]:
    return getattr(cls, "__parameters__", ())


@pytest.mark.parametrize(
    "cls",
    [getattr(op.copy, k) for k in op.copy.__all__ if not k.endswith("Self")],
)
def test_protocols(cls: type) -> None:
    # ensure correct name
    assert cls.__module__ == "optype.copy"
    assert cls.__name__ == cls.__qualname__
    assert cls.__name__.startswith("Can")

    # ensure exported
    assert cls.__name__ in op.copy.__all__

    # ensure each `Can{}` has a corresponding `Can{}Self` sub-protocol
    cls_self: type = getattr(op.copy, f"{cls.__name__}Self")
    assert cls_self is not cls
    assert cls_self.__name__ in op.copy.__all__
    assert issubclass(cls_self, cls)
    assert len(get_type_params(cls)) == len(get_type_params(cls_self)) + 1

    # ensure single-method protocols
    assert len(get_protocol_members(cls)) == 1
    assert len(get_protocol_members(cls_self)) == 1

    # ensure @runtime_checkable
    assert is_runtime_protocol(cls)
    assert is_runtime_protocol(cls_self)


def test_can_copy() -> None:
    a = Fraction(1, 137)

    a_copy: op.copy.CanCopy[Fraction] = a
    a_copy_self: op.copy.CanCopySelf = a

    assert isinstance(a, op.copy.CanCopy)
    assert isinstance(a, op.copy.CanCopySelf)


def test_can_deepcopy() -> None:
    a = Fraction(1, 137)

    a_copy: op.copy.CanDeepcopy[Fraction] = a
    a_copy_self: op.copy.CanDeepcopySelf = a

    assert isinstance(a, op.copy.CanDeepcopy)
    assert isinstance(a, op.copy.CanDeepcopySelf)


@require_py313
def test_can_replace() -> None:
    d = date(2024, 10, 1)

    # this seemingly redundant `if` statement prevents pyright errors
    if sys.version_info >= (3, 13):
        d_replace: op.copy.CanReplace[op.CanIndex, date] = d
        d_copy_self: op.copy.CanReplaceSelf[op.CanIndex] = d

    assert isinstance(d, op.copy.CanReplace)
    assert isinstance(d, op.copy.CanReplaceSelf)
