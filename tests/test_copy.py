# ruff: noqa: F841
import sys
from datetime import date
from fractions import Fraction
from typing import Any, Final

import pytest

import optype as opt
from optype.inspect import get_protocol_members, is_runtime_protocol


require_py313: Final = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason='requires python>=3.13',
)


def get_type_params(cls: type) -> tuple[Any, ...]:
    return getattr(cls, '__parameters__', ())


@pytest.mark.parametrize(
    'cls',
    [getattr(opt.copy, k) for k in opt.copy.__all__ if not k.endswith('Self')],
)
def test_protocols(cls: type):
    # ensure correct name
    assert cls.__module__ == 'optype.copy'
    assert cls.__name__ == cls.__qualname__
    assert cls.__name__.startswith('Can')

    # ensure exported
    assert cls.__name__ in opt.copy.__all__

    # ensure each `Can{}` has a corresponding `Can{}Self` sub-protocol
    cls_self = getattr(opt.copy, f'{cls.__name__}Self')
    assert cls_self is not cls
    assert cls_self.__name__ in opt.copy.__all__
    assert issubclass(cls_self, cls)
    assert len(get_type_params(cls)) == len(get_type_params(cls_self)) + 1

    # ensure single-method protocols
    assert len(get_protocol_members(cls)) == 1
    assert len(get_protocol_members(cls_self)) == 1

    # ensure @runtime_checkable
    assert is_runtime_protocol(cls)
    assert is_runtime_protocol(cls_self)


def test_can_copy():
    a = Fraction(1, 137)

    a_copy: opt.copy.CanCopy[Fraction] = a
    a_copy_self: opt.copy.CanCopySelf = a

    assert isinstance(a, opt.copy.CanCopy)
    assert isinstance(a, opt.copy.CanCopySelf)


def test_can_deepcopy():
    a = Fraction(1, 137)

    a_copy: opt.copy.CanDeepcopy[Fraction] = a
    a_copy_self: opt.copy.CanDeepcopySelf = a

    assert isinstance(a, opt.copy.CanDeepcopy)
    assert isinstance(a, opt.copy.CanDeepcopySelf)


@require_py313
def test_can_replace():
    d = date(2024, 10, 1)

    # this seemingly redundant `if` statement prevents pyright errors
    if sys.version_info >= (3, 13):
        d_replace: opt.copy.CanReplace[opt.CanIndex, date] = d
        d_copy_self: opt.copy.CanReplaceSelf[opt.CanIndex] = d

    assert isinstance(d, opt.copy.CanReplace)
    assert isinstance(d, opt.copy.CanReplaceSelf)
