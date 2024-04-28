# ruff: noqa: I001
from typing import Any, Final

import numpy as np
import pytest

import optype.numpy as onp
from optype.numpy import _scalar  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
from ..helpers import get_args  # noqa: TID252

_NP_V1: Final[bool] = np.__version__.startswith('1.')


def _get_attr_args(obj: Any, name: str) -> tuple[Any, ...]:
    tp = getattr(obj, name)
    return get_args(tp) or (tp, )


def _get_dtype_info(name: str) -> tuple[
    frozenset[type],
    frozenset[str],
    frozenset[str],
]:
    types = _get_attr_args(onp, f'Any{name}')
    names = _get_attr_args(_scalar, f'Any{name}Name')
    chars = _get_attr_args(_scalar, f'Any{name}Char')
    return frozenset(types), frozenset(names), frozenset(chars)


@pytest.mark.parametrize(
    'name',
    [
        'Bool',
        'UInt8', 'UInt16', 'UInt32', 'UInt64',
        'UByte', 'UShort', 'UIntC', 'UIntP', 'UInt', 'ULong', 'ULongLong',
        'Int8', 'Int16', 'Int32', 'Int64',
        'Byte', 'Short', 'IntC', 'IntP', 'Int', 'Long', 'LongLong',
        'Float16', 'Float32', 'Float64',
        'Half', 'Single', 'Float', 'Double', 'LongDouble',
        'Complex64', 'Complex128',
        'CSingle', 'CDouble', 'CLongDouble',
        'Timedelta64', 'Datetime64',
        'Str',
        'Bytes',
        'Void',
        'Object',
    ],
)
def test_scalars(name: str):
    dtype_expect = np.dtype(name.lower())
    types, names, chars = _get_dtype_info(name)

    assert dtype_expect.type in types

    for arg in types | names | {c for c in chars if c[0] not in '=<>'}:
        dtype = np.dtype(arg)
        assert dtype == dtype_expect


@pytest.mark.parametrize(
    'name',
    [
        'Bool',
        'UInt8', 'UInt16', 'UInt32', 'UInt64',
        'Int8', 'Int16', 'Int32', 'Int64',
        'Float16', 'Float32', 'Float64',
        'Complex64', 'Complex128',
        'Timedelta64', 'Datetime64',
        'Str',
        'Bytes',
        'Void',
        'Object',
    ],
)
def test_scalar_name(name: str):
    dtype_expect = np.dtype(name.lower())
    _, names, _ = _get_dtype_info(name)

    assert dtype_expect.name in names


@pytest.mark.parametrize(
    'name',
    [
        'Bool',
        'UByte', 'UShort', 'ULong',
        'Byte', 'Short', 'Long',
        'Half', 'Single', 'Double',
        'CSingle', 'CDouble',
        'Timedelta64', 'Datetime64',
        'Str',
        'Bytes',
        'Void',
        'Object',
    ],
)
def test_scalar_char(name: str):
    dtype_expect = np.dtype(name.lower())
    _, _, chars = _get_dtype_info(name)

    assert dtype_expect.char in chars
